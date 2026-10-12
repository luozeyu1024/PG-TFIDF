
from esm.models.esmc import ESMC  
import torch
import torch.nn as nn

from esm.pretrained import LOCAL_MODEL_REGISTRY
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
from esm.sdk.api import ESMProtein, LogitsConfig


class EsmcLastHiddenFeatureExtractor:
    def __init__(self,  compute_cls=True, compute_eos=True, compute_mean=True,
                 compute_segments=False, num_segments=10, device_choose='auto'):
        self.compute_cls = compute_cls
        self.compute_eos = compute_eos
        self.compute_mean = compute_mean
        self.compute_segments = compute_segments
        self.num_segments = num_segments

        if device_choose == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_choose.startswith('cuda'):
            if torch.cuda.is_available():
                device_id = int(device_choose.split(':')[-1]) if ':' in device_choose else 0
                self.device = torch.device(f"cuda:{device_id}")
            else:
                raise TypeError("CUDA is not available.")
        elif device_choose == 'cpu':
            self.device = torch.device("cpu")
        else:
            raise ValueError("Invalid device choice. Use 'auto', 'cpu', or 'cuda[:<id>]'.")

    def get_last_cls_token(self, embeddings):
        return embeddings[:, 0, :]

    def get_last_eos_token(self, embeddings, eos_position):
        return embeddings[:, eos_position, :]

    def get_last_mean_token(self, embeddings, eos_position):
        embeddings = embeddings
        return embeddings[:, 1:eos_position, :].mean(dim=1)

    def get_segment_mean_tokens(self, embeddings, eos_position):
        embeddings = embeddings
        seq_len = eos_position - 1
        segment_size, remainder = divmod(seq_len, self.num_segments)
        segment_means = []

        start = 1
        for i in range(self.num_segments):
            end = start + segment_size + (1 if i < remainder else 0)

            if end > start:
                segment_mean = embeddings[:, start:end, :].mean(dim=1)
            else:
                segment_mean = torch.zeros(embeddings[:, start:start + 1, :].shape,
                                           device=embeddings.device)

            segment_means.append(segment_mean.squeeze().tolist())
            start = end

        return segment_means

    def get_last_hidden_features_combine(self, X_input, sequence_name='sequence'):
        X_input = X_input.reset_index(drop=True)

        sequences = X_input[sequence_name].tolist()

        features_length = {}
        columns = None
        all_results = []
        # Load the registered local model.
        client = ESMC.from_pretrained("custom-esmc-600m",device=self.device)

        with torch.no_grad():
            for idx, seq in tqdm(enumerate(sequences), desc='Sequences inference', total=len(sequences)):

                # Encode the sequence with the ESMC API.
                protein = ESMProtein(sequence=seq)


                # Convert the protein to its tensor representation.
                protein_tensor = client.encode(protein)

                logits_output = client.logits(
                    protein_tensor,
                    LogitsConfig(
                        sequence=True,
                        return_embeddings=True,
                        return_hidden_states=False
                    )
                )

                embeddings = logits_output.embeddings  # Raw embeddings without layer normalization.

                eos_position = len(seq) + 1

                features = []
                if self.compute_cls:
                    cls_features = self.get_last_cls_token(embeddings).squeeze().tolist()
                    features_length.setdefault('cls', len(cls_features))
                    features.extend(cls_features)

                if self.compute_eos:
                    eos_features = self.get_last_eos_token(embeddings, eos_position).squeeze().tolist()
                    features_length.setdefault('eos', len(eos_features))
                    features.extend(eos_features)

                if self.compute_mean:
                    mean_features = self.get_last_mean_token(embeddings, eos_position).squeeze().tolist()
                    features_length.setdefault('mean', len(mean_features))
                    features.extend(mean_features)

                if self.compute_segments:
                    segment_means = self.get_segment_mean_tokens(embeddings, eos_position)
                    for seg, segment_mean in enumerate(segment_means):
                        features.extend(segment_mean)
                        features_length.setdefault(f'segment{seg}_mean', len(segment_mean))

                if columns is None:
                    columns = [f"ESMC_{ftype}{k}" for ftype, length in features_length.items() for k in range(length)]

                result = pd.DataFrame([features], columns=columns, index=[idx])
                all_results.append(result)

                del protein_tensor, logits_output
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        X_outcome = pd.concat(all_results, axis=0)

        print(f'Features dimensions: {features_length}')

        combined_result = pd.concat([X_input, X_outcome], axis=1)
        return combined_result




    def get_all_token_embeddings(self, X_input, sequence_name='sequence', id_column=None):
        """
        Compute all-token embeddings for each protein sequence.

        ``X_input`` is a DataFrame containing a sequence column. The returned
        dictionary maps the requested protein ID, or the row index, to an array
        with shape (n_tokens, embedding_dim).
        """
        embeddings_dict = {}
        X_input = X_input.reset_index(drop=True)

        with torch.no_grad():

            # Load the registered ESMC model.
            client = ESMC.from_pretrained("custom-esmc-600m").to(self.device)

            # Process each protein record.
            for idx, row in tqdm(X_input.iterrows(), total=len(X_input), desc='Protein embeddings extraction'):
                seq = row[sequence_name]
                # Use the supplied protein ID or the row index.
                protein_id = row[id_column] if id_column is not None else idx
                # Construct and encode the protein object.
                protein = ESMProtein(sequence=seq)

                protein_tensor = client.encode(protein)
                # Request sequence logits and embeddings without hidden states.
                logits_output = client.logits(
                    protein_tensor,
                    LogitsConfig(
                        sequence=True,
                        return_embeddings=True,
                        return_hidden_states=False
                    )
                )
                # Shape: (1, seq_length, embed_dim).
                embeddings = logits_output.embeddings
                # Remove the batch dimension.
                embeddings_np = embeddings[0].cpu().numpy()
                # Store the token matrix by protein ID.
                embeddings_dict[protein_id] = embeddings_np

                del protein_tensor, logits_output
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        return embeddings_dict


    def get_combined_features_and_all_token_embeddings(self, X_input, sequence_name='sequence', id_column=None):
        """
        Extract pooled features and all-token embeddings in one pass.

        Returns the input table joined to CLS, EOS, mean and optional segment
        features, plus a dictionary of token matrices keyed by protein ID.
        """
        # Reset the index so output rows remain aligned.
        X_input = X_input.reset_index(drop=True)
        features_list = []
        feature_names = None
        embeddings_dict = {}


        # Load the registered model on the requested device.
        client = ESMC.from_pretrained("custom-esmc-600m", device=self.device)

        # Process each protein record.
        for index, row in tqdm(X_input.iterrows(), total=len(X_input), desc='Processing proteins'):
            seq = row[sequence_name]
            # Use the supplied protein ID or the row index.
            protein_id = row[id_column] if id_column is not None else index

            # Construct and encode the protein object.
            protein = ESMProtein(sequence=seq)
            protein_tensor = client.encode(protein)
            logits_output = client.logits(
                protein_tensor,
                LogitsConfig(
                    sequence=True,
                    return_embeddings=True,
                    return_hidden_states=False
                )
            )
            embeddings = logits_output.embeddings  # Shape: (1, seq_length, embed_dim).
            # Store all token embeddings without the batch dimension.
            embeddings_np = embeddings[0].cpu().numpy()
            embeddings_dict[protein_id] = embeddings_np

            # CLS is at index 0 and EOS follows the sequence tokens.
            eos_position = len(seq) + 1

            # Extract each requested d-dimensional feature.
            d = embeddings.shape[-1]
            current_features = []

            if self.compute_cls:
                # CLS feature at index 0.
                cls_feature = self.get_last_cls_token(embeddings).squeeze().tolist()
                current_features.extend(cls_feature)

            if self.compute_eos:
                # EOS feature at eos_position.
                eos_feature = self.get_last_eos_token(embeddings, eos_position).squeeze().tolist()
                current_features.extend(eos_feature)

            if self.compute_mean:
                # Mean over sequence tokens in [1, eos_position).
                mean_feature = self.get_last_mean_token(embeddings, eos_position).squeeze().tolist()
                current_features.extend(mean_feature)

            if self.compute_segments:
                # Split sequence tokens into num_segments and mean-pool each part:
                #   segment_size = floor((n) / num_segments)
                #   remainder = n mod num_segments
                # Each segment mean is a d-dimensional vector.
                segment_features = self.get_segment_mean_tokens(embeddings, eos_position)
                for seg_feature in segment_features:
                    current_features.extend(seg_feature)

            # Build feature names from the first sample.
            if feature_names is None and current_features:
                names = []
                if self.compute_cls:
                    names.extend([f"ESMC_cls{k}" for k in range(d)])
                if self.compute_eos:
                    names.extend([f"ESMC_eos{k}" for k in range(d)])
                if self.compute_mean:
                    names.extend([f"ESMC_mean{k}" for k in range(d)])
                if self.compute_segments:
                    for seg in range(self.num_segments):
                        names.extend([f"ESMC_segment{seg}_mean{k}" for k in range(d)])
                feature_names = names

            features_list.append(current_features)

            # Release intermediate tensors.
            del protein_tensor, logits_output
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        # Join extracted features to the input table.
        if feature_names is not None and len(features_list) > 0:
            features_df = pd.DataFrame(features_list, columns=feature_names)
            combined_result = pd.concat([X_input, features_df], axis=1)

        else:
            combined_result = X_input.copy()
            print('No features extracted.')

        return combined_result, embeddings_dict

    def extract_residue_features(self, df, embeddings_dict, id_column, position_column, sequence_column=None, mode='strict'):
        """
        Extract features at specified residue positions from precomputed embeddings.

        Positions are one-based sequence coordinates. In strict mode, the token
        count is checked against the protein sequence and its CLS/EOS tokens.
        Fast mode indexes the embedding directly and does not require a sequence.
        """
        results = []
        
        # Process all requested positions for each protein ID.
        for protein_id, group in tqdm(df.groupby(id_column), desc="Processing Proteins"):
            if protein_id not in embeddings_dict:
                print(f"Warning: protein ID {protein_id} was not found; skipping")
                continue
                
            embeddings = embeddings_dict[protein_id]
            

            if mode == 'strict':
                if sequence_column is None:
                    raise ValueError("sequence_column is required in strict mode")
                # Validate sequence length plus CLS and EOS tokens.
                protein_seq = group[sequence_column].iloc[0]

                expected_length = len(protein_seq) + 2
                if embeddings.shape[0] != expected_length:
                    print(f"Error: protein ID {protein_id} has {embeddings.shape[0]} tokens; "
                        f"expected {expected_length}. Check the reference UniProt sequence.")
                    continue


            # Process all residue positions for this protein.
            for _, row in group.iterrows():
                position = int(row[position_column])
                
                # Validate the one-based residue position.
                if mode == 'strict':
                    if position < 1 or position > len(protein_seq):
                        print(f"Warning: residue {position} for protein ID {protein_id} is outside "
                              f"the sequence range 1-{len(protein_seq)}")
                        continue
                
                # Index position directly because token 0 is CLS.
                try:
                    feature_vector = embeddings[position]
                    
                    # Copy the source row and append feature columns.
                    result_row = row.copy()
                    for i, value in enumerate(feature_vector):
                        result_row[f'Residue_feature_{i}'] = value
                        
                    results.append(result_row)
                except IndexError:
                    if mode == 'strict':
                        print(f"Error: could not extract residue {position} for protein ID {protein_id}")
                    else:
                        print(f"Warning: residue {position} for protein ID {protein_id} is outside "
                              f"the embedding range 0-{embeddings.shape[0]-1}")
        
        if not results:
            print("Warning: no residue features were extracted")
            return pd.DataFrame()
            
        return pd.DataFrame(results)



# Demonstration with short example sequences.
if __name__ == '__main__':

    # Set these paths before running the demonstration.
    WEIGHTS_PATH = "/path/to/esmc_600m_2024_12_v0.pth"
    TOKENIZER_PATH = "/path/to/esmc_tokenizer"


    def custom_ESMC_600M(device: torch.device | str = "cpu", use_flash_attn: bool = True) -> nn.Module:
        # Initialize the model architecture.
        with torch.device(device):
            # Load the tokenizer with Hugging Face.
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
            model = ESMC(
                d_model=1152,
                n_heads=18,
                n_layers=36,
                tokenizer=tokenizer,
                use_flash_attn=use_flash_attn,
            ).eval()
        # Load pretrained weights.
        state_dict = torch.load(
            WEIGHTS_PATH,
            map_location=device,
        )
        model.load_state_dict(state_dict)
        return model


    # Register the local model loader.
    LOCAL_MODEL_REGISTRY["custom-esmc-600m"] = custom_ESMC_600M



    # Step 1: extract CLS, EOS, global-mean and segment-mean features.
    extractor = EsmcLastHiddenFeatureExtractor(
        compute_cls=True,
        compute_eos=True,
        compute_mean=True,
        compute_segments=True,
        num_segments=10,device_choose='cpu')

    # Build an example sequence table.
    data = pd.DataFrame({
        'sequence': ['AAAAAAAAAAARAA', 'AAA', 'AARA', 'AAAXA']
    })

    # Join extracted features to the input table.
    result = extractor.get_last_hidden_features_combine(data, sequence_name='sequence')

    print("Combined feature table:")
    print(result)

    # Step 2: extract all-token embeddings, including CLS and EOS.
    data = {
        "protein_id": ["P1", "P2"],
        "sequence": ["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", "GILGFVFTLTVPSER"]
    }
    df = pd.DataFrame(data)

    extractor = EsmcLastHiddenFeatureExtractor(device_choose='cpu')
    embeddings_dict = extractor.get_all_token_embeddings(df, sequence_name='sequence', id_column='protein_id')

    for prot_id, emb in embeddings_dict.items():
        print(f"Protein ID: {prot_id}, Embedding shape: {emb.shape}")


    # Step 3: save and load the embedding dictionary.
    import numpy as np
    
    # Save the embedding dictionary.
    np.savez("embeddings_dict.npz", **embeddings_dict)
    print("Saved embeddings_dict.npz")
    
    # Load the embedding dictionary.
    loaded_data = np.load("embeddings_dict.npz")
    loaded_embeddings_dict = {key: loaded_data[key] for key in loaded_data}
    print("Loaded protein IDs:", list(loaded_embeddings_dict.keys()))
    
    # Step 4: build a table of proteins and residue positions.
    residue_data = {
        "UniportID": ["P1", "P1", "P1", "P2", "P2"],
        "sequence": ["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", 
                    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", "GILGFVFTLTVPSER", "GILGFVFTLTVPSER"],
        "position": [5, 10, 15, 3, 8]
    }
    residue_df = pd.DataFrame(residue_data)
    print("Example residue-position table:")
    print(residue_df)
    
    # Extract residue features in strict mode.
    strict_features = extractor.extract_residue_features(
        residue_df, 
        loaded_embeddings_dict,
        id_column="UniportID",
        position_column="position",
        sequence_column="sequence",
        mode="strict"
    )
    print("\nStrict-mode features (first five columns):")
    if not strict_features.empty:
        print(strict_features.iloc[:, :5])
    
    # Extract residue features in fast mode.
    fast_features = extractor.extract_residue_features(
        residue_df, 
        loaded_embeddings_dict,
        id_column="UniportID",
        position_column="position",
        mode="fast"
    )
    print("\nFast-mode features (first five columns):")
    if not fast_features.empty:
        print(fast_features.iloc[:, :5])

    # Save example tables.
    strict_features.to_excel("strict_features.xlsx", index=False)
    fast_features.to_excel("fast_features.xlsx", index=False)
    print("\nSaved strict_features.xlsx and fast_features.xlsx")







    # Step 5: run combined feature and all-token extraction.

    # Build an example protein table.
    data5 = pd.DataFrame({
        "protein_id": ['P1','P2','P3','P4'],
        'sequence': ['AAAAAAAAAAARAA', 'AAA', 'AARA', 'AAAXA']

    })

    # Enable all feature types and use CPU.
    extractor = EsmcLastHiddenFeatureExtractor(
        compute_cls=True,
        compute_eos=True,
        compute_mean=True,
        compute_segments=True,
        num_segments=10,
        device_choose='cpu'
    )

    # Extract combined features and all-token embeddings.
    combined_result, embeddings_dict = extractor.get_combined_features_and_all_token_embeddings(
        X_input=data5,
        sequence_name='sequence',
        id_column='protein_id'
    )

    # Print the combined table.
    print("Step5 - Combined Features DataFrame:")
    print(combined_result)

    # Print token-embedding shapes.
    print("\nStep5 - Token Embeddings Shapes:")
    for prot_id, emb in embeddings_dict.items():
        print(f"Protein ID: {prot_id}, Embedding Shape: {emb.shape}")
