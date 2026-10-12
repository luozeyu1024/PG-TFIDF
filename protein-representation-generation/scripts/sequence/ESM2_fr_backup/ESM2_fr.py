


import pandas as pd
from packaging import version
import warnings
import numpy as np
from itertools import islice


try:
    import torch
    if version.parse(torch.__version__) < version.parse('1.12.1'):
        warnings.warn("Your torch version is older than 1.12.1 and may not operate correctly.")
except ImportError:
    warnings.warn("Torch not found. Some functions will not be available.")

# Check if tqdm is installed and its version
try:
    import tqdm
    if version.parse(tqdm.__version__) < version.parse('4.63.0'):
        warnings.warn("Your tqdm version is older than 4.63.0 and may not operate correctly.")
    from tqdm import tqdm
except ImportError:
    warnings.warn("tqdm is not installed. Some features may not work as expected.")

# Check if re is installed and its version
try:
    import re
    if version.parse(re.__version__) < version.parse('2.2.1'):
        warnings.warn("Your re version is older than 2.2.1 and may not operate correctly.")
except ImportError:
    warnings.warn("re is not installed. Some features may not work as expected.")

try:
    import sklearn
    if version.parse(sklearn.__version__) < version.parse('1.0.2'):
        warnings.warn("Your sklearn version is older than 1.0.2 and may not operate correctly.")
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    warnings.warn("Sklearn not found. Some functions will not be available.")

##Calculate cls, eos, and amino acid mean representations for each protein one by one
def get_last_hidden_features_single(X_input, tokenizer, model, sequence_name='sequence',device_choose = 'auto'):
    X_input = X_input.reset_index(drop=True)
    X_outcome = pd.DataFrame()


    if device_choose == 'auto':
        # Automatically select device: prefer CUDA if available, otherwise use CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_choose.startswith('cuda'):
        # Support specifying a specific CUDA device number, e.g., 'cuda:0', 'cuda:1'
        if torch.cuda.is_available():
            try:
                # Extract device id, default is 0
                device_id = int(device_choose.split(':')[-1]) if ':' in device_choose else 0
                device = torch.device(f"cuda:{device_id}")
            except ValueError:
                raise ValueError(
                    "Invalid CUDA device format. Use 'cuda' or 'cuda:<id>', where <id> is an integer.")
        else:
            raise TypeError("CUDA is not available. Please check your GPU settings.")
    elif device_choose == 'cpu':
        # Force using CPU
        device = torch.device("cpu")
    else:
        # Handle invalid device_choose value
        raise ValueError("Invalid device choice. Use 'auto', 'cpu', or 'cuda[:<id>]'.")

    model.to(device)
    with torch.no_grad():
        for index, sequence in tqdm(enumerate(X_input[sequence_name]),
                                    desc='one batch for infer time',
                                    total=len(X_input[sequence_name])):
            inputs = tokenizer(sequence, return_tensors="pt").to(device)
            outputs = model(**inputs)

            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
            eos_position = tokens.index(tokenizer.eos_token) if tokenizer.eos_token in tokens else len(tokens) - 1

            last_hidden_state = outputs.hidden_states[-1]

            last_cls_token = last_hidden_state[:, 0, :]
            last_eos_token = last_hidden_state[:, eos_position, :]
            last_mean_token = last_hidden_state[:, 1:eos_position, :].mean(dim=1)

            features = {}

            cls_features = last_cls_token.squeeze().tolist()
            for i, feature in enumerate(cls_features):
                features[f"ESM2_cls{i}"] = feature

            eos_features = last_eos_token.squeeze().tolist()
            for i, feature in enumerate(eos_features):
                features[f"ESM2_eos{i}"] = feature

            mean_features = last_mean_token.squeeze().tolist()
            for i, feature in enumerate(mean_features):
                features[f"ESM2_mean{i}"] = feature

            result = pd.DataFrame.from_dict(features, orient='index').T
            result.index = [index]
            X_outcome = pd.concat([X_outcome, result], axis=0)
            
            del inputs, outputs
            # Only clean and synchronize when using CUDA
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
     
    return X_outcome






   
class Esm2LastHiddenFeatureExtractor:
    def __init__(self, tokenizer, model, compute_cls=True, compute_eos=True, compute_mean=True, compute_segments=False,num_segments=10,device_choose = 'auto'):
        self.tokenizer = tokenizer
        self.model = model
        self.compute_cls = compute_cls
        self.compute_eos = compute_eos
        self.compute_mean = compute_mean
        self.compute_segments = compute_segments
        self.num_segments = num_segments

        if device_choose == 'auto':
            # Automatically select device: prefer CUDA if available, otherwise use CPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_choose.startswith('cuda'):
            # Support specifying a specific CUDA device number, e.g., 'cuda:0', 'cuda:1'
            if torch.cuda.is_available():
                try:
                    # Extract device id, default is 0
                    device_id = int(device_choose.split(':')[-1]) if ':' in device_choose else 0
                    self.device = torch.device(f"cuda:{device_id}")
                except ValueError:
                    raise ValueError(
                        "Invalid CUDA device format. Use 'cuda' or 'cuda:<id>', where <id> is an integer.")
            else:
                raise TypeError("CUDA is not available. Please check your GPU settings.")
        elif device_choose == 'cpu':
            # Force using CPU
            self.device = torch.device("cpu")
        else:
            # Handle invalid device_choose value
            raise ValueError("Invalid device choice. Use 'auto', 'cpu', or 'cuda[:<id>]'.")

    def get_last_hidden_states(self, outputs):
        last_hidden_state = outputs.hidden_states[-1]
        return last_hidden_state

    def get_last_cls_token(self, last_hidden_state):
        return last_hidden_state[:, 0, :]

    def get_last_eos_token(self, last_hidden_state, eos_position):
        return last_hidden_state[:, eos_position, :]

    def get_last_mean_token(self, last_hidden_state, eos_position):
        return last_hidden_state[:, 1:eos_position, :].mean(dim=1)

    def get_segment_mean_tokens(self, last_hidden_state, eos_position):
        seq_len = eos_position - 1
        segment_size, remainder = divmod(seq_len, self.num_segments)
        segment_means = []

        start = 1
        for i in range(self.num_segments):
            end = start + segment_size + (1 if i < remainder else 0)
            
            if end > start:  # Check if the segment has amino acids
                segment_mean = last_hidden_state[:, start:end, :].mean(dim=1)
            else:  # If the segment is empty, create a zero tensor with the same dimensions as the hidden state
                segment_mean = torch.zeros(last_hidden_state[:, start:start+1, :].shape, device=last_hidden_state.device)
            
            segment_means.append(segment_mean.squeeze().tolist())
            start = end

        return segment_means
        
    
    
    # Compute CLS, EOS, residue-mean and segment-mean representations.
    def get_last_hidden_features_combine(self, X_input, sequence_name='sequence', batch_size=32):
        X_input = X_input.reset_index(drop=True)
        
        
        self.model.to(self.device)

        sequence = X_input[sequence_name].tolist()

        features_length = {}  # save the length of different features
        columns = None  # initialize the column names
        all_results = []  # Store all batch results
        with torch.no_grad():
            for i in tqdm(range(0, len(sequence), batch_size), desc='batches for inference'):
                batch_sequences = sequence[i:i+batch_size]
                inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model(**inputs)

                for j in range(len(batch_sequences)):
                    idx = i + j
                    tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][j])
                    eos_position = tokens.index(self.tokenizer.eos_token) if self.tokenizer.eos_token in tokens else len(batch_sequences[j])
                    last_hidden_state = self.get_last_hidden_states(outputs)
                    last_cls_token = self.get_last_cls_token(last_hidden_state[j:j+1]) if self.compute_cls else None
                    last_eos_token = self.get_last_eos_token(last_hidden_state[j:j+1], eos_position) if self.compute_eos else None
                    last_mean_token = self.get_last_mean_token(last_hidden_state[j:j+1], eos_position) if self.compute_mean else None
                    segment_means = self.get_segment_mean_tokens(last_hidden_state[j:j+1], eos_position) if self.compute_segments else None

                    # extract features and add them to DataFrame directly
                    features = []
                    if last_cls_token is not None:
                        cls_features = last_cls_token.squeeze().tolist()
                        if 'cls' not in features_length:
                            features_length['cls'] = len(cls_features)
                        features.extend(cls_features)

                    if last_eos_token is not None:
                        eos_features = last_eos_token.squeeze().tolist()
                        if 'eos' not in features_length:
                            features_length['eos'] = len(eos_features)
                        features.extend(eos_features)

                    if last_mean_token is not None:
                        mean_features = last_mean_token.squeeze().tolist()
                        if 'mean' not in features_length:
                            features_length['mean'] = len(mean_features)
                        features.extend(mean_features)

                    if segment_means is not None:
                        # In the new version, we keep each segment mean as a separate list
                        for seg, segment_mean in enumerate(segment_means):
                            features.extend(segment_mean)
                            if f'segment{seg}_mean' not in features_length:
                                features_length[f'segment{seg}_mean'] = len(segment_mean)

                    # create the column names only for the first item
                    if columns is None:
                        columns = []
                        for feature_type, length in features_length.items():
                            for k in range(length):
                                columns.append(f"ESM2_{feature_type}{k}")

                    # Create DataFrame for this batch
                    result = pd.DataFrame([features], columns=columns, index=[idx])
                    all_results.append(result)

                del inputs, outputs
                
                # Clear and synchronize only when CUDA is active.
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        # Combine all batch results outside the loop
        X_outcome = pd.concat(all_results, axis=0)

        print(f'Features dimensions: {features_length}')

        # Combine X_input and X_outcome along axis 1
        combined_result = pd.concat([X_input, X_outcome], axis=1)
        return combined_result


    # Compute and return all-token embeddings.
    def get_all_token_embeddings(self, X_input, sequence_name='sequence', id_column=None):
        """
        Compute all-token embeddings for each protein sequence.

        The returned dictionary maps a protein ID, or the row index when no ID
        column is supplied, to an array with shape (n_tokens, embedding_dim).
        """
        X_input = X_input.reset_index(drop=True)
        sequence_list = X_input[sequence_name].tolist()

        self.model.to(self.device)
        embeddings_dict = {}

        with torch.no_grad():
            for i in tqdm(range(len(sequence_list)), desc='Protein embeddings extraction'):
                seq = sequence_list[i]
                # Resolve the protein ID.
                protein_id = X_input.loc[i, id_column] if id_column is not None else i

                inputs = self.tokenizer(seq, return_tensors="pt", padding=True).to(self.device)

                outputs = self.model(**inputs)

                # Select the final hidden layer.
                last_hidden = outputs.hidden_states[-1]

                embeddings_np = last_hidden[0].cpu().numpy()

                # Store the token matrix.
                embeddings_dict[protein_id] = embeddings_np

                # Release intermediate tensors.
                del inputs, outputs
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        return embeddings_dict

    # Extract phosphorylation-site representations.
    def get_last_hidden_phosphorylation_position_feature(self, X_input, sequence_name='sequence', 
                                                         phosphorylation_positions='phosphorylation_positions', batch_size=32):
        
        X_input = X_input.reset_index(drop=True)
        
        
            
        self.model.to(self.device)

        # Group X_input by sequence
        grouped_X_input = X_input.groupby(sequence_name)
        sequence_to_indices = grouped_X_input.groups

        # Pre-compute the number of features
        num_features = self.model.config.hidden_size
        columns = [f"ESM2_phospho_pos{k}" for k in range(num_features)]

        # Create an empty DataFrame with the column names
        X_outcome = pd.DataFrame(columns=columns)

        with torch.no_grad():
            for i in tqdm(range(0, len(grouped_X_input), batch_size), desc='batches for inference'):
                batch_sequences = list(islice(sequence_to_indices.keys(), i, i + batch_size))
                batch_grouped_sequences = {seq: X_input.loc[sequence_to_indices[seq]] for seq in batch_sequences}

                # Get the unique sequences in the batch
                inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model(**inputs)

                for j, sequence in enumerate(batch_sequences):
                    sequence_indices = batch_grouped_sequences[sequence].index
                    sequence_positions = batch_grouped_sequences[sequence][phosphorylation_positions].tolist()
                    last_hidden_state = self.get_last_hidden_states(outputs)[j:j+1]

                    for idx, position in zip(sequence_indices, sequence_positions):
                        position = int(position)  # Make sure position is an integer
                        position_feature = last_hidden_state[:, position, :]  # Removed +1 since the sequence starts from 1, and consider removing the cls token
                        features = position_feature.squeeze().tolist()

                        # Add the new row to the DataFrame
                        X_outcome.loc[idx] = features

                del inputs, outputs
                # Clear and synchronize only when CUDA is active.
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()


        # Print the dimension of the final phosphorylation features
        print(f"The dimension of the final phosphorylation features is: {X_outcome.shape[1]}")

        # Combine X_input and X_outcome along axis 1
        combined_result = pd.concat([X_input, X_outcome], axis=1)
        return combined_result

    # Extract phosphorylation-site representations in fast mode.
    def get_last_hidden_phosphorylation_position_feature_fast(self, X_input, sequence_name='sequence',
                                                              phosphorylation_positions='phosphorylation_positions',
                                                              batch_size=32):

        X_input = X_input.reset_index(drop=True)

        self.model.to(self.device)

        # Group X_input by sequence
        grouped_X_input = X_input.groupby(sequence_name)
        sequence_to_indices = grouped_X_input.groups

        # Pre-compute the number of features
        num_features = self.model.config.hidden_size
        columns = [f"ESM2_phospho_pos{k}" for k in range(num_features)]

        # Create an empty DataFrame with the column names
        X_outcome = pd.DataFrame(columns=columns)

        with torch.no_grad():
            for i in tqdm(range(0, len(grouped_X_input), batch_size), desc='batches for inference'):
                batch_sequences = list(islice(sequence_to_indices.keys(), i, i + batch_size))
                batch_grouped_sequences = {seq: X_input.loc[sequence_to_indices[seq]] for seq in batch_sequences}

                # Get the unique sequences in the batch
                inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model(**inputs)

                for j, sequence in enumerate(batch_sequences):
                    sequence_indices = batch_grouped_sequences[sequence].index
                    sequence_positions = batch_grouped_sequences[sequence][phosphorylation_positions].tolist()
                    last_hidden_state = self.get_last_hidden_states(outputs)[j:j + 1]

                    sequence_positions = np.array(sequence_positions).astype(int)
                    # Allocate a list for the extracted vectors.
                    features = []

                    # Extract all requested positions with advanced indexing.
                    # Expected shape: (1, sequence_length, hidden_size).
                    position_features = last_hidden_state[:, sequence_positions, :]

                    # Remove the singleton batch dimension.
                    if last_hidden_state.shape[0] == 1:
                        position_features = position_features.squeeze(0)

                    # Convert each vector to a list.
                    for feature in position_features:
                        features.append(feature.tolist())

                    # Map the vectors back to their DataFrame indices.
                    for idx, feature in zip(sequence_indices.tolist(), features):
                        X_outcome.loc[idx] = feature

                del inputs, outputs
                # Clear and synchronize only when CUDA is active.
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        # Print the dimension of the final phosphorylation features
        print(f"The dimension of the final phosphorylation features is: {X_outcome.shape[1]}")

        # Combine X_input and X_outcome along axis 1
        combined_result = pd.concat([X_input, X_outcome], axis=1)
        return combined_result


    def get_amino_acid_representation(self, sequence, amino_acid, position):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Check if the amino acid at the given position matches the input
        if sequence[position - 1] != amino_acid:
            raise ValueError(f"The amino acid at position {position} is not {amino_acid}.")

        # Convert the sequence to input tensors
        inputs = self.tokenizer([sequence], return_tensors="pt", padding=True).to(self.device)
        
        # Get the model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get the last hidden state
        last_hidden_state = self.get_last_hidden_states(outputs)
        
        # Get the tokens from the input ids
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        
        # Get the position of the amino acid token in the tokens list
        # We add 1 to the position to account for the CLS token at the start
        token_position =  position if amino_acid == tokens[position] else -1
        
        if token_position == -1:
            raise ValueError(f"The token for amino acid {amino_acid} could not be found in the tokenized sequence.")
        
        # Get the feature vector for the amino acid
        amino_acid_features = last_hidden_state[:, token_position, :].squeeze().tolist()
        
        
        # Get the feature vector for the amino acid
        amino_acid_features = last_hidden_state[:, token_position, :].squeeze().tolist()
        
        # Prepare the DataFrame
        feature_names = [f"ESM2_{k}" for k in range(len(amino_acid_features))]
        amino_acid_features_df = pd.DataFrame(amino_acid_features, index=feature_names, columns=[amino_acid]).T

        return amino_acid_features_df

    

class Esm2LayerHiddenFeatureExtractor:
    def __init__(self, tokenizer, model, layer_indicat, compute_cls=True, compute_eos=True, compute_mean=True, compute_segments=False, num_segments=10,device_choose = 'auto'):
        self.tokenizer = tokenizer
        self.model = model
        self.layer_indicat = layer_indicat
        self.compute_cls = compute_cls
        self.compute_eos = compute_eos
        self.compute_mean = compute_mean
        self.compute_segments = compute_segments
        self.num_segments = num_segments

        if device_choose == 'auto':
            # Automatically select device: prefer CUDA if available, otherwise use CPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_choose.startswith('cuda'):
            # Support specifying a specific CUDA device number, e.g., 'cuda:0', 'cuda:1'
            if torch.cuda.is_available():
                try:
                    # Extract device id, default is 0
                    device_id = int(device_choose.split(':')[-1]) if ':' in device_choose else 0
                    self.device = torch.device(f"cuda:{device_id}")
                except ValueError:
                    raise ValueError(
                        "Invalid CUDA device format. Use 'cuda' or 'cuda:<id>', where <id> is an integer.")
            else:
                raise TypeError("CUDA is not available. Please check your GPU settings.")
        elif device_choose == 'cpu':
            # Force using CPU
            self.device = torch.device("cpu")
        else:
            # Handle invalid device_choose value
            raise ValueError("Invalid device choice. Use 'auto', 'cpu', or 'cuda[:<id>]'.")

    def get_layer_hidden_states(self, outputs):
        layer_hidden_state = outputs.hidden_states[self.layer_indicat]
        return layer_hidden_state

    def get_layer_cls_token(self, layer_hidden_state):
        return layer_hidden_state[:, 0, :]

    def get_layer_eos_token(self, layer_hidden_state, eos_position):
        return layer_hidden_state[:, eos_position, :]

    def get_layer_mean_token(self, layer_hidden_state, eos_position):
        return layer_hidden_state[:, 1:eos_position, :].mean(dim=1)

    def get_segment_mean_tokens(self, layer_hidden_state, eos_position):
        seq_len = eos_position - 1
        segment_size, remainder = divmod(seq_len, self.num_segments)
        segment_means = []

        start = 1
        for i in range(self.num_segments):
            end = start + segment_size + (1 if i < remainder else 0)
            
            if end > start:  # Check if the segment has amino acids
                segment_mean = layer_hidden_state[:, start:end, :].mean(dim=1)
            else:  # If the segment is empty, create a zero tensor with the same dimensions as the hidden state
                segment_mean = torch.zeros(layer_hidden_state[:, start:start+1, :].shape, device=layer_hidden_state.device)
            
            segment_means.append(segment_mean.squeeze().tolist())
            start = end

        return segment_means
    
    
    # Compute layer-specific CLS, EOS, residue-mean and segment-mean features.
    def get_layer_hidden_features_combine(self, X_input, sequence_name='sequence', batch_size=32):
        X_input = X_input.reset_index(drop=True)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        sequence = X_input[sequence_name].tolist()

        features_length = {}  # save the length of different features
        columns = None  # initialize the column names
        all_results = []  # Store all batch results
        with torch.no_grad():
            for i in tqdm(range(0, len(sequence), batch_size), desc='batches for inference'):
                batch_sequences = sequence[i:i+batch_size]
                inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model(**inputs)

                for j in range(len(batch_sequences)):
                    idx = i + j
                    tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][j])
                    eos_position = tokens.index(self.tokenizer.eos_token) if self.tokenizer.eos_token in tokens else len(batch_sequences[j])
                    layer_hidden_state = self.get_layer_hidden_states(outputs)
                    layer_cls_token = self.get_layer_cls_token(layer_hidden_state[j:j+1]) if self.compute_cls else None
                    layer_eos_token = self.get_layer_eos_token(layer_hidden_state[j:j+1], eos_position) if self.compute_eos else None
                    layer_mean_token = self.get_layer_mean_token(layer_hidden_state[j:j+1], eos_position) if self.compute_mean else None
                    segment_means = self.get_segment_mean_tokens(layer_hidden_state[j:j+1], eos_position) if self.compute_segments else None

                    # extract features and add them to DataFrame directly
                    features = []
                    if layer_cls_token is not None:
                        cls_features = layer_cls_token.squeeze().tolist()
                        if 'cls' not in features_length:
                            features_length['cls'] = len(cls_features)
                        features.extend(cls_features)

                    if layer_eos_token is not None:
                        eos_features = layer_eos_token.squeeze().tolist()
                        if 'eos' not in features_length:
                            features_length['eos'] = len(eos_features)
                        features.extend(eos_features)

                    if layer_mean_token is not None:
                        mean_features = layer_mean_token.squeeze().tolist()
                        if 'mean' not in features_length:
                            features_length['mean'] = len(mean_features)
                        features.extend(mean_features)

                    if segment_means is not None:
                        # In the new version, we keep each segment mean as a separate list
                        for seg, segment_mean in enumerate(segment_means):
                            features.extend(segment_mean)
                            if f'segment{seg}_mean' not in features_length:
                                features_length[f'segment{seg}_mean'] = len(segment_mean)

                    # create the column names only for the first item
                    if columns is None:
                        columns = []
                        for feature_type, length in features_length.items():
                            for k in range(length):
                                columns.append(f"ESM2_{feature_type}{k}")

                    # Create DataFrame for this batch
                    result = pd.DataFrame([features], columns=columns, index=[idx])
                    all_results.append(result)

                del inputs, outputs
                # Clear and synchronize only when CUDA is active.
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        # Combine all batch results outside the loop
        X_outcome = pd.concat(all_results, axis=0)

        print(f'Features dimensions: {features_length}')

        # Combine X_input and X_outcome along axis 1
        combined_result = pd.concat([X_input, X_outcome], axis=1)
        return combined_result
    


    
    # Extract layer-specific phosphorylation-site representations.
    def get_layer_hidden_phosphorylation_position_feature(self, X_input, sequence_name='sequence', phosphorylation_positions='phosphorylation_positions', batch_size=32):
        X_input = X_input.reset_index(drop=True)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Group X_input by sequence
        grouped_X_input = X_input.groupby(sequence_name)
        sequence_to_indices = grouped_X_input.groups

        # Pre-compute the number of features
        num_features = self.model.config.hidden_size
        columns = [f"ESM2_phospho_pos{k}" for k in range(num_features)]

        # Create an empty DataFrame with the column names
        X_outcome = pd.DataFrame(columns=columns)

        with torch.no_grad():
            for i in tqdm(range(0, len(grouped_X_input), batch_size), desc='batches for inference'):
                batch_sequences = list(islice(sequence_to_indices.keys(), i, i + batch_size))
                batch_grouped_sequences = {seq: X_input.loc[sequence_to_indices[seq]] for seq in batch_sequences}

                # Get the unique sequences in the batch
                inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model(**inputs)

                for j, sequence in enumerate(batch_sequences):
                    sequence_indices = batch_grouped_sequences[sequence].index
                    sequence_positions = batch_grouped_sequences[sequence][phosphorylation_positions].tolist()
                    layer_hidden_state = self.get_layer_hidden_states(outputs)[j:j+1]

                    for idx, position in zip(sequence_indices, sequence_positions):
                        position = int(position)  # Make sure position is an integer
                        position_feature = layer_hidden_state[:, position, :]  # Removed +1 since the sequence starts from 1, and consider removing the cls token
                        features = position_feature.squeeze().tolist()

                        # Add the new row to the DataFrame
                        X_outcome.loc[idx] = features

                del inputs, outputs
                # Clear and synchronize only when CUDA is active.
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()


        # Print the dimension of the final phosphorylation features
        print(f"The dimension of the final phosphorylation features is: {X_outcome.shape[1]}")

        # Combine X_input and X_outcome along axis 1
        combined_result = pd.concat([X_input, X_outcome], axis=1)
        return combined_result
    
 
    def get_amino_acid_representation(self, sequence, amino_acid, position):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Check if the amino acid at the given position matches the input
        if sequence[position - 1] != amino_acid:
            raise ValueError(f"The amino acid at position {position} is not {amino_acid}.")

        # Convert the sequence to input tensors
        inputs = self.tokenizer([sequence], return_tensors="pt", padding=True).to(self.device)
        
        # Get the model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get the layer hidden state
        layer_hidden_state = self.get_layer_hidden_states(outputs)
        
        # Get the tokens from the input ids
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        
        # Get the position of the amino acid token in the tokens list
        # We add 1 to the position to account for the CLS token at the start
        token_position =  position if amino_acid == tokens[position] else -1
        
        if token_position == -1:
            raise ValueError(f"The token for amino acid {amino_acid} could not be found in the tokenized sequence.")
        
        # Get the feature vector for the amino acid
        amino_acid_features = layer_hidden_state[:, token_position, :].squeeze().tolist()
        
        
        # Get the feature vector for the amino acid
        amino_acid_features = layer_hidden_state[:, token_position, :].squeeze().tolist()
        
        # Prepare the DataFrame
        feature_names = [f"ESM2_{k}" for k in range(len(amino_acid_features))]
        amino_acid_features_df = pd.DataFrame(amino_acid_features, index=feature_names, columns=[amino_acid]).T

        return amino_acid_features_df







def NetPhos_classic_txt_DataFrame(pattern, data):
    '''
    This function takes a pattern and data as input and returns a DataFrame containing
    the parsed information.

    Parameters
    ----------
    pattern : str
        A regular expression pattern used to match lines in the input data.
    data : str
        The input data containing the information to be parsed.

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame containing the parsed information.

    Example
    -------
    To use this function with the example file provided in the package:

    >>> import os
    >>> import protloc_mex_X
    >>> from protloc_mex_X.ESM2_fr import NetPhos_classic_txt_DataFrame
    ...
    >>> example_data = os.path.join(protloc_mex_X.__path__[0], "examples", "test1.txt")
    ...
    >>> with open(example_data, "r") as f:
    ...     data = f.read()
    ... print(data)
    >>> pattern = r".*YES"
    >>> result_df = NetPhos_classic_txt_DataFrame(pattern, data)

    '''

    # Extract lines that match the pattern
    seq_lines = re.findall(pattern, data)

    # Split the extracted lines into lists
    split_lines = [line.split() for line in seq_lines]

    # Remove '#' character
    split_lines = [line[1:] for line in split_lines]

    # Check if each list element has a length of 7, if not, skip that line
    filtered_split_lines = [line for line in split_lines if len(line) == 7]

    # Convert the filtered list to a DataFrame and set column names
    column_names = ['Sequence', 'position', 'x', 'Context', 'Score', 'Kinase', 'Answer']
    df = pd.DataFrame(filtered_split_lines, columns=column_names)

    # Convert the 'Score' column to float type
    df['Score'] = df['Score'].astype(float)

    return df



def phospho_feature_sim_cosine_weighted_average(dim, df_cls, df_phospho):
    # Check if all protein_id in df_phospho are in df_cls
    if not df_phospho.index.isin(df_cls.index).all():
        raise ValueError("Protein_id in df_phospho is not matched with df_cls.")

    # Merge df_cls and df_phospho on index (protein_id)
    df = pd.merge(df_cls, df_phospho, how='inner', left_index=True, right_index=True)

    # Calculate cosine similarity for each row in the merged dataframe
    # First dim columns.
    array_cls = df.iloc[:, :dim].values

    # Next dim columns.
    array_phospho = df.iloc[:, dim:2*dim].values

    # Add epsilon to all-zero rows to avoid division by zero.
    epsilon = 1e-10
    array_cls[np.where(~array_cls.any(axis=1))[0]] += epsilon
    array_phospho[np.where(~array_phospho.any(axis=1))[0]] += epsilon

    # Compute cosine similarity.
    similarity = np.sum(array_cls * array_phospho, axis=1) / (np.linalg.norm(array_cls, axis=1) * np.linalg.norm(array_phospho, axis=1))

    # Add the similarity column.
    df['similarity'] = similarity
    #Multiply similarity with phospho_feature for each row
    
    # Apply the similarity weights with NumPy.
    array_weighted_features= df['similarity'].values[:, None] * array_phospho

    # Convert the weighted vectors to a DataFrame and merge them.
    weighted_features_df = pd.DataFrame(array_weighted_features, columns=[f'weighted{i}' for i in range(dim)], index=df.index)
    df = pd.concat([df, weighted_features_df], axis=1)

    # Calculate total weights (sum of abs similarities) for each protein
    total_weights = df['similarity'].abs().groupby(df.index).sum()
    # Calculate sum of weighted features for each protein
    grouped = df.groupby(df.index).agg({**{f'weighted{i}': 'sum' for i in range(dim)}})

    # Calculate weighted average by dividing sum of weighted features by total weights
    # for i in range(dim):
    #     grouped[f'average{i}'] = grouped[f'weighted{i}'] / total_weights
    average_features = {f'pho_average{i}': grouped[f'weighted{i}'] / total_weights for i in range(dim)}
    average_df = pd.DataFrame(average_features)
    grouped = pd.concat([grouped, average_df], axis=1)

    # Merge df_cls and grouped dataframe by protein_id (index)
    df_cls = pd.merge(df_cls, grouped[[f'pho_average{i}' for i in range(dim)]], how='left', left_index=True, right_index=True)

    # For proteins that do not have phospho_feature, set average_feature to zero
    # for i in range(dim):
    #     df_cls[f'average{i}'] = df_cls[f'average{i}'].fillna(0)
    if df_cls.isnull().any().any():
        df_cls = df_cls.fillna(0)
        
    return df_cls


def phospho_feature_sim_cosine_weighted_average_test(dim, df_cls, df_phospho):
    """
    This function computes the overall phospho-representation for a single amino acid sequence 
    using a cosine similarity-based weighting scheme. It merges the given dataframes on their 
    index (protein_id), calculates the cosine similarity for each row, and then calculates the 
    total weights for each protein. It then computes the sum of weighted features for each protein 
    and returns this sum.
    
    Note: This function is designed to work with one amino acid sequence at a time.
    
    Parameters:
    dim (int): The dimensionality of the feature vectors.
    df_cls (DataFrame): The cls feature DataFrame.
    df_phospho (DataFrame): The phospho feature DataFrame.
    
    Returns:
    Series: The sum of the weighted features.
    """
    # Merge df_cls and df_phospho on index (protein_id)
    df = pd.merge(df_cls, df_phospho, how='inner', left_index=True, right_index=True)
    
    # Calculate cosine similarity for each row in the merged dataframe
    similarity = np.sum(df_cls.to_numpy() * df_phospho.to_numpy(), axis=1) / (np.linalg.norm(df_cls.to_numpy(), axis=1) * np.linalg.norm(df_phospho.to_numpy(), axis=1))
    
    # Add the similarity to the DataFrame
    df['similarity'] = similarity
    
    # Calculate total weights (sum of abs similarities) for each protein
    total_weights = df['similarity'].abs().groupby(df.index).sum()
    
    # Calculate sum of weighted features for each protein
    weighted_features = df_phospho.copy()
    weighted_features.columns = [f'weighted{i}' for i in range(dim)]
    
    # Calculate the weight for each row
    df['weight'] = df['similarity'] / total_weights
    
    # Repeat the weights to the same dimension as the features
    weights_matrix = np.repeat(df['weight'].values[:, np.newaxis], dim, axis=1)
    
    # Multiply each row of features by its weight
    average_total = weighted_features.multiply(weights_matrix, axis=0)
    
    # Calculate the sum of the weighted features
    average_total_final = average_total.sum(axis=0)
    
    return average_total_final


##the new function (update version)


class Esm2LogitsExtractor:
    """
    Esm2LogitsExtractor class is designed to perform batch inference on input protein sequences and extract the model's logits matrix.

    Main functionalities:
      - Perform batch inference on protein sequences stored in a DataFrame.
      - Use a tokenizer to encode sequences (with automatic padding and truncation).
      - Compute logits using the specified model.
      - Concatenate logits from all batches and convert the result to a NumPy array.

    Expected output:
      The returned NumPy array, logits_matrix, has the shape (B, L, V), where:
        - B represents the number of proteins, i.e., the number of samples in the input DataFrame.
        - L represents the sequence length, i.e., the unified length after padding/truncation (default 4024).
        - V represents the vocabulary size, i.e., the size of the last dimension of the model's logits output.

    Parameters:
      tokenizer: The tokenizer corresponding to the model, used to encode protein sequences into the model input format.
      model: The preloaded model used to compute logits.
      device_choose (str): Specifies the device selection method. Options include 'auto' (default), 'cpu', or 'cuda[:<id>]' (e.g., 'cuda:1').
    """

    def __init__(self, tokenizer, model, device_choose='auto'):
        self.tokenizer = tokenizer
        self.model = model

        # Auto-select device: use CUDA if available, otherwise CPU.
        if device_choose == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_choose.startswith('cuda'):
            if torch.cuda.is_available():
                try:
                    device_id = int(device_choose.split(':')[-1]) if ':' in device_choose else 0
                    self.device = torch.device(f"cuda:{device_id}")
                except ValueError:
                    raise ValueError("Invalid CUDA device format. Use 'cuda' or 'cuda:<id>', where <id> is an integer.")
            else:
                raise TypeError("CUDA is not available. Please check your GPU settings.")
        elif device_choose == 'cpu':
            self.device = torch.device("cpu")
        else:
            raise ValueError("Invalid device choice. Use 'auto', 'cpu', or 'cuda[:<id>]'.")

    def get_logits(
            self,
            X_input,
            sequence_name='sequence',
            batch_size=32,
            do_padding=True,
            do_truncation=True,
            max_length=4024
    ):
        """
        Perform batch inference on input protein sequences to extract the logits matrix,
        returning a NumPy array with shape (B, L, V).

        Args:
          X_input (pd.DataFrame): The input DataFrame containing protein sequences.
          sequence_name (str): The column name in X_input that contains the sequences. Default 'sequence'.
          batch_size (int): The batch size for inference. Default 32.
          do_padding (bool): Whether to pad sequences. If True, sequences will be padded to max_length. Default True.
          do_truncation (bool): Whether to truncate sequences longer than max_length. Default True.
          max_length (int): The fixed sequence length for padding/truncation. Default 4024.

        Returns:
          logits_matrix (np.ndarray): A NumPy array of shape (B, L, V), where:
            B = number of protein sequences,
            L = sequence length after padding/truncation (max_length),
            V = vocabulary size from the model.
        """
        X_input = X_input.reset_index(drop=True)
        self.model.to(self.device)
        sequences = X_input[sequence_name].tolist()

        # Configure tokenizer parameters for fixed max_length padding and truncation.
        tokenizer_args = {
            "return_tensors": "pt",
            "padding": "max_length" if do_padding else False,
            "truncation": do_truncation,
            "max_length": max_length
        }

        logits_list = []
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc='batches for inference'):
                batch_sequences = sequences[i: i + batch_size]
                inputs = self.tokenizer(batch_sequences, **tokenizer_args).to(self.device)
                outputs = self.model(**inputs)
                # outputs.logits shape: (batch_size, seq_length, vocab_size)
                logits_list.append(outputs.logits.cpu())

                del inputs, outputs
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        # Concatenate logits along the batch dimension, resulting in shape (B, L, V)
        logits_matrix = torch.cat(logits_list, dim=0).numpy()
        return logits_matrix




### The remaining legacy code is kept for model checks and is not run by the main workflow.


# class Esm2LastHiddenFeatureExtractor_legacy:
#     def __init__(self, tokenizer, model, compute_cls=True, compute_eos=True, compute_mean=True, compute_segments=False,device_choose = 'auto'):
#         self.tokenizer = tokenizer
#         self.model = model
#         self.compute_cls = compute_cls
#         self.compute_eos = compute_eos
#         self.compute_mean = compute_mean
#         self.compute_segments = compute_segments
        
#         if device_choose == 'auto':
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         elif device_choose == 'cuda':
#             if torch.cuda.is_available():
#                 self.device = torch.device("cuda")
#             else:
#                 raise TypeError("CUDA is not available. Please check your GPU settings.")
#         elif device_choose == 'cpu':
#             self.device = torch.device("cpu")
        
#     def get_last_hidden_states(self, outputs):
#         last_hidden_state = outputs.hidden_states[-1]
#         return last_hidden_state

#     def get_last_cls_token(self, last_hidden_state):
#         return last_hidden_state[:, 0, :]

#     def get_last_eos_token(self, last_hidden_state, eos_position):
#         return last_hidden_state[:, eos_position, :]

#     def get_last_mean_token(self, last_hidden_state, eos_position):
#         return last_hidden_state[:, 1:eos_position, :].mean(dim=1)

#     def get_segment_mean_tokens(self, last_hidden_state, eos_position, num_segments=10):
#         seq_len = eos_position - 1
#         segment_size, remainder = divmod(seq_len, num_segments)
#         segment_means = []

#         start = 1
#         for i in range(num_segments):
#             end = start + segment_size + (1 if i < remainder else 0)
            
#             if end > start:  # Check if the segment has amino acids
#                 segment_mean = last_hidden_state[:, start:end, :].mean(dim=1)
#             else:  # If the segment is empty, create a zero tensor with the same dimensions as the hidden state
#                 segment_mean = torch.zeros(last_hidden_state[:, start:start+1, :].shape, device=last_hidden_state.device)
            
#             segment_means.append(segment_mean.squeeze().tolist())
#             start = end

#         return segment_means
        
    
    
#     # Compute CLS, EOS, residue-mean and segment-mean representations.
#     def get_last_hidden_features_combine(self, X_input, sequence_name='sequence', batch_size=32):
#         X_input = X_input.reset_index(drop=True)
        
        
#         self.model.to(self.device)
#         sequence = X_input[sequence_name].tolist()

#         features_length = {}  # save the length of different features
#         columns = None  # initialize the column names
#         all_results = []  # Store all batch results
#         with torch.no_grad():
#             for i in tqdm(range(0, len(sequence), batch_size), desc='batches for inference'):
#                 batch_sequences = sequence[i:i+batch_size]
#                 inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True).to(self.device)
#                 outputs = self.model(**inputs)

#                 for j in range(len(batch_sequences)):
#                     idx = i + j
#                     tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][j])
#                     eos_position = tokens.index(self.tokenizer.eos_token) if self.tokenizer.eos_token in tokens else len(batch_sequences[j])
#                     last_hidden_state = self.get_last_hidden_states(outputs)
#                     last_cls_token = self.get_last_cls_token(last_hidden_state[j:j+1]) if self.compute_cls else None
#                     last_eos_token = self.get_last_eos_token(last_hidden_state[j:j+1], eos_position) if self.compute_eos else None
#                     last_mean_token = self.get_last_mean_token(last_hidden_state[j:j+1], eos_position) if self.compute_mean else None
#                     segment_means = self.get_segment_mean_tokens(last_hidden_state[j:j+1], eos_position) if self.compute_segments else None

#                     # extract features and add them to DataFrame directly
#                     features = []
#                     if last_cls_token is not None:
#                         cls_features = last_cls_token.squeeze().tolist()
#                         if 'cls' not in features_length:
#                             features_length['cls'] = len(cls_features)
#                         features.extend(cls_features)

#                     if last_eos_token is not None:
#                         eos_features = last_eos_token.squeeze().tolist()
#                         if 'eos' not in features_length:
#                             features_length['eos'] = len(eos_features)
#                         features.extend(eos_features)

#                     if last_mean_token is not None:
#                         mean_features = last_mean_token.squeeze().tolist()
#                         if 'mean' not in features_length:
#                             features_length['mean'] = len(mean_features)
#                         features.extend(mean_features)

#                     if segment_means is not None:
#                         # In the new version, we keep each segment mean as a separate list
#                         for seg, segment_mean in enumerate(segment_means):
#                             features.extend(segment_mean)
#                             if f'segment{seg}_mean' not in features_length:
#                                 features_length[f'segment{seg}_mean'] = len(segment_mean)

#                     # create the column names only for the first item
#                     if columns is None:
#                         columns = []
#                         for feature_type, length in features_length.items():
#                             for k in range(length):
#                                 columns.append(f"ESM2_{feature_type}{k}")

#                     # Create DataFrame for this batch
#                     result = pd.DataFrame([features], columns=columns, index=[idx])
#                     all_results.append(result)

#                 del inputs, outputs
                
#                 # Clear and synchronize only when CUDA is active.
#                 if self.device.type == 'cuda':
#                     torch.cuda.empty_cache()
#                     torch.cuda.synchronize()

#         # Combine all batch results outside the loop
#         X_outcome = pd.concat(all_results, axis=0)

#         print(f'Features dimensions: {features_length}')

#         # Combine X_input and X_outcome along axis 1
#         combined_result = pd.concat([X_input, X_outcome], axis=1)
#         return combined_result
    


    
#     # Extract phosphorylation-site representations.
#     def get_last_hidden_phosphorylation_position_feature(self, X_input, sequence_name='sequence', 
#                                                          phosphorylation_positions='phosphorylation_positions', batch_size=32):
        
#         X_input = X_input.reset_index(drop=True)
        
        
            
#         self.model.to(self.device)

#         # Group X_input by sequence
#         grouped_X_input = X_input.groupby(sequence_name)
#         sequence_to_indices = grouped_X_input.groups

#         # Pre-compute the number of features
#         num_features = self.model.config.hidden_size
#         columns = [f"ESM2_phospho_pos{k}" for k in range(num_features)]

#         # Create an empty DataFrame with the column names
#         X_outcome = pd.DataFrame(columns=columns)

#         with torch.no_grad():
#             for i in tqdm(range(0, len(grouped_X_input), batch_size), desc='batches for inference'):
#                 batch_sequences = list(islice(sequence_to_indices.keys(), i, i + batch_size))
#                 batch_grouped_sequences = {seq: X_input.loc[sequence_to_indices[seq]] for seq in batch_sequences}

#                 # Get the unique sequences in the batch
#                 inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True).to(self.device)
#                 outputs = self.model(**inputs)

#                 for j, sequence in enumerate(batch_sequences):
#                     sequence_indices = batch_grouped_sequences[sequence].index
#                     sequence_positions = batch_grouped_sequences[sequence][phosphorylation_positions].tolist()
#                     last_hidden_state = self.get_last_hidden_states(outputs)[j:j+1]

#                     for idx, position in zip(sequence_indices, sequence_positions):
#                         position = int(position)  # Make sure position is an integer
#                         position_feature = last_hidden_state[:, position, :]  # Removed +1 since the sequence starts from 1, and consider removing the cls token
#                         features = position_feature.squeeze().tolist()

#                         # Add the new row to the DataFrame
#                         X_outcome.loc[idx] = features

#                 del inputs, outputs
#                 # Clear and synchronize only when CUDA is active.
#                 if self.device.type == 'cuda':
#                     torch.cuda.empty_cache()
#                     torch.cuda.synchronize()


#         # Print the dimension of the final phosphorylation features
#         print(f"The dimension of the final phosphorylation features is: {X_outcome.shape[1]}")

#         # Combine X_input and X_outcome along axis 1
#         combined_result = pd.concat([X_input, X_outcome], axis=1)
#         return combined_result
    
 
#     def get_amino_acid_representation(self, sequence, amino_acid, position):
#         # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)
        
#         # Check if the amino acid at the given position matches the input
#         if sequence[position - 1] != amino_acid:
#             raise ValueError(f"The amino acid at position {position} is not {amino_acid}.")

#         # Convert the sequence to input tensors
#         inputs = self.tokenizer([sequence], return_tensors="pt", padding=True).to(self.device)
        
#         # Get the model outputs
#         with torch.no_grad():
#             outputs = self.model(**inputs)
        
#         # Get the last hidden state
#         last_hidden_state = self.get_last_hidden_states(outputs)
        
#         # Get the tokens from the input ids
#         tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        
#         # Get the position of the amino acid token in the tokens list
#         # We add 1 to the position to account for the CLS token at the start
#         token_position =  position if amino_acid == tokens[position] else -1
        
#         if token_position == -1:
#             raise ValueError(f"The token for amino acid {amino_acid} could not be found in the tokenized sequence.")
        
#         # Get the feature vector for the amino acid
#         amino_acid_features = last_hidden_state[:, token_position, :].squeeze().tolist()
        
        
#         # Get the feature vector for the amino acid
#         amino_acid_features = last_hidden_state[:, token_position, :].squeeze().tolist()
        
#         # Prepare the DataFrame
#         feature_names = [f"ESM2_{k}" for k in range(len(amino_acid_features))]
#         amino_acid_features_df = pd.DataFrame(amino_acid_features, index=feature_names, columns=[amino_acid]).T

#         return amino_acid_features_df




# class Esm2LayerHiddenFeatureExtractor_legacy:
#     def __init__(self, tokenizer, model, layer_indicat, compute_cls=True, compute_eos=True, compute_mean=True, compute_segments=False, device_choose = 'auto'):
#         self.tokenizer = tokenizer
#         self.model = model
#         self.layer_indicat = layer_indicat
#         self.compute_cls = compute_cls
#         self.compute_eos = compute_eos
#         self.compute_mean = compute_mean
#         self.compute_segments = compute_segments
    
#         if device_choose == 'auto':
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         elif device_choose == 'cuda':
#             if torch.cuda.is_available():
#                 self.device = torch.device("cuda")
#             else:
#                 raise TypeError("CUDA is not available. Please check your GPU settings.")
#         elif device_choose == 'cpu':
#             self.device = torch.device("cpu")
    
    
#     def get_layer_hidden_states(self, outputs):
#         layer_hidden_state = outputs.hidden_states[self.layer_indicat]
#         return layer_hidden_state

#     def get_layer_cls_token(self, layer_hidden_state):
#         return layer_hidden_state[:, 0, :]

#     def get_layer_eos_token(self, layer_hidden_state, eos_position):
#         return layer_hidden_state[:, eos_position, :]

#     def get_layer_mean_token(self, layer_hidden_state, eos_position):
#         return layer_hidden_state[:, 1:eos_position, :].mean(dim=1)

#     def get_segment_mean_tokens(self, layer_hidden_state, eos_position, num_segments=10):
#         seq_len = eos_position - 1
#         segment_size, remainder = divmod(seq_len, num_segments)
#         segment_means = []

#         start = 1
#         for i in range(num_segments):
#             end = start + segment_size + (1 if i < remainder else 0)
            
#             if end > start:  # Check if the segment has amino acids
#                 segment_mean = layer_hidden_state[:, start:end, :].mean(dim=1)
#             else:  # If the segment is empty, create a zero tensor with the same dimensions as the hidden state
#                 segment_mean = torch.zeros(layer_hidden_state[:, start:start+1, :].shape, device=layer_hidden_state.device)
            
#             segment_means.append(segment_mean.squeeze().tolist())
#             start = end

#         return segment_means
    
    
#     # Compute layer-specific CLS, EOS, residue-mean and segment-mean features.
#     def get_layer_hidden_features_combine(self, X_input, sequence_name='sequence', batch_size=32):
#         X_input = X_input.reset_index(drop=True)
#         # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)
#         sequence = X_input[sequence_name].tolist()

#         features_length = {}  # save the length of different features
#         columns = None  # initialize the column names
#         all_results = []  # Store all batch results
#         with torch.no_grad():
#             for i in tqdm(range(0, len(sequence), batch_size), desc='batches for inference'):
#                 batch_sequences = sequence[i:i+batch_size]
#                 inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True).to(self.device)
#                 outputs = self.model(**inputs)

#                 for j in range(len(batch_sequences)):
#                     idx = i + j
#                     tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][j])
#                     eos_position = tokens.index(self.tokenizer.eos_token) if self.tokenizer.eos_token in tokens else len(batch_sequences[j])
#                     layer_hidden_state = self.get_layer_hidden_states(outputs)
#                     layer_cls_token = self.get_layer_cls_token(layer_hidden_state[j:j+1]) if self.compute_cls else None
#                     layer_eos_token = self.get_layer_eos_token(layer_hidden_state[j:j+1], eos_position) if self.compute_eos else None
#                     layer_mean_token = self.get_layer_mean_token(layer_hidden_state[j:j+1], eos_position) if self.compute_mean else None
#                     segment_means = self.get_segment_mean_tokens(layer_hidden_state[j:j+1], eos_position) if self.compute_segments else None

#                     # extract features and add them to DataFrame directly
#                     features = []
#                     if layer_cls_token is not None:
#                         cls_features = layer_cls_token.squeeze().tolist()
#                         if 'cls' not in features_length:
#                             features_length['cls'] = len(cls_features)
#                         features.extend(cls_features)

#                     if layer_eos_token is not None:
#                         eos_features = layer_eos_token.squeeze().tolist()
#                         if 'eos' not in features_length:
#                             features_length['eos'] = len(eos_features)
#                         features.extend(eos_features)

#                     if layer_mean_token is not None:
#                         mean_features = layer_mean_token.squeeze().tolist()
#                         if 'mean' not in features_length:
#                             features_length['mean'] = len(mean_features)
#                         features.extend(mean_features)

#                     if segment_means is not None:
#                         # In the new version, we keep each segment mean as a separate list
#                         for seg, segment_mean in enumerate(segment_means):
#                             features.extend(segment_mean)
#                             if f'segment{seg}_mean' not in features_length:
#                                 features_length[f'segment{seg}_mean'] = len(segment_mean)

#                     # create the column names only for the first item
#                     if columns is None:
#                         columns = []
#                         for feature_type, length in features_length.items():
#                             for k in range(length):
#                                 columns.append(f"ESM2_{feature_type}{k}")

#                     # Create DataFrame for this batch
#                     result = pd.DataFrame([features], columns=columns, index=[idx])
#                     all_results.append(result)

#                 del inputs, outputs
#                 # Clear and synchronize only when CUDA is active.
#                 if self.device.type == 'cuda':
#                     torch.cuda.empty_cache()
#                     torch.cuda.synchronize()

#         # Combine all batch results outside the loop
#         X_outcome = pd.concat(all_results, axis=0)

#         print(f'Features dimensions: {features_length}')

#         # Combine X_input and X_outcome along axis 1
#         combined_result = pd.concat([X_input, X_outcome], axis=1)
#         return combined_result
    


    
#     # Extract layer-specific phosphorylation-site representations.
#     def get_layer_hidden_phosphorylation_position_feature(self, X_input, sequence_name='sequence', phosphorylation_positions='phosphorylation_positions', batch_size=32):
#         X_input = X_input.reset_index(drop=True)
#         # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)

#         # Group X_input by sequence
#         grouped_X_input = X_input.groupby(sequence_name)
#         sequence_to_indices = grouped_X_input.groups

#         # Pre-compute the number of features
#         num_features = self.model.config.hidden_size
#         columns = [f"ESM2_phospho_pos{k}" for k in range(num_features)]

#         # Create an empty DataFrame with the column names
#         X_outcome = pd.DataFrame(columns=columns)

#         with torch.no_grad():
#             for i in tqdm(range(0, len(grouped_X_input), batch_size), desc='batches for inference'):
#                 batch_sequences = list(islice(sequence_to_indices.keys(), i, i + batch_size))
#                 batch_grouped_sequences = {seq: X_input.loc[sequence_to_indices[seq]] for seq in batch_sequences}

#                 # Get the unique sequences in the batch
#                 inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True).to(self.device)
#                 outputs = self.model(**inputs)

#                 for j, sequence in enumerate(batch_sequences):
#                     sequence_indices = batch_grouped_sequences[sequence].index
#                     sequence_positions = batch_grouped_sequences[sequence][phosphorylation_positions].tolist()
#                     layer_hidden_state = self.get_layer_hidden_states(outputs)[j:j+1]

#                     for idx, position in zip(sequence_indices, sequence_positions):
#                         position = int(position)  # Make sure position is an integer
#                         position_feature = layer_hidden_state[:, position, :]  # Removed +1 since the sequence starts from 1, and consider removing the cls token
#                         features = position_feature.squeeze().tolist()

#                         # Add the new row to the DataFrame
#                         X_outcome.loc[idx] = features

#                 del inputs, outputs
#                 # Clear and synchronize only when CUDA is active.
#                 if self.device.type == 'cuda':
#                     torch.cuda.empty_cache()
#                     torch.cuda.synchronize()


#         # Print the dimension of the final phosphorylation features
#         print(f"The dimension of the final phosphorylation features is: {X_outcome.shape[1]}")

#         # Combine X_input and X_outcome along axis 1
#         combined_result = pd.concat([X_input, X_outcome], axis=1)
#         return combined_result
    
 
#     def get_amino_acid_representation(self, sequence, amino_acid, position):
#         # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)
        
#         # Check if the amino acid at the given position matches the input
#         if sequence[position - 1] != amino_acid:
#             raise ValueError(f"The amino acid at position {position} is not {amino_acid}.")

#         # Convert the sequence to input tensors
#         inputs = self.tokenizer([sequence], return_tensors="pt", padding=True).to(self.device)
        
#         # Get the model outputs
#         with torch.no_grad():
#             outputs = self.model(**inputs)
        
#         # Get the layer hidden state
#         layer_hidden_state = self.get_layer_hidden_states(outputs)
        
#         # Get the tokens from the input ids
#         tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        
#         # Get the position of the amino acid token in the tokens list
#         # We add 1 to the position to account for the CLS token at the start
#         token_position =  position if amino_acid == tokens[position] else -1
        
#         if token_position == -1:
#             raise ValueError(f"The token for amino acid {amino_acid} could not be found in the tokenized sequence.")
        
#         # Get the feature vector for the amino acid
#         amino_acid_features = layer_hidden_state[:, token_position, :].squeeze().tolist()
        
        
#         # Get the feature vector for the amino acid
#         amino_acid_features = layer_hidden_state[:, token_position, :].squeeze().tolist()
        
#         # Prepare the DataFrame
#         feature_names = [f"ESM2_{k}" for k in range(len(amino_acid_features))]
#         amino_acid_features_df = pd.DataFrame(amino_acid_features, index=feature_names, columns=[amino_acid]).T

#         return amino_acid_features_df
