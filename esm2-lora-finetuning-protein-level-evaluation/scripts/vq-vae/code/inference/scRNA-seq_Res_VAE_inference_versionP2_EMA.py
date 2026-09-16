
import numpy as np
import pandas as pd
import os
import torch

from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics.pairwise import cosine_distances


from tqdm import tqdm
import argparse

from matplotlib.ticker import MaxNLocator

import random

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(current_dir, '../'))

sys.path.append(project_root)

import torch.nn.functional as F
from VAE_model_utils.VAE_model import VQVAEEMA





def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)


def strict_str2bool(v):
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Only "True" or "False" are allowed.')


# Create the parser
parser = argparse.ArgumentParser(description="Set up and use command-line arguments for model handling.")

# Add arguments
parser.add_argument("--open_path", type=str, required=True, help="Path to open input data.")
parser.add_argument("--save_path", type=str, required=True, help="Path to save output data.")
parser.add_argument("--open_path_reference_data", type=str, required=True, help="Path to open conference data.")
parser.add_argument("--file_prefix", type=str, default='scRNA-seq_panglao_0_1_', help="Prefix for files to handle.")
parser.add_argument("--model_parameters_file", type=str, default='model_parameters.pth', help="File name for storing model parameters.")
parser.add_argument("--best_parameter_name", type=str, default='best_hyperparameters.xlsx', help="File name for best hyperparameters.")
parser.add_argument("--execute_model_post_analysis", type=bool, default=True, help="Flag to execute model post-analysis.")
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--device_choose', type=str, default='cuda:0', help='cpu choose')

# Parse the arguments
args = parser.parse_args()

# Assign the arguments to variables
open_path = args.open_path
save_path = args.save_path
open_path_reference_data = args.open_path_reference_data
file_prefix = args.file_prefix
model_parameters_file = args.model_parameters_file
best_parameter_name = args.best_parameter_name
execute_model_post_analysis = args.execute_model_post_analysis
batch_size = args.batch_size
device_choose = args.device_choose
# REC_beta = args.REC_beta

# Now you can use these variables in your script
print(f"Open Path: {open_path}")
print(f"Save Path: {save_path}")
print(f"Open Path Conference Data: {open_path_reference_data}")
print(f"File Prefix: {file_prefix}")
print(f"Model Parameters File: {model_parameters_file}")
print(f"Best Parameter Name: {best_parameter_name}")
print(f"Execute Model Post Analysis: {execute_model_post_analysis}")



# Create the directory if it does not exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

##input_data
filename = os.listdir(open_path)[0]

scFoundation_Embedding = np.load(os.path.join(open_path, filename))


# Check data for NaN values
print("DATA is containing NA?: ", np.isnan(scFoundation_Embedding).any())


# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)




# Get the input feature dimension
input_dim = scFoundation_Embedding.shape[1]

# Read the label mapping dictionary
best_params = pd.read_excel(os.path.join(open_path_reference_data,best_parameter_name ))
best_params = dict(zip(best_params.iloc[:,0], best_params.iloc[:,1]))

##capture the best hyperparamter
learning_rate = best_params['learning_rate']
weight_decay = best_params['weight_decay']
num_epochs = int(best_params['num_epochs'])
z_dim = int(best_params['z_dim'])
hidden_dim_layer0 = int(best_params['hidden_dim_layer0'])

hidden_dim_layer_out_Z = int(best_params['hidden_dim_layer_out_Z'])

num_blocks = best_params['num_blocks']
# layer_dims = [int(best_params[f'layer_{i+1}_dim']) for i in range(int(num_blocks)-1)]

layer_encoder_dims = [int(best_params[f'layer_encoder_{i+1}_dim']) for i in range(int(num_blocks)-1)]

layer_decoder_dims = [int(best_params[f'layer_decoder_{i+1}_dim']) for i in range(int(num_blocks)-1)]



Encoder_layer_dims =[input_dim]+ [hidden_dim_layer0] + layer_encoder_dims + [hidden_dim_layer_out_Z]

Decoder_layer_dims = [z_dim] + layer_decoder_dims + [hidden_dim_layer0]



device = torch.device(device_choose if torch.cuda.is_available() else "cpu")
print("Using device:", device)



##load model

# Added: configure VQ-VAE hyperparameters
num_embeddings = int(best_params['num_embeddings'])
commitment_cost = float(best_params['commitment_cost'])

load_model = VQVAEEMA(
    input_dim=input_dim,
    Encoder_layer_dims=Encoder_layer_dims,
    Decoder_layer_dims=Decoder_layer_dims,
    z_dim=z_dim,
    num_embeddings=num_embeddings,
    commitment_cost=commitment_cost,
    reduction='mean'
).to(device)

# print(load_model)
#


# Load the state dictionary
loaded_state_dict = torch.load(os.path.join(open_path_reference_data, model_parameters_file))


# # Adjust key names
# fixed_state_dict = fix_state_dict(loaded_state_dict)

load_model.load_state_dict(loaded_state_dict)

inference_dataset = TensorDataset(torch.Tensor(scFoundation_Embedding))
inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False,
                              drop_last=False)


## ------------------ Infer continuous latent representations z_e and codebook labels ------------------ ##
# [Change A]: call get_model_inference_info to obtain continuous latent representations and codebook labels
latent_vectors_e = []      # Store continuous latent representations z_e
codebook_labels = []       # Store the codebook index for each sample (encoding_indices)
sample_indices = []        # Record original input indices


running_rec_loss = 0.0
running_codebook_loss = 0.0
running_commit_loss = 0.0
running_total_loss = 0.0
total_seen_samples = 0


load_model.eval()  # Switch to evaluation mode
with torch.no_grad():
    for i, data in enumerate(tqdm(inference_loader, desc="Processing Data")):
        data_batch = data[0].to(device)

        recon_x, z_e_batch, z_q_batch = load_model(data_batch)

        # Call get_model_inference_info to obtain continuous latent variables z_e and codebook labels
        z_e, encoding_indices = load_model.get_model_inference_info(data_batch)
        latent_vectors_e.append(z_e.cpu().detach().numpy())
        codebook_labels.append(encoding_indices.cpu().detach().numpy())
        # Record sample indices, assuming consecutive data order within each batch
        batch_indices = np.arange(i * batch_size, min((i+1) * batch_size, scFoundation_Embedding.shape[0]))
        sample_indices.append(batch_indices)

        # ====== Compute the three losses ======
        rec_loss_batch = F.mse_loss(recon_x, data_batch, reduction='mean')
        codebook_loss_batch = F.mse_loss(z_q_batch, z_e_batch.detach(), reduction='mean')
        commit_loss_batch = F.mse_loss(z_e_batch, z_q_batch.detach(), reduction='mean')
        total_loss_batch = (rec_loss_batch
                            + codebook_loss_batch
                            + load_model.commitment_cost * commit_loss_batch)

        # ====== Accumulate ======
        bs = data_batch.size(0)
        running_rec_loss += rec_loss_batch.item() * bs
        running_codebook_loss += codebook_loss_batch.item() * bs
        running_commit_loss += commit_loss_batch.item() * bs
        running_total_loss += total_loss_batch.item() * bs
        total_seen_samples += bs

mean_rec_loss = running_rec_loss / total_seen_samples
mean_codebook_loss = running_codebook_loss / total_seen_samples
mean_commit_loss = running_commit_loss / total_seen_samples
mean_total_loss = running_total_loss / total_seen_samples


latent_vectors_e = np.concatenate(latent_vectors_e, axis=0)
codebook_labels = np.concatenate(codebook_labels, axis=0)
sample_indices = np.concatenate(sample_indices, axis=0)

## ------------------ Save continuous latent representations and discrete codebook information ------------------ ##
# Save continuous latent representations z_e as .npy for optional later use
np.save(os.path.join(save_path, f'{file_prefix}_VAE_ContinuousEmbedding.npy'), latent_vectors_e)


# ------------------ Retrieve and save the codebook matrix ------------------ #
# Create a dummy input to satisfy the function signature: a zero tensor of shape [1, input_dim]
dummy_input = torch.zeros((1, input_dim)).to(device)

# Call get_model_inference_z to retrieve the full codebook matrix
codebook_matrix = load_model.get_model_inference_z(dummy_input, output_codebook=True)

# Convert the codebook matrix to a NumPy array
codebook_matrix_np = codebook_matrix.cpu().detach().numpy()

# Save as .npy with file_prefix in the filename
save_file = os.path.join(save_path, f'{file_prefix}_codebook_matrix.npy')
np.save(save_file, codebook_matrix_np)
print(f"Codebook matrix saved to {save_file}")


## ------------------ Run downstream VAE analysis ------------------ ##
if execute_model_post_analysis:
    import matplotlib.pyplot as plt
    import umap  # Use the umap module instead of umap.plot




    # ------------------ 1. Apply UMAP to continuous latent space z_e and label codebook indices ------------------ #
    umap_reducer = umap.UMAP(random_state=0, min_dist=0.5, n_neighbors=15, n_jobs=1)
    embedding_2d = umap_reducer.fit_transform(latent_vectors_e)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1],
                          c=codebook_labels, cmap='tab20', s=5)
    plt.colorbar(scatter, label='Codebook Index')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    umap_fig_path = os.path.join(save_path, f'{file_prefix}_z_e_UMAP.pdf')
    plt.savefig(umap_fig_path)
    plt.close()
    print(f"UMAP of continuous latent space saved to {umap_fig_path}")

    df_z_e = pd.DataFrame(
        embedding_2d,
        columns=['UMAP1', 'UMAP2']
    )
    df_z_e.to_excel(os.path.join(save_path, f'{file_prefix}_z_e_UMAP.xlsx'))

    # =========== 1B.(Added) Apply UMAP to the original feature matrix and label codebook indices =========== #
    # [Change B]: added block starts
    umap_reducer_orig = umap.UMAP(random_state=0, min_dist=0.5,
                                  n_neighbors=15, n_jobs=1)
    embedding_orig_2d = umap_reducer_orig.fit_transform(scFoundation_Embedding)

    plt.figure(figsize=(12, 10))
    scatter_orig = plt.scatter(embedding_orig_2d[:, 0], embedding_orig_2d[:, 1],
                               c=codebook_labels, cmap='tab20', s=5)
    plt.colorbar(scatter_orig, label='Codebook Index')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    orig_umap_fig_path = os.path.join(save_path, f'{file_prefix}_origFeat_UMAP.pdf')
    plt.savefig(orig_umap_fig_path)
    plt.close()
    print(f"UMAP of original feature matrix saved to {orig_umap_fig_path}")

    df_orig = pd.DataFrame(
        embedding_orig_2d,
        columns=['UMAP1', 'UMAP2']
    )
    df_orig.to_excel(os.path.join(save_path, f'{file_prefix}_origFeat_UMAP.xlsx'))



    # ------------------ 1.C. Predefined cosine similarity matrix + Dense UMAP ------------------ #

    def compute_and_plot_umap(X, labels, file_prefix, save_path, max_samples=5000):
        # 1. Randomly sample indices
        N = X.shape[0]
        if N > max_samples:
            idx = np.random.choice(N, max_samples, replace=False)
        else:
            idx = np.arange(N)
        X_sub = X[idx]
        labels_sub = labels[idx]

        # 2. Compute the cosine distance matrix
        dist_sub = cosine_distances(X_sub)

        # 3. UMAP (Precomputed distances + DensMAP)
        umap_reducer = umap.UMAP(
            metric='precomputed',
            densmap=True,
            dens_lambda=2.0,
            min_dist=0.5,
            n_neighbors=15,
            random_state=0,
            n_jobs=1
        )
        embedding_2d = umap_reducer.fit_transform(dist_sub)

        # 4. Plot and save
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            embedding_2d[:, 0], embedding_2d[:, 1],
            c=labels_sub, cmap='tab20', s=5
        )
        plt.colorbar(scatter, label='Codebook Index')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        fig_path = os.path.join(save_path, f'{file_prefix}_precomputed_densmap_UMAP.pdf')
        plt.savefig(fig_path)
        plt.close()


    # Apply to the z_e embedding space
    compute_and_plot_umap(
        X=latent_vectors_e,
        labels=codebook_labels,
        file_prefix=f'{file_prefix}_z_e',
        save_path=save_path
    )

    # Apply to the original feature matrix
    compute_and_plot_umap(
        X=scFoundation_Embedding,
        labels=codebook_labels,
        file_prefix=f'{file_prefix}_origFeat',
        save_path=save_path
    )




    # ------------------ 2. Summarize the distribution of codebook indices ------------------ #


    # Count occurrences of each index
    unique_codes, counts = np.unique(codebook_labels, return_counts=True)

    # Create figure and axes objects
    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw a bar chart
    ax.bar(unique_codes, counts, color='skyblue')

    # Set the title and axis labels
    ax.set_title('Distribution of Codebook Assignments')
    ax.set_xlabel('Codebook Index')
    ax.set_ylabel('Frequency')

    # Force integer ticks on both axes
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Save the figure
    os.makedirs(save_path, exist_ok=True)
    bar_fig_path = os.path.join(save_path, f'{file_prefix}_codebook_distribution.pdf')
    fig.savefig(bar_fig_path, bbox_inches='tight')
    plt.close(fig)

    print(f"Codebook distribution plot saved to {bar_fig_path}")


    # ------------------ 3. Visualize the codebook embedding matrix ------------------ #
    codebook_embeddings = codebook_matrix_np
    umap_reducer_codebook = umap.UMAP(random_state=0, min_dist=0.5, n_neighbors=5, n_jobs=1)
    codebook_embedding_2d = umap_reducer_codebook.fit_transform(codebook_embeddings)
    plt.figure(figsize=(8, 6))
    plt.scatter(codebook_embedding_2d[:, 0], codebook_embedding_2d[:, 1], c='red', s=50)
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.title('Codebook Embedding Visualization')
    codebook_fig_path = os.path.join(save_path, f'{file_prefix}_codebook_UMAP.pdf')
    plt.savefig(codebook_fig_path)
    plt.close()
    print(f"Codebook embedding visualization saved to {codebook_fig_path}")

    df_codebook_embedding_2d = pd.DataFrame(
        codebook_embedding_2d,
        columns=['UMAP1', 'UMAP2']
    )
    df_codebook_embedding_2d.to_excel(os.path.join(save_path, f'{file_prefix}_codebook_UMAP.xlsx'))


    # ------------------ 4. Save codebook labels and input indices ------------------ #
    df_labels = pd.DataFrame({
        "Sample_Index": sample_indices,
        "Codebook_Label": codebook_labels
    })
    excel_path = os.path.join(save_path, f'{file_prefix}_codebook_labels.xlsx')
    df_labels.to_excel(excel_path, index=False)
    print(f"Codebook labels with sample indices saved to {excel_path}")


     # ------------------ 5. Compute Shannon entropy, perplexity, and the number of active codes ------------------ #
    

    # 5.1 Basic statistics
    N = scFoundation_Embedding.shape[0]   # Total samples (protein sequence entries)
    K = num_embeddings                    # Total codebook capacity
    counts_full = np.bincount(codebook_labels, minlength=K)   # Usage count of each code
    probs = counts_full / N                                  # Usage probability p_k
    active_mask = counts_full > 0
    m = active_mask.sum()                                    # Number of active codes

    # 5.2 Shannon entropy and perplexity
    nonzero_probs = probs[active_mask]                       # Avoid log(0)
    entropy = -np.sum(nonzero_probs * np.log(nonzero_probs)) # nats
    perplexity = np.exp(entropy)                             # Effective number of codes
    normalized_entropy = entropy / np.log(K) if K > 1 else 0 # Normalize to 0-1
    utilization = m / K                                      # Codebook utilization

    # 5.3 Print results
    print("\n=== Codebook Usage Statistics ===")
    print(f"Total samples (N)          : {N}")
    print(f"Total codebook size (K)    : {K}")
    print(f"Active codebook entries (m): {m}")
    print(f"Codebook utilization       : {utilization:.4f}")
    print(f"Shannon entropy  (H) [nat] : {entropy:.4f}")
    print(f"Perplexity (exp(H))        : {perplexity:.4f}")
    print(f"Normalized entropy H/ln(K) : {normalized_entropy:.4f}")
    print("==================================\n")

    # 5.4 Save to Excel
    stats_df = pd.DataFrame({
        "Total_samples_N"        : [N],
        "Total_codebook_K"       : [K],
        "Active_codebook_m"      : [m],
        "Utilization_m_div_K"    : [utilization],
        "Shannon_entropy_H_nat"  : [entropy],
        "Perplexity_expH"        : [perplexity],
        "Normalized_entropy"     : [normalized_entropy],
        "Loss_rec_mean": [mean_rec_loss],
        "Loss_codebook_mean": [mean_codebook_loss],
        "Loss_commit_mean": [mean_commit_loss],
        "Loss_total_mean": [mean_total_loss]
    })
    stats_path = os.path.join(save_path, f"{file_prefix}_codebook_stats.xlsx")
    stats_df.to_excel(stats_path, index=False)
    print(f"Codebook statistics saved to {stats_path}")
