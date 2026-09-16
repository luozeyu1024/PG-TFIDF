import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import logging
import torch.nn.init as init
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import re
import argparse

from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(current_dir, '../'))

sys.path.append(project_root)

import random
from feature_utils import feature_column_utils
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



# Create the ArgumentParser
parser = argparse.ArgumentParser(description="VAE Pretraining")

# Add command-line arguments
parser.add_argument('--open_path', type=str, required=True,
                    help='Path to the input data')
parser.add_argument('--save_path_outer', type=str,
                    required=True,
                    help='Outer path to save output results')
parser.add_argument('--open_path_reference_data', type=str,
                    required=True,
                    help='Path to the conference data')
parser.add_argument('--file_prefix', type=str, default='scRNA-seq_panglao_0_1_Random_0_3',
                    help='File prefix for saving results')
parser.add_argument("--best_parameter_name", type=str, default='best_hyperparameters.xlsx', help="File name for best hyperparameters.")
parser.add_argument('--epoch_start_for_loss_plot_only', type=int, default=1, help='Epoch start for loss plot only')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--device_choose', type=str, default='cuda:0', help='cpu choose')

parser.add_argument("--random_choose", type=strict_str2bool, default=True, help="")
parser.add_argument('--random_choose_num', type=float, default=0.3, help='')

parser.add_argument('--feature_columns_file', type=str,
                    default='Eukaryota_Reviewed_Protein level_uniprotkb_Eukaryota_AND_reviewed_true_A_2025_02_28_feature_columns.xlsx',
                    help='Excel file containing feature column names')
parser.add_argument('--feature_logits_file', type=str,
                    default='Eukaryota_Reviewed_Protein level_uniprotkb_Eukaryota_AND_reviewed_true_A_2025_02_28_logits.npy',
                    help='Numpy file containing feature logits')
parser.add_argument("--model_tag", type=str, default="ESM2",
                   help="Model feature column prefix (case-insensitive), e.g. ESM2 or ESMC")
parser.add_argument("--need_tokens", default="CLS,EOS,SEG",
               help="Token groups, e.g. CLS,EOS,MEAN,SEG or SEG:0,3")

# Parse command-line arguments
args = parser.parse_args()

# Assemble paths
save_path = os.path.join(args.save_path_outer, args.file_prefix)

# Print all argument values to verify parsing
# Assign the arguments to variables
open_path = args.open_path
save_path_outer = args.save_path_outer
open_path_reference_data = args.open_path_reference_data
file_prefix = args.file_prefix
best_parameter_name = args.best_parameter_name
epoch_start_for_loss_plot_only = args.epoch_start_for_loss_plot_only
batch_size = args.batch_size

device_choose = args.device_choose

model_tag  = args.model_tag.upper()
# tag_prefix  = f"{model_tag}_"


# Now you can use these variables in your script
print(f"Open Path: {open_path}")
print(f"Save Path Outer: {save_path_outer}")
print(f"Open Path Conference Data: {open_path_reference_data}")
print(f"File Prefix: {file_prefix}")
print(f"Best Parameter Name: {best_parameter_name}")
print(f"Save Path: {save_path}")
print(f"Epoch Start for Loss Plot Only: {epoch_start_for_loss_plot_only}")
print(f"Batch Size: {batch_size}")



# Create the directory if it does not exist
if not os.path.exists(save_path):
    os.makedirs(save_path)


random_choose_num = args.random_choose_num ##Fraction of data randomly selected for hyperparameter screening

random_choose=args.random_choose  #Whether to perform hyperparameter screening




# =============================================================================
# =======[Modified section starts]=======

ESM_feature = feature_column_utils.extract_esm_features(
    open_path=open_path,
    feature_columns_file=args.feature_columns_file,
    feature_logits_file=args.feature_logits_file,
    model_tag=model_tag,
    need_tokens=args.need_tokens
)

# =======[Modified section ends]=======
# =============================================================================





# Check data for NaN values
print("DATA is containing NA?: ", np.isnan(ESM_feature).any())



# Generate random indices for training
if random_choose:
    num_samples = ESM_feature.shape[0]
    num_train_samples = int(num_samples * random_choose_num)  # Select 30% of the data for training
    random_indices = np.random.choice(num_samples, size=num_train_samples, replace=False)

    # Select training data using random indices
    train_features = ESM_feature[random_indices,:]
else:
    train_features = ESM_feature

# Define the DNN model

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)


# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.FileHandler(save_path+r'\cross_validation_training_log.log'), logging.StreamHandler()])
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.FileHandler(os.path.join(save_path, 'cross_validation_training_log.log')), logging.StreamHandler()])
logger = logging.getLogger()


def train(model, device, train_loader, optimizer):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        data = batch[0].to(device)
        optimizer.zero_grad()
        # VQVAE forward returns (recon_batch, z_e, z_q)
        recon_batch, z_e, z_q = model(data)
        loss = model.loss_function(recon_batch, data, z_e, z_q)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss


# Validation function
def validate(model, device, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            data = batch[0].to(device)
            recon_batch, z_e, z_q = model(data)
            loss = model.loss_function(recon_batch, data, z_e, z_q)
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    return avg_loss

#
# ##vision1
##KF5_scRNA-seq_panglao_Res-VAEpretraining.py




# Define stratified cross-validation
kf =KFold(n_splits=10, shuffle=True, random_state=42)
folds = list(kf.split(X=train_features))

# Get the input feature dimension
input_dim = train_features.shape[1]

# Read the label mapping dictionary
best_params = pd.read_excel(os.path.join(open_path_reference_data, best_parameter_name ))
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

# Added: configure VQ-VAE hyperparameters
num_embeddings = int(best_params['num_embeddings'])
commitment_cost = float(best_params['commitment_cost'])

# # Extract labels for stratified sampling
# labels = train_info_df['Label'].values

# Check CUDA availability and set the device

device = torch.device(device_choose if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# hidden_dim_layer1 = 3072
# fold_min_loss =[]


#Create the model,
for fold, (train_idx, val_idx) in enumerate(folds):

    # For train_dataset
    train_features_sub = train_features[train_idx]

    train_dataset = TensorDataset(torch.Tensor(train_features_sub))

    # For val_dataset
    val_features_sub = train_features[val_idx]
    val_dataset = TensorDataset(torch.Tensor(val_features_sub))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            drop_last=False)

    # Create the VQ-VAE model [change: replace ContinuousResidualVAE with VQVAE and pass the new arguments]
    model = VQVAEEMA(
        input_dim=input_dim,
        Encoder_layer_dims=Encoder_layer_dims,
        Decoder_layer_dims=Decoder_layer_dims,
        z_dim=z_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost,
        reduction='mean'
    ).to(device)

    # Define the loss function and optimizer

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=int(num_epochs / 4), eta_min=learning_rate * (1e-2))


    # Round to integer epochs for i=1,2,3,4,5; i=5 is the final epoch
    save_epochs = [int(round(num_epochs * i / 5)) for i in range(1, 6)]
    # Deduplicate and sort (rounding may produce duplicates)
    save_epochs = sorted(set(save_epochs))

    for epoch in range(num_epochs):
        train_loss = train(model, device, train_loader, optimizer)
        val_loss = validate(model, device, val_loader)
        # Output through both logger and print

        message = f" Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"

        logger.info(message)
        print(message)

        ## update learning rate
        scheduler.step()


        # Save a model checkpoint if the current 1-based epoch is a scheduled save point
        if (epoch + 1) in save_epochs:
            checkpoint_path = os.path.join(save_path, f'checkpoint_fold{fold + 1}_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch + 1} to {checkpoint_path}")






# Function to extract log data from a logger file
def extract_log_data(file_path):
    with open(file_path, 'r') as file:
        log_data = file.readlines()

    data = []
    pattern = re.compile(r"Fold (\d+), Epoch (\d+)/(\d+), Train Loss: ([\d.]+), Val Loss: ([\d.]+)")

    for line in log_data:
        match = pattern.search(line)
        if match:
            fold = int(match.group(1))
            epoch = int(match.group(2))
            train_loss = float(match.group(4))
            val_loss = float(match.group(5))
            data.append([fold, epoch, train_loss, val_loss])

    return pd.DataFrame(data, columns=['Fold', 'Epoch', 'Train Loss', 'Val Loss'])


# Path to the logger file
file_path = (os.path.join(save_path, 'cross_validation_training_log.log') ) # Replace with the actual path to your logger file

# Extract log data
df = extract_log_data(file_path)

# Remove the first epoch from each fold
df = df[df['Epoch'] > epoch_start_for_loss_plot_only]
print('epoch_start_for_loss_plot_only:', epoch_start_for_loss_plot_only)
# Calculate mean and standard deviation for each epoch across folds
train_loss_mean = df.groupby('Epoch')['Train Loss'].mean()
train_loss_std = df.groupby('Epoch')['Train Loss'].std()
val_loss_mean = df.groupby('Epoch')['Val Loss'].mean()
val_loss_std = df.groupby('Epoch')['Val Loss'].std()

# Plotting the train and validation loss curves
plt.figure(figsize=(12, 6))

# Plot Train Loss
plt.subplot(1, 2, 1)
plt.plot(train_loss_mean.index, train_loss_mean, label='Train Loss')
plt.fill_between(train_loss_mean.index, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, alpha=0.2)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Validation Loss
plt.subplot(1, 2, 2)
plt.plot(val_loss_mean.index, val_loss_mean, label='Validation Loss', color='orange')
plt.fill_between(val_loss_mean.index, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, alpha=0.2,
                 color='orange')
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()

# Save the plot as a PDF
plt.savefig(os.path.join(save_path,f"{file_prefix}model_Loss_plot.pdf"),dpi = 1000, format='pdf')


# Plotting the combined train and validation loss curves
plt.figure(figsize=(8, 6))

plt.plot(train_loss_mean.index, train_loss_mean, label='Train Loss')
plt.fill_between(train_loss_mean.index, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, alpha=0.2)
plt.plot(val_loss_mean.index, val_loss_mean, label='Validation Loss', color='orange')
plt.fill_between(val_loss_mean.index, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, alpha=0.2, color='orange')

plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()

# Save the plot as a PDF
plt.savefig(os.path.join(save_path,f"{file_prefix}combined_model_Loss_plot.pdf"),dpi = 1000, format='pdf')
