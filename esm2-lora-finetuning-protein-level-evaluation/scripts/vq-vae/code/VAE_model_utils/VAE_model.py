import torch
from torch import nn
from torch.nn import functional as F

class Swish(nn.Module):
    def __init__(self, trainable_beta=False, initial_beta=1.0):
        super(Swish, self).__init__()
        if trainable_beta:
            self.beta = nn.Parameter(torch.tensor(initial_beta))  # Trainable beta
        else:
            self.beta = initial_beta  # Fixed beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class VQVAE(nn.Module):
    class ResBlock(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            # Apply spectral regularization first
            self.fc = nn.Linear(in_dim, out_dim)
            self.bn = nn.BatchNorm1d(out_dim)
            self.swish = Swish(trainable_beta=True)
            init.kaiming_normal_(self.fc.weight, nonlinearity='leaky_relu')
            if in_dim != out_dim:
                self.downsample = nn.Linear(in_dim, out_dim)
                self.bn = nn.BatchNorm1d(out_dim)
                init.kaiming_normal_(self.downsample.weight, nonlinearity='leaky_relu')
            else:
                self.downsample = None

        def forward(self, x):
            out = self.swish(self.bn(self.fc(x)))
            if self.downsample is not None:
                x = self.downsample(x)
            return out + x

    def __init__(self, input_dim,  Encoder_layer_dims, Decoder_layer_dims,
                 z_dim, num_embeddings, commitment_cost=0.25, reduction='mean'):
        super().__init__()
        # Encoder: Similar to the original architecture, but maps to continuous latent space z_e
        self.Encoder_resblocks = nn.ModuleList()
        for i in range(len(Encoder_layer_dims) - 1):
            self.Encoder_resblocks.append(self.ResBlock(Encoder_layer_dims[i], Encoder_layer_dims[i + 1]))
        # Added: fully connected layer mapping to z_dim as the encoder output
        self.fc_enc = nn.Linear(Encoder_layer_dims[-1], z_dim)
        init.xavier_normal_(self.fc_enc.weight)

        # Added: codebook (embedding matrix) [change 1]
        self.num_embeddings = num_embeddings
        self.z_dim = z_dim
        self.embedding = nn.Embedding(num_embeddings, z_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

        # Decoder: Similar to the original architecture
        self.Decoder_resblocks = nn.ModuleList()
        for i in range(len(Decoder_layer_dims) - 1):
            self.Decoder_resblocks.append(self.ResBlock(Decoder_layer_dims[i], Decoder_layer_dims[i + 1]))
        self.fc_dec = nn.Linear(Decoder_layer_dims[-1], input_dim)
        init.xavier_normal_(self.fc_dec.weight)

        self.commitment_cost = commitment_cost  # Added: hyperparameter controlling the commitment loss weight [change 2]
        self.reduction = reduction
        if reduction not in ['mean', 'sum']:
            raise ValueError("Invalid reduction type. Expected 'mean' or 'sum', but got %s" % reduction)

    def encode(self, x):
        h = x
        for block in self.Encoder_resblocks:
            h = block(h)
        z_e = self.fc_enc(h)  # Continuous latent representation
        return z_e

    def quantize(self, z_e):
        # Compute Euclidean distances between z_e and codebook vectors [change 3]
        z_e_flat = z_e.view(-1, self.z_dim)  # shape: (batch, z_dim)
        z_e_sq = torch.sum(z_e_flat ** 2, dim=1, keepdim=True)
        embedding_sq = torch.sum(self.embedding.weight ** 2, dim=1)  # shape: (num_embeddings,)
        distances = z_e_sq + embedding_sq - 2 * torch.matmul(z_e_flat, self.embedding.weight.t())
        encoding_indices = torch.argmin(distances, dim=1) ##Equivalent to K=1 in KNN
        z_q = self.embedding(encoding_indices)
        return z_q, encoding_indices

    def decode(self, z):
        h = z
        for block in self.Decoder_resblocks:
            h = block(h)
        x_recon = self.fc_dec(h)
        return x_recon

    def forward(self, x):
        # Obtain the continuous latent representation from the encoder
        z_e = self.encode(x.view(-1, x.shape[1]))
        # Quantize z_e to obtain the discrete representation z_q [change 4]
        z_q, encoding_indices = self.quantize(z_e)
        # Use a straight-through estimator to pass gradients from z_q to z_e
        z_q_st = z_e + (z_q - z_e).detach()
        x_recon = self.decode(z_q_st)
        return x_recon, z_e, z_q

    def loss_function(self, recon_x, x, z_e, z_q):
        # Reconstruction loss
        if self.reduction == 'mean':
            rec_loss = F.mse_loss(recon_x, x.view(-1, x.shape[1]), reduction='mean')
        else:
            rec_loss = F.mse_loss(recon_x, x.view(-1, x.shape[1]), reduction='sum')
        # Codebook loss: Encourage the codebook to approach encoder outputs (no gradients to the encoder)
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction=self.reduction)
        # Commitment loss: Encourage encoder outputs to approach quantized values (stop gradients through quantized values)
        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction=self.reduction)
        total_loss = rec_loss + codebook_loss + self.commitment_cost * commitment_loss
        return total_loss

    def get_model_inference_z(self, x, output_codebook=False):
        """
        Extracts the discrete latent representation from the input x.
        In VQ-VAE, this function first computes the encoder output (z_e),
        then quantizes it to obtain the discrete latent representation (z_q),
        which is returned.

        If output_codebook is True, the function ignores x and returns the codebook
        matrix (with codebook indices as row indices) directly.

        Parameters:
        - x: Input data (required when output_codebook=False)
        - output_codebook (bool): If True, return the codebook matrix instead of sample-based mappings.

        Returns:
        - z_q: When output_codebook is False, return quantized sample latent representations;
               when output_codebook is True, return the full codebook matrix (shape [num_embeddings, z_dim]).
        """
        self.eval()  # Set evaluation mode
        with torch.no_grad():
            if output_codebook:
                # Return the codebook matrix directly
                return self.embedding.weight
            else:
                z_e = self.encode(x.view(-1, x.shape[1]))
                z_q, _ = self.quantize(z_e)
                return z_q

    def get_model_inference_info(self, x):
        """
        Extracts both the continuous latent representation (z_e) and
        the corresponding codebook indices (encoding_indices) from input x.
        This is used for visualization and analysis.
        """
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            z_e = self.encode(x.view(-1, x.shape[1]))
            _, encoding_indices = self.quantize(z_e)
        return z_e, encoding_indices
    









import torch.nn.init as init



# =======================  VQ‑VAE with EMA  ==================
class VQVAEEMA(nn.Module):
    class ResBlock(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.fc  = nn.Linear(in_dim, out_dim)
            self.bn  = nn.BatchNorm1d(out_dim)
            self.act = Swish(trainable_beta=True)
            init.kaiming_normal_(self.fc.weight, nonlinearity='leaky_relu')
            if in_dim != out_dim:
                self.downsample = nn.Linear(in_dim, out_dim)
                init.kaiming_normal_(self.downsample.weight, nonlinearity='leaky_relu')
            else:
                self.downsample = None

        def forward(self, x):
            out = self.act(self.bn(self.fc(x)))
            if self.downsample is not None:
                x = self.downsample(x)
            return out + x

    def __init__(self, input_dim, Encoder_layer_dims, Decoder_layer_dims,
                 z_dim, num_embeddings,
                 commitment_cost=0.25, reduction='mean',
                 ema_decay=0.99, ema_eps=1e-5):
        super().__init__()

        # --------- Encoder ---------
        self.Encoder_resblocks = nn.ModuleList()
        for i in range(len(Encoder_layer_dims) - 1):
            self.Encoder_resblocks.append(
                self.ResBlock(Encoder_layer_dims[i], Encoder_layer_dims[i + 1])
            )
        self.fc_enc = nn.Linear(Encoder_layer_dims[-1], z_dim)
        init.xavier_normal_(self.fc_enc.weight)

        # --------- Codebook buffers（EMA） ---------
        self.num_embeddings = num_embeddings
        self.z_dim          = z_dim

        embed_init = torch.randn(num_embeddings, z_dim)
        embed_init = embed_init / embed_init.norm(dim=1, keepdim=True)
        self.register_buffer("embedding", embed_init)
        self.embedding.requires_grad = False

        # EMA accumulators
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w",             self.embedding.clone())

        self.ema_decay = ema_decay
        self.ema_eps   = ema_eps

        # --------- Decoder ---------
        self.Decoder_resblocks = nn.ModuleList()
        for i in range(len(Decoder_layer_dims) - 1):
            self.Decoder_resblocks.append(
                self.ResBlock(Decoder_layer_dims[i], Decoder_layer_dims[i + 1])
            )
        self.fc_dec = nn.Linear(Decoder_layer_dims[-1], input_dim)
        init.xavier_normal_(self.fc_dec.weight)

        # --------- Other settings -------
        self.commitment_cost = commitment_cost
        self.reduction       = reduction
        if reduction not in ['mean', 'sum']:
            raise ValueError("reduction must be 'mean' or 'sum'")

    def quantize(self, z_e):
        # ── Inference mode: quantize only, without EMA updates ─────────────────────
        if not self.training:
            z_e_flat = z_e.view(-1, self.z_dim)
            distances = (
                    z_e_flat.pow(2).sum(1, keepdim=True)
                    - 2 * z_e_flat @ self.embedding.t()
                    + self.embedding.pow(2).sum(1)
            )
            encoding_indices = distances.argmin(1)
            z_q = self.embedding[encoding_indices]
            return z_q, encoding_indices
        # ─────────────────────────────────────────────────────

        # Training mode: quantize, then update the codebook using EMA
        z_e_flat = z_e.view(-1, self.z_dim)
        distances = (
                z_e_flat.pow(2).sum(1, keepdim=True)
                - 2 * z_e_flat @ self.embedding.t()
                + self.embedding.pow(2).sum(1)
        )
        encoding_indices = distances.argmin(1)
        z_q = self.embedding[encoding_indices]

        # ---------- EMA update ----------
        with torch.no_grad():
            enc_onehot = F.one_hot(encoding_indices, self.num_embeddings).type_as(z_e_flat)
            cluster_size = enc_onehot.sum(0)
            dw = enc_onehot.t() @ z_e_flat
            self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size * (1 - self.ema_decay))
            self.ema_w.mul_(self.ema_decay).add_(dw * (1 - self.ema_decay))
            n = (self.ema_cluster_size + self.ema_eps).unsqueeze(1)
            self.embedding.copy_(self.ema_w / n)
        # --------------------------------

        return z_q, encoding_indices

    def encode(self, x):
        h = x
        for blk in self.Encoder_resblocks:
            h = blk(h)
        return self.fc_enc(h)

    def decode(self, z):
        h = z
        for blk in self.Decoder_resblocks:
            h = blk(h)
        return self.fc_dec(h)

    def forward(self, x):
        z_e = self.encode(x.view(-1, x.shape[1]))
        # Exclude quantization and EMA updates from autograd
        with torch.no_grad():
            z_q, _ = self.quantize(z_e)
        # Straight‑through
        z_q_st  = z_e + (z_q - z_e).detach()
        x_recon = self.decode(z_q_st)
        return x_recon, z_e, z_q

    def loss_function(self, recon_x, x, z_e, z_q):
        rec = F.mse_loss(
            recon_x, x.view_as(recon_x),
            reduction=self.reduction
        )
        vq  = F.mse_loss(z_q, z_e.detach(),   reduction=self.reduction)
        com = F.mse_loss(z_e, z_q.detach(),   reduction=self.reduction)
        return rec + vq + self.commitment_cost * com


    def get_model_inference_z(self, x, output_codebook=False):
        """
        Extracts the discrete latent representation from the input x.
        In VQ-VAE, this function first computes the encoder output (z_e),
        then quantizes it to obtain the discrete latent representation (z_q),
        which is returned.

        If output_codebook is True, the function ignores x and returns the codebook
        matrix (with codebook indices as row indices) directly.

        Parameters:
        - x: Input data (required when output_codebook=False)
        - output_codebook (bool): If True, return the codebook matrix instead of sample-based mappings.

        Returns:
        - z_q: When output_codebook is False, return quantized sample latent representations;
               when output_codebook is True, return the full codebook matrix (shape [num_embeddings, z_dim]).
        """
        self.eval()  # Set evaluation mode
        with torch.no_grad():
            if output_codebook:
                # Return the codebook matrix directly
                return self.embedding
            else:
                z_e = self.encode(x.view(-1, x.shape[1]))
                z_q, _ = self.quantize(z_e)
                return z_q

    def get_model_inference_info(self, x):
        """
        Extracts both the continuous latent representation (z_e) and
        the corresponding codebook indices (encoding_indices) from input x.
        This is used for visualization and analysis.
        """
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            z_e = self.encode(x.view(-1, x.shape[1]))
            _, encoding_indices = self.quantize(z_e)
        return z_e, encoding_indices



