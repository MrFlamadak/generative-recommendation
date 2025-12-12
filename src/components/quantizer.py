import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualVQ

class Encoder(nn.Module):
    def __init__(self, input_dim=384, latent_dim=32, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_to_latent = nn.Linear(hidden_dim, latent_dim)  
    
    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        latent = self.fc_to_latent(h2)
        return latent
    
    
class Decoder(nn.Module):
    def __init__(self, latent_dim=32, output_dim=384, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_to_output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z):
        h1 = F.relu(self.fc1(z))
        h2 = F.relu(self.fc2(h1))
        output = self.hidden_to_output(h2)
        return output
    
    
class RQVAE(nn.Module):
    def __init__(self, input_dim=384, latent_dim=32, hidden_dim=256, codebook_size=512, num_quantizers=4):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, input_dim, hidden_dim)
        self.residual_vq = ResidualVQ(dim=latent_dim, num_quantizers=num_quantizers, codebook_size=codebook_size)

    def forward(self, x):
        z = self.encoder(x)
        z_quantized, indices, commitment_loss = self.residual_vq(z)
        x_recon = self.decoder(z_quantized)
        return x_recon, indices, commitment_loss
    
    def encode_to_semantic_ids(self, x):
        # Retrieve the semantic IDs given an input embedding
        z = self.encoder(x)
        _, indices, _ = self.residual_vq(z)
        return indices
    
def rqvae_loss(x, x_recon, commitment_loss):
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    total_loss = recon_loss + commitment_loss.mean()
    return total_loss
