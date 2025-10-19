import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
    
class VAE(nn.Module):
    def __init__(self, input_dim=384, latent_dim=32, hidden_dim=256):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, input_dim, hidden_dim)
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon
    
    def vae_loss(x, x_recon):
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        # TODO: Commitment loss from Quantizer, then add together
        return recon_loss
