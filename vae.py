import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32, hidden_dim=256):
        super().__init__()
        
        pass
    
    
    def forward(self, x):
        pass
    
    
class Decoder(nn.Module):
    def __init__(self, latent_dim=32, output_dim=784, hidden_dim=256):
        super().__init__()
        
        pass
    
    
    def forward(self, x):
        pass
    
    
class VAE(nn.Module):
    def __init__(self, input_dim=384, latent_dim=32, hidden_dim=256):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, input_dim, hidden_dim)
        
        
    def reparametrize(self, mu, std):
        pass
    
    
    def forward(self, x):
        mu = self.encoder(x)
        # skipping reparametrization for now
        x_recon = self.decoder(mu)
        
        return x_recon, mu
    
    
def vae_loss(x, x_recon, mu, logvar=0):
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss