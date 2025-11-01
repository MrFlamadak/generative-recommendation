import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

def rqvae_loss(x, x_recon, commitment_loss, beta=0.25):
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    c_loss = commitment_loss.mean()
    total_loss = recon_loss + beta * c_loss
    return total_loss

def train_rqvae_sanity_check(rqvae, embeddings, n_samples=100, epochs=5, batch_size=64, lr=1e-3, verbose=True):
    test_embeddings = embeddings[:n_samples].clone()
    if verbose:
        print(f"Testing with {n_samples} samples.")
    
    dataset = TensorDataset(test_embeddings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(rqvae.parameters(), lr=lr)
    rqvae.train()
    losses = []

    for epoch in range(epochs):
        epoch_losses = []
        for batch_idx, (batch_data,) in enumerate(dataloader):
            optimizer.zero_grad()
            x_recon, _, commitment_loss = rqvae(batch_data)
            loss = rqvae_loss(batch_data, x_recon, commitment_loss)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)

        if verbose:
            print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
    
    rqvae.eval()
    return rqvae

def train_rqvae_full(rqvae, embeddings, epochs=100, batch_size=256, lr=1e-3, save_path=None, checkpoint_freq=10, verbose=True, early_stopping_patience=15, min_delta=1e-4):
    dataset = TensorDataset(embeddings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(rqvae.parameters(), lr=lr)

    rqvae.train()
    losses = []
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_losses = []
        for batch_idx, (batch_data,) in enumerate(dataloader):
            optimizer.zero_grad()
            x_recon, _, commitment_loss = rqvae(batch_data)
            loss = rqvae_loss(batch_data, x_recon, commitment_loss)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)

        # Early stopping
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            patience_counter = 0
            if save_path:
                best_path = f'{save_path}_best.pth'
                torch.save(rqvae.state_dict(), best_path)
        else:
            patience_counter += 1

        if verbose and (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, loss = {avg_loss:.4f}')

        # checkpoint
        if save_path and (epoch+1) % checkpoint_freq == 0:
            checkpoint_path = f'{save_path}_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': rqvae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
        
            if verbose:
                print(f'saved checkpoint at epoch {epoch+1}')

        # check early stopping criteria
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # save not only best model so far, but also the final model
    if save_path:
        final_path = f'{save_path}_final.pth'
        torch.save(rqvae.state_dict(), final_path)

    rqvae.eval()
    return rqvae

def load_trained_rqvae(rqvae, model_path):
    rqvae.load_state_dict(torch.load(model_path))
    rqvae.eval()
    return rqvae