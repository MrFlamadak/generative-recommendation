import torch
import torch.nn.functional as F
from rqvae import RQVAE, rqvae_loss

def train_rqvae(embeddings):
    # Handle code for dataset here, probably use dataloader etc? 

    # handle training loop here
    num_epochs = 100
    model = RQVAE(input_dim=embeddings.shape[1],
                  latent_dim=32,
                  hidden_dim=256,
                  codebook_size=512,
                  num_quantizers=4)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        total_recon_loss = 0
        total_commitment_loss = 0
        for _, (batch_embeddings, ) in enumerate(dataloader):

            x_recon, semantic_ids, commitment_loss = model(batch_embeddings)

            loss = rqvae_loss(batch_embeddings, x_recon, commitment_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            recon_loss = F.mse_loss(x_recon, batch_embeddings, reduction="mean")
            total_recon_loss += recon_loss.item()
            total_commitment_loss += commitment_loss.mean().item()

        if (epoch+1) % 10 == 0:
            average_loss = total_loss / len(dataloader)
            average_recon_loss = total_recon_loss / len(dataloader)
            average_commitment_loss = total_commitment_loss / len(dataloader)

            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Total loss: {average_loss:.4f}')
            print(f'Reconstruction loss: {average_recon_loss:.4f}')
            print(f'Commitment loss: {average_commitment_loss:.4f}')
    
    return model

if __name__ == '__main__':
    # get embeddings
    # train model
    # save trained model?
    pass