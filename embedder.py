import torch
import clip
import numpy as np
from typing import List

class Embedder:
    def __init__(self, model_name="clip"):
        # Default embedder is clip, but others
        # can be used. 
        self.model_name = model_name.lower()
        self.model, self.preprocess = self.load_model()

    def load_model(self):
        if self.model_name == 'clip':
            device = "cuda" if torch.cuda.is_available() else "cpu" # will be cpu unless we use colab or similar.
            model, preprocess = clip.load("ViT-B/32", device=device)
            self.device = device
            return model, preprocess
        
        # We can add more conditions to match other models here if we would like to do so. I've chonse CLIP to 
        # ensure that the code is as modular as possible (we will add images later). Unsure about CLIP's performance
        # on text only.

        else:
            raise ValueError("The model is not supported.")

    def encode(self, phrases: List[str]) -> np.ndarray:
        if self.model_name == 'clip':
            text = clip.tokenize(phrases).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text)
            embeddings = text_features.cpu().numpy()
            # print(embeddings.shape) # For 3 phrases, we have shape (3, 512)
        # if statement: Space for additional conditions / models

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        return embeddings

if __name__ == '__main__':
    embedder = Embedder()
    phrases = ["red dress, women's, medium", "blue jeans, men's, large", "gray jeans, women's, large"]
    embeddings = embedder.encode(phrases)