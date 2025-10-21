import torch
import clip
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="clip"):
        # Default embedder is clip, but others can be used as well.
        self.model_name = model_name.lower()
        self.model, self.preprocess = self.load_model()

    def load_model(self):
        if self.model_name == 'clip':
            device = "cuda" if torch.cuda.is_available() else "cpu" # likely cpu
            model, preprocess = clip.load("ViT-B/32", device=device)
            self.device = device
            return model, preprocess
        
        elif self.model_name == 'sbert':
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model, None
        # We can add support for other models and embedders here.
        else:
            raise ValueError("The model is not supported.")

    def encode(self, phrases: List[str]) -> np.ndarray:
        if self.model_name == 'clip': # 512 dimensional embeddings
            text = clip.tokenize(phrases).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text)
            embeddings = text_features.cpu().numpy()
        elif self.model_name == 'sbert': # 384 dimensional embeddings
            embeddings = self.model.encode(phrases)
        else:
            raise ValueError("The model is not supported.")

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        return embeddings

if __name__ == '__main__':
    embedder = Embedder(model_name="sbert") # use model_name="sbert" to use SBERT
    phrases = ["red dress, women's, medium", "blue jeans, men's, large", "gray jeans, women's, large"]
    embeddings = embedder.encode(phrases)