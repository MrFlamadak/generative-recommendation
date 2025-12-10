import torch
import clip
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel

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
        elif self.model_name == 'bert-large':
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
            model = BertModel.from_pretrained("bert-large-uncased")
            model.to(device)
            model.eval()
            self.device = device
            self.tokenizer = tokenizer
            return model, None
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
        elif self.model_name == 'bert-large':
            embeddings = self.get_bert_embeddings(phrases)
        else:
            raise ValueError("The model is not supported.")

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        return embeddings
    
    def get_bert_embeddings(self, phrases):
        embeddings = []
        with torch.no_grad():
            for phrase in phrases:
                inputs = self.tokenizer(
                    phrase, 
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                outputs = self.model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embedding[0])
        return np.array(embeddings)
                

if __name__ == '__main__':
    embedder = Embedder(model_name="bert-large") # use model_name="sbert" to use SBERT
    phrases = ["red dress, women's, medium", "blue jeans, men's, large", "gray jeans, women's, large"]
    embeddings = embedder.encode(phrases)
    print(embeddings.shape)