from embedder import Embedder
import numpy as np

def cos_similarity(embeddings):
    """
    Assumes that the embeddings are normalised.
    """
    return np.dot(embeddings, embeddings.T)

if __name__ == '__main__':
    phrases = ["red dress, women's, medium", "blue jeans, men's, small", "gray jeans, women's, large", "black jeans, women's, large"]
    embedders = ['clip', 'sbert']
    for embedder_name in embedders:
        embedder = Embedder(model_name=f'{embedder_name}')
        embeddings = embedder.encode(phrases)
        if embedder_name == 'clip':
            print("Cosine similarity between text embeddings using CLIP")
            print(cos_similarity(embeddings))
            print()
        else:
            print("Cosine similarity between text embeddings using Sentence BERT.")
            print(embedder.model.similarity(embeddings, embeddings))