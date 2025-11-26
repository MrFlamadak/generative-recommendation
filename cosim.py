import numpy as np
import random
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.stats import mannwhitneyu
import pandas as pd


# -----------------------------------------------------
# Helper: cosine similarity
# -----------------------------------------------------
def cosine_sim(a, b):
    return 1 - cosine(a, b)   # scipy cosine() returns distance


def sample_random_pairs(item_ids, similar_pairs, k, seed=42):
    random.seed(seed)

    similar_set = {tuple(sorted(p)) for p in similar_pairs}
    random_pairs = set()

    n = len(item_ids)

    while len(random_pairs) < k:
        a, b = random.sample(item_ids, 2)
        pair = tuple(sorted((a, b)))

        if pair not in similar_set and pair not in random_pairs:
            random_pairs.add(pair)

    return list(random_pairs)

def get_embeddings_dict(item_2_semantic, embeddings_matrix):

    embeddings_dict = {aid: embeddings for embeddings, aid in zip(embeddings_matrix, item_2_semantic)}
    return embeddings_dict
# -----------------------------------------------------
# Example: embeddings dictionary
# -----------------------------------------------------
# embeddings = {item_id: np.array([...]), ...}

# Example inputs (you replace these with your data)
# similar_pairs = [(id1, id2), (id3, id4), ...]
def compare_cosine(embeddings_lookup, similar_pairs, item_ids,
                   random_pairs=None, seed=42):
    """
    embeddings_lookup: dict OR matrix accessor
    similar_pairs: list of (id1, id2)
    item_ids: list of all IDs in the dataset
    random_pairs: OPTIONAL — list of pairs to use instead of generating new ones
    """

    # 1. Generate random pairs ONLY if not given
    if random_pairs is None:
        random_pairs = sample_random_pairs(
            item_ids=item_ids,
            similar_pairs=similar_pairs,
            k=len(similar_pairs),
            seed=seed
        )

    # 2. Convert similar pairs → vectors
    sim_true = [
        cosine_sim(embeddings_lookup[a], embeddings_lookup[b])
        for a, b in similar_pairs
    ]

    # 3. Convert random pairs → vectors
    sim_rand = [
        cosine_sim(embeddings_lookup[a], embeddings_lookup[b])
        for a, b in random_pairs
    ]
    print("Cosine-Similarity scores from top to bottom")
    pairs_and_scores = list(zip(sim_true, similar_pairs))

    # Sort by score (index 0 in each tuple), descending
    pairs_and_scores_sorted = sorted(pairs_and_scores, key=lambda x: x[0], reverse=True)

    # Print nicely
    print("Pairs                   Cosine-Similarity Score")
    for score, pair in pairs_and_scores_sorted:
        print(f"({pair[0]}, {pair[1]})  {score:.4f}")
    print(random_pairs)
    # 4. return both the scores AND the random pairs used
    return sim_true, sim_rand, random_pairs

def cosine_sim_boxplot(sim_scores_true, sim_scores_random):
    # -------------------------------------------------
    # Statistical significance (optional)
    # -------------------------------------------------
    stat, pvalue = mannwhitneyu(sim_scores_true, sim_scores_random, alternative='greater')
    print(f"Mann–Whitney U test p-value = {pvalue:.4g}")

    # -------------------------------------------------
    # Visualization
    # -------------------------------------------------
    data = {
        "Cosine Similarity": sim_scores_true + sim_scores_random,
        "Group": (["True Similar"] * len(sim_scores_true) +
                  ["Random"] * len(sim_scores_random))
    }

    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Group", y="Cosine Similarity", data=data)
    sns.stripplot(x="Group", y="Cosine Similarity", data=data, color='black', alpha=0.4)

    plt.title("Cosine Similarity Distribution: True vs Random Pairs 384D Embeddings")
    plt.show()
    return

if __name__ == '__main__':

    # Manually curated similar Article pairs
    similar_pairs = [
        ('108775015', '108775044'),
        ('110065001', '110065002'),
        ('111565001', '111586001'),
        ('112679048', '112679052'),
        ('118458034', '118458028'),
        ('120129025', '120129001'),
        ('146721001', '126589012'),
        ('146730001', '148033001'),
        ('156224001', '160442007'),
        ('162074069', '164912035'),
        ('179393001', '179208008'),
        ('183815013', '224606019'),
        ('189654047', '189626001'),
        ('189955076', '203027045'),
        ('212629004', '212629049'),
        ('212766042', '212766046'),
        ('215303001', '215324023'),
        ('217207045', '217207055'),
        ('224314007', '224314018'),
        ('233091003', '233091016'),
    ]
    embeddings_matrix = np.load('SBERT_embeddings_fulldata.npy')
    item_2_semantic = pd.read_pickle("item_2_semantic.pkl")
    item_ids = list(item_2_semantic.keys())
    item_2_embedding = get_embeddings_dict(item_ids, embeddings_matrix)

    sim_true_sem, sim_rand_sem, random_pairs_used = compare_cosine(
        embeddings_lookup=item_2_embedding,
        similar_pairs=similar_pairs,
        random_pairs=None,
        item_ids=item_ids,
        seed=42
    )
    print(f"Mean cosim:{np.array(sim_true_sem).mean()}")
    print(f"Mean cosim:{np.array(sim_rand_sem).mean()}")

    cosine_sim_boxplot(sim_true_sem, sim_rand_sem)





