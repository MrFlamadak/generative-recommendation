import numpy as np

def MAP_at_K(predicted, labels, K=12):
    '''
    Evaluates the results using the Mean Average Precision @ K (MAP@K) metric.
    K is by default set to 12. This is a special version of MAP@K where
    the number of labels for each customer always is 1.
    
    Parameters
    ----------
    predicted : np.array
                A NumPy array of NumPy arrays of predicted items
    labels : np.array
             A NumPy array of correct items
    K : int, optional
        Decides the K in Top-K
        
    Returns
    -------
    score : double
            Mean Average Precision @ K
    '''
    scores = []
    for (pred, label) in zip(predicted, labels):
        if len(pred) > K:
            pred = list(pred[:K])
        else:
            pred = list(pred)
            
        if label in pred:
            rank = pred.index(label) + 1
            scores.append(1.0 / rank)
        else:
            scores.append(0.0)
    
    score = np.mean(scores)
    
    return score
    
def precision_at_K(gen, test, K):
    if len(gen) < K:
        raise ValueError(f"Generated item list too short: Expected at least {K}, got {len(gen)}.")
    
    pred = gen[:K]
    t_p = len(set(pred) & set(test))
    precision = t_p / K    
    return precision

def recall_at_K(gen, test, K):
    if len(gen) < K:
        raise ValueError(f"Generated item list too short: Expected at least {K}, got {len(gen)}.")
    
    pred = gen[:K]
    t_p = len(set(pred) & set(test))
    recall = t_p / len(test)
    return recall
