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