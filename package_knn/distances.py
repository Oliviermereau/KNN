
"""
Module distances.py

Fonctions utilitaires pour KNN :
- get_distance : calcule la distance entre deux points
- get_distances : calcule les distances d'un point vers un ensemble
- get_k_closest : récupère les indices des K plus proches voisins
"""

import numpy as np

def get_distance(p, q, metric):
    """
    Calcule la distance entre deux points selon la métrique spécifiée.
    """
    if metric == "euclidean":
        return (sum((pi - qi) ** 2 for pi, qi in zip(p, q))) ** 0.5
    elif metric == "manhattan":
        return sum(abs(pi - qi) for pi, qi in zip(p, q))
    else:
        raise ValueError("Unsupported metric. Use 'euclidean' or 'manhattan'.")

def get_distances(p, X, metric):
    """
    Retourne la liste des distances entre p et chaque point de X.
    """
    return [get_distance(p,q,metric) for q in X]

def get_k_closest(l, K=1):
    """
    Retourne les indices des K plus petites valeurs dans une liste.
    """
    assert K >= 1, "K parameter must be greater than 1"
    return np.argsort(l)[:K]