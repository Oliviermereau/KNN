"""
Module metrics.py

Fonctions pour évaluer la performance des modèles (accuracy, etc.).
"""

import numpy as np

def accuracy_score(y_true, y_pred):
    """
    Calcule l'accuracy entre les labels réels et les prédictions.

    Args:
        y_true: list ou array 1D, labels réels
        y_pred: list ou array 1D, labels prédits

    Returns:
        float: proportion de prédictions correctes
    """
    assert len(y_true) == len(y_pred), "y_true et y_pred doivent avoir la même longueur"
    return sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
