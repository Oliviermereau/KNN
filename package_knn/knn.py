from statistics import mode
from .distances import get_distances, get_k_closest
from collections.abc import Iterable
import numpy as np

class KNN():
    """
    Implémentation du K-Nearest Neighbors (k-NN)
    """

    def __init__(self, K=1, metric="euclidean"):
        """
        Initialisation du modèle k-NN.

        Arguments:
        ----
        K : int
            Le nombre de voisins à considérer pour la prédiction.
        metric : str
            La métrique à utiliser, soit "euclidean" ou "manhattan".
        """
        self.K = K
        self.metric = metric

    def fit(self, X, y):
        """ 
        Entraînement du modèle k-NN (enregistrer simplement les données).
        
        Arguments:
        ----
        X : array-like
            Matrice des caractéristiques d'entrée (n_samples, n_features).
        y : array-like
            Labels associés aux données d'entrée.
        """
        
        if not isinstance(X, Iterable):
            raise TypeError("X must be an iterable")
        if not isinstance(y, Iterable):
            raise TypeError("y must be an iterable")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        self.X = X
        self.y = y

    def _predict_single_point(self, x):
        """
        Prédit la classe d'une seule observation x en fonction des k voisins les plus proches.

        Arguments:
        ----
        x : array-like
            Point de données à prédire.

        Retourne:
        -------
        vote : int
            La classe prédite pour le point x.
        """
        # Calcul des distances entre le point x et tous les autres points de X
        dists = get_distances(x, self.X, metric=self.metric)

        # Trouver les indices des K plus proches voisins
        k_closest_idx = get_k_closest(dists, K=self.K)

        # Obtenir les labels des voisins les plus proches
        k_closest_votes = [self.y[idx] for idx in k_closest_idx]

        # Appliquer la majorité pour obtenir la classe prédite
        try:
            vote = mode(k_closest_votes)
        except:
            # En cas d'égalité, on choisit aléatoirement parmi les valeurs possibles
            vote = np.random.choice(k_closest_votes)
        
        return vote

    def predict(self, X):
        """
        Prédit les classes pour un ensemble de points X.

        Arguments:
        ----
        X : array-like
            Ensemble de données pour lesquelles effectuer des prédictions.

        Retourne:
        -------
        List of predictions : list
            Liste des classes prédites pour chaque point de X.
        """
        # Applique _predict_single_point à chaque point de l'ensemble X
        return [self._predict_single_point(x) for x in X]
