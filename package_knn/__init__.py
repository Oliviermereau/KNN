"""
Package package_knn

Expose les éléments principaux pour un usage simple :
- KNN : classe principale
- accuracy_score : fonction d'évaluation
- find_k_optimal : recherche du meilleur K
"""

from .knn import KNN
from .metrics import accuracy_score
from .distances import get_distances, get_k_closest
from .k_optimal import find_k_optimal
