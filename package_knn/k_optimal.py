"""
Module k_optimal.py

Fonction pour trouver la valeur optimale de K pour le modèle KNN
en testant différentes valeurs et en visualisant l'accuracy.
"""

from .knn import KNN
from .metrics import accuracy_score
import matplotlib.pyplot as plt

def find_k_optimal(X_train, X_test, y_train, y_test, metric="euclidean"):
    """
    Trouve la valeur optimale de K pour le KNN en se basant sur l'accuracy du test set.

    Args:
        X_train, X_test: numpy arrays, features d'entraînement et de test
        y_train, y_test: numpy arrays, labels d'entraînement et de test
        metric: str, 'euclidean' ou 'manhattan'

    Returns:
        optimal_k: int, valeur de K donnant la meilleure accuracy
    """
    k_values = range(1, 31)  # Plage de K à tester
    accuracies = []

    for k in k_values:
        knn = KNN(K=k, metric=metric)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

    optimal_k = k_values[accuracies.index(max(accuracies))]
    print(f"Optimal value of k (with {metric} metric): {optimal_k}")

    # Visualisation
    plt.plot(k_values, accuracies, marker='o')
    plt.xlabel("Value of k")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs. Number of Neighbors (k) with {metric} metric")
    plt.axvline(optimal_k, color='red', linestyle='--', label=f'Optimal k = {optimal_k}')
    plt.legend()
    plt.show()

    return optimal_k
