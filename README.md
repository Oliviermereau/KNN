# KNN-from-scratch

## Description
Package KNN développé from scratch avec notebooks d'exemple :  
- Prédiction de survie des passagers du Titanic  
- Classification de genres musicaux

---

## Organisation du repo

```
KNN-from-scratch/
├─ package_knn/ <- ton package k-NN
│ ├─ init.py
│ ├─ knn.py
│ ├─ distances.py
│ ├─ metrics.py
│ └─ k_optimal.py
├─ Titanic/
│ ├─ Titanic_KNN.ipynb
│ └─ data/
│ └─ titanic.csv
└─ Music/
├─ Music_KNN.ipynb
└─ data/
├─ music.csv
└─ music_labels.csv
```


- `package_knn/` : package réutilisable  
- `Titanic/` et `Music/` : notebooks et leurs données

---

## Utilisation

1. Cloner le repo :  
```bash
git clone <URL-du-repo>
```
Ou télécharger directement le ZIP du repo si vous préférez.

2. Ouvrir un notebook dans Jupyter ou VS Code et exécuter les cellules:

Les notebooks importent automatiquement le package `package_knn`.

3. Exemple d’utilisation du package :
```python
from package_knn import KNN, accuracy_score, find_k_optimal

knn = KNN(K=5, metric='euclidean')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

## Notes


Les notebooks fonctionnent directement si la structure des dossiers est respectée.

Pas besoin de configurer PYTHONPATH ni d’installer le package pour les notebooks inclus.

---
