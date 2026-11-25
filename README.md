# Pokédex intelligent — projet Fondements mathématiques de l'IA

Le code suit `Consignes.pdf` et couvre les 3 niveaux, avec une appli graphique façon Pokédex qui regroupe tout.

| Dossier / fichier | Contenu |
|---|---|
| `pokedex_app.py` | **L'application à lancer** (interface Pokédex, 4 onglets) |
| `niveau1/` | Données tabulaires : clustering K-Means, probabilité légendaire, type d'après les stats + notebook |
| `niveau2_3/` | Images : types d'après une image (niv. 2), identification du Pokémon (niv. 3) |
| `pokemon.csv` | Jeu de données tabulaire (Kaggle) |

---

## 1. Installation (une seule fois)

**Version de Python** : TensorFlow (utilisé pour les images) ne suit pas toujours les toutes dernières
versions de Python. Le plus sûr est **Python 3.12** (ou 3.11 / 3.13). Vérifie avec `py --list`.

Dans un terminal (PowerShell), à la racine du dépôt :

```powershell
py -3.12 -m venv .venv          # crée un environnement Python isolé
.venv\Scripts\activate          # l'active (à refaire à chaque nouveau terminal)
pip install -r requirements.txt
```

Sans TensorFlow, l'appli marche quand même, mais les onglets Image et Caméra sont désactivés.

## 2. Entraîner les modèles d'images (une seule fois, niveaux 2 et 3)

Si `niveau2_3/modeles/` contient déjà `identification.joblib` et `types.joblib`, tu peux sauter cette étape.
Sinon :

```powershell
python niveau2_3/entrainer.py --images "C:\chemin\vers\le\dataset\pokemon"
```

Le dossier doit contenir **un sous-dossier par Pokémon** (`Abra\`, `Pikachu\`...), comme le dataset Kaggle
*Pokemon Images First Generation*. Compte environ 10 à 20 min la 1re fois : l'extraction des caractéristiques
de toutes les images est mise en cache, les fois suivantes sont plus rapides. Ajoute `--rapide` pour sauter
la forêt aléatoire, qui est la plus lente.

Le script crée :
- `niveau2_3/modeles/*.joblib` : les modèles utilisés par l'appli ;
- `niveau2_3/resultats/` : tableaux de comparaison (.csv) et figures, pour le rapport ;
- `niveau2_3/miniatures/` : une image par Pokémon, affichée dans le Pokédex.

## 3. Lancer l'application

```powershell
python pokedex_app.py
```

| Onglet | Niveau | Ce qu'il fait |
|---|---|---|
| Pokédex | 3 | Liste des 800 Pokémon : recherche, filtre par type, **regroupement par type**, tri par stat, fiche complète (stats, types, légendaire, groupe) |
| Stats → IA | 1 | 6 stats saisies → groupe (clustering), probabilité d'être légendaire, types probables |
| Image | 2 | Image choisie → type(s) prédit(s) + Pokémon reconnu |
| Caméra | 3 | Webcam → capture → Pokémon identifié → bouton vers sa fiche |

## 4. Notebook et analyses

- Niveau 1 : `jupyter notebook niveau1/analyse_niveau1.ipynb`
- Niveaux 2-3 : résultats dans `niveau2_3/resultats/` (générés par `entrainer.py`)

## Méthode (résumé)

- **Niveau 1** : 6 stats → K-Means (k = 5) ; légendaire = régression logistique ; type = forêt aléatoire.
  Les modèles sont comparés par validation croisée stratifiée.
- **Niveaux 2-3** : *transfer learning*. Un réseau MobileNetV2 pré-entraîné sur ImageNet transforme chaque image
  en un vecteur de 1280 nombres. Des classifieurs classiques (régression logistique, k-NN, forêt, MLP) apprennent
  ensuite sur ces vecteurs. La référence sans deep learning est un histogramme de couleurs.
  Le niveau 2 est évalué de deux façons : sur de nouvelles images de Pokémon déjà vus, et sur des Pokémon
  **jamais vus** (GroupKFold).
