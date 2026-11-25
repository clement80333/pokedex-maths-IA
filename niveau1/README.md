# Pokédex IA — Niveau 1 (données tabulaires)

Code écrit d'après `Consignes.pdf` (niveau 1 = module Fondements mathématiques de l'IA).

| Fichier | Rôle |
|---|---|
| `pokedex_ml.py` | Données, clustering K-Means, comparaison des modèles, classe `PokedexIA` |
| `analyse_niveau1.ipynb` | Notebook à rendre : exploration, clustering, comparaison des modèles, choix justifiés |
| `construire_notebook.py` | Régénère le notebook (si on modifie son contenu) |
| `app_pokedex_niveau1.py` | Application (tkinter) : groupe + probabilité légendaire + top-3 types |

## Lancer

```bash
pip install -r requirements.txt
python app_pokedex_niveau1.py          # l'application
jupyter notebook analyse_niveau1.ipynb # l'analyse
```

Le CSV utilisé est `../pokemon.csv` (racine du dépôt).

## Résultats principaux (validation croisée 5 plis, sur 80 % des données)

- **Clustering** : k = 5 groupes (faibles / défensifs / endurants / rapides / surpuissants).
- **Légendaire** : régression logistique retenue (PR-AUC ≈ 0,77, rappel ≈ 0,96).
- **Type 1** : forêt aléatoire retenue, mais performances faibles (accuracy ≈ 22 %, top-3 ≈ 47 %) :
  les stats seules ne suffisent pas à déterminer le type.
