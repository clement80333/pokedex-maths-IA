"""
Génère le notebook analyse_niveau1.ipynb (cellules Markdown + code).

Pourquoi un script ? Pour garder le contenu du notebook lisible et versionnable.
Usage :
    python construire_notebook.py
    jupyter nbconvert --to notebook --execute --inplace analyse_niveau1.ipynb
(la 2e commande exécute le notebook pour y enregistrer les résultats et figures)
"""

import nbformat as nbf

cellules = []


def md(texte):
    cellules.append(nbf.v4.new_markdown_cell(texte.strip()))


def code(texte):
    cellules.append(nbf.v4.new_code_cell(texte.strip()))


md("""
# Pokédex intelligent — Niveau 1 : données tabulaires

Objectifs (consignes) :
1. **Clustering** : identifier des groupes dans les statistiques et les interpréter.
2. **Classification** à partir des 6 statistiques de base :
   - le Pokémon est-il **légendaire** ? (on veut une **probabilité**) ;
   - quel est son **type** (on prédit le `Type 1`).
3. **Comparer rigoureusement plusieurs modèles** avec des métriques adaptées et justifier le choix.

Tout le code « métier » est dans `pokedex_ml.py` ; ce notebook l'appelle et commente les résultats.
""")

code("""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import pokedex_ml as pml

warnings.filterwarnings("ignore", category=UserWarning)  # avertissement "classe Flying < 5 exemples"
pd.set_option("display.precision", 3)
df = pml.charger_donnees()
df.head()
""")

md("## 1. Exploration rapide")

code("""
print(df.shape[0], "Pokémon")
print("Légendaires :", df["Legendary"].sum(), f"({df['Legendary'].mean():.1%})")
df[pml.STATS + ["Total"]].describe()
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
df["Type 1"].value_counts().plot.bar(ax=axes[0], title="Nombre de Pokémon par Type 1")
df.boxplot(column="Total", by="Legendary", ax=axes[1])
axes[1].set_title("Total des stats : légendaires vs non légendaires"); fig.suptitle("")
plt.tight_layout(); plt.show()
""")

md("""
**À retenir** : les deux problèmes sont **déséquilibrés** (8 % de légendaires ; 112 « Water » contre
4 « Flying »). L'accuracy seule sera donc trompeuse : on utilisera des métriques adaptées.
Les légendaires ont un Total nettement plus élevé, on s'attend donc à bien les détecter.
""")

md("""
## 2. Clustering (K-Means)

Les stats sont **standardisées** (moyenne 0, écart-type 1) avant le K-Means, sinon la distance
serait dominée par les stats à grandes valeurs. Choix du nombre de groupes `k` :
méthode du **coude** (inertie) et **silhouette score**.
""")

code("""
choix_k = pml.scores_choix_k(df)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
choix_k["inertie"].plot(marker="o", ax=axes[0], title="Méthode du coude (inertie)")
choix_k["silhouette"].plot(marker="o", ax=axes[1], title="Silhouette score")
plt.tight_layout(); plt.show()
choix_k
""")

md("""
**Choix de k = 5** (constante `NB_CLUSTERS` dans pokedex_ml.py) : le silhouette est maximal pour k = 2, mais ce découpage est trivial
(« faibles » / « forts »). Le coude n'est pas net. k = 5 forme un petit pic local du silhouette
(meilleur que k = 3, 4 et 6) et donne des groupes **interprétables**. Remarque honnête : un silhouette autour de 0,2 signifie que les groupes se
chevauchent ; les stats des Pokémon forment plutôt un continuum que des groupes bien séparés.
""")

code("""
clustering = pml.Clustering(df)
clustering.description
""")

code("""
# Profil de chaque groupe en écart-types par rapport à la moyenne générale
ax = clustering.profils_z.rename(columns=pml.STATS_FR).T.plot.bar(figsize=(10, 4))
ax.axhline(0, color="black", lw=0.8); ax.set_ylabel("z-score moyen")
ax.legend([f"{g} : {n}" for g, n in clustering.noms.items()], fontsize=8)
plt.title("Profil des groupes"); plt.tight_layout(); plt.show()
for g, ex in clustering.exemples.items():
    print(f"Groupe {g} ({clustering.noms[g]}) — exemples : {', '.join(ex)}")
""")

code("""
# Visualisation 2D par ACP (analyse en composantes principales)
coords, variance = clustering.projection_2d(df)
plt.figure(figsize=(8, 6))
for g in range(clustering.k):
    m = clustering.labels == g
    plt.scatter(coords[m, 0], coords[m, 1], s=12, label=f"{g} : {clustering.noms[g]}")
leg = df["Legendary"].values
plt.scatter(coords[leg, 0], coords[leg, 1], s=40, facecolors="none", edgecolors="k", label="légendaire")
plt.xlabel(f"CP1 ({variance[0]:.0%} de variance)"); plt.ylabel(f"CP2 ({variance[1]:.0%})")
plt.legend(fontsize=8); plt.title("Groupes K-Means projetés sur les 2 premières composantes")
plt.tight_layout(); plt.show()
""")

md("""
**Interprétation** (à vérifier sur la sortie ci-dessus) : un groupe de Pokémon **faibles**
(souvent des formes non évoluées), trois groupes **intermédiaires** qui se distinguent par leur style
(défensifs, endurants avec beaucoup de PV, rapides) et un groupe **surpuissant** qui contient presque tous les
légendaires (et les Méga-évolutions). La 1re composante principale correspond en gros au Total.
""")

md("""
## 3. Prédire si un Pokémon est légendaire

Protocole (le même pour les deux problèmes) :
- découpage **80 % entraînement / 20 % test stratifié** (le test n'est utilisé qu'une fois, à la fin) ;
- **validation croisée stratifiée à 5 plis** sur l'entraînement pour comparer les modèles ;
- `class_weight="balanced"` pour compenser le faible nombre de légendaires ;
- un modèle **de référence** (toujours « non légendaire ») pour savoir ce que vaut « ne rien apprendre ».

Critère principal : **PR-AUC** (aire sous la courbe précision-rappel), adaptée quand la classe positive
est rare et qu'on exploite une **probabilité** (ce que l'application affiche).
""")

code("""
comparaison_leg = pml.comparer_modeles(df, "Legendary")
comparaison_leg[[c for c in comparaison_leg.columns if "écart" not in c]]
""")

code("""
# Stabilité : écart-type des scores sur les 5 plis
comparaison_leg[[c for c in comparaison_leg.columns if "écart" in c]]
""")

code("""
nom_leg = pml.MODELE_RETENU["Legendary"]
scores_leg, y_test, y_pred, modele_leg = pml.evaluer_sur_test(df, "Legendary", nom_leg)
print("Modèle retenu :", nom_leg)
print(pd.Series(scores_leg).round(3))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Non légendaire", "Légendaire"])
plt.title(f"Matrice de confusion (test) — {nom_leg}"); plt.show()
""")

code("""
# Interprétation de la régression logistique : poids de chaque stat (stats standardisées)
if nom_leg == "Régression logistique":
    coefs = pd.Series(modele_leg[-1].coef_[0], index=[pml.STATS_FR[s] for s in pml.STATS])
    coefs.sort_values().plot.barh(title="Poids de chaque stat (régression logistique)")
    plt.show()
""")

md("""
**Choix** : régression logistique et forêt aléatoire sont à égalité en PR-AUC (≈ 0,77 en validation
croisée). On retient la **régression logistique** : meilleur rappel (elle rate très peu de légendaires),
plus simple, interprétable (un poids par stat), et ses écarts-types montrent qu'elle est stable.

**Limites** :
- `class_weight="balanced"` gonfle les probabilités de la classe rare : la « probabilité d'être
  légendaire » est surtout un **score de confiance**, à ne pas lire comme une fréquence exacte ;
- les **Méga-évolutions** non légendaires ont des stats de légendaire => principaux faux positifs ;
- 65 légendaires seulement : les scores sur le test (13 légendaires) varient beaucoup d'un tirage à l'autre.
""")

md("""
## 4. Prédire le type principal (Type 1)

18 classes déséquilibrées. Critère principal : **F1 macro** (chaque type compte autant, même les rares).
On regarde aussi le **top-3 accuracy** : le vrai type est-il parmi les 3 propositions du Pokédex ?
""")

code("""
comparaison_type = pml.comparer_modeles(df, "Type 1")
comparaison_type[[c for c in comparaison_type.columns if "écart" not in c]]
""")

code("""
nom_type = pml.MODELE_RETENU["Type 1"]
scores_type, y_test_t, y_pred_t, modele_type = pml.evaluer_sur_test(df, "Type 1", nom_type)
print("Modèle retenu :", nom_type)
print(pd.Series(scores_type).round(3))
fig, ax = plt.subplots(figsize=(10, 9))
ConfusionMatrixDisplay.from_predictions(y_test_t, y_pred_t, ax=ax, xticks_rotation=90, colorbar=False)
plt.title(f"Matrice de confusion (test) — {nom_type}"); plt.tight_layout(); plt.show()
""")

code("""
# Importance des stats pour la forêt aléatoire
if nom_type == "Forêt aléatoire":
    imp = pd.Series(modele_type.feature_importances_, index=[pml.STATS_FR[s] for s in pml.STATS])
    imp.sort_values().plot.barh(title="Importance des stats pour prédire le type")
    plt.show()
""")

md("""
**Choix** : la **forêt aléatoire** a le meilleur F1 macro. Mais soyons francs : les performances sont
**faibles** (≈ 20-25 % d'accuracy, ~50 % en top-3). C'est attendu : les **stats seules ne déterminent
pas le type** (un Pokémon Eau et un Pokémon Normal peuvent avoir des stats identiques). Seuls quelques
types ont un profil marqué (Acier/Roche = Défense haute, Électrik = Vitesse haute, Psy = Att. Spé. haute).
C'est une vraie **limite du niveau 1**, que le niveau 2 (images) cherche à dépasser.

Pistes d'amélioration : ajouter des features (Génération, Type 2, taille/poids si disponibles),
optimiser les hyperparamètres (GridSearchCV), prédire les deux types (multi-label).
""")

md("""
## 5. Utilisation dans l'application

`PokedexIA` ré-entraîne les modèles retenus sur **toutes** les données, puis affiche pour un
Pokémon : son groupe, sa probabilité d'être légendaire et ses 3 types les plus probables.
⚠️ Pour un Pokémon **déjà présent** dans le CSV, la prédiction est trop optimiste (le modèle l'a vu
à l'entraînement) ; l'intérêt est surtout pour des stats **nouvelles**.
""")

code("""
ia = pml.PokedexIA(df)
for nom in ["Pikachu", "Mewtwo", "Snorlax"]:
    print(nom, "->", ia.predire(ia.stats_de(nom)))
print("Stats inventées ->", ia.predire([95, 120, 90, 110, 90, 105]))
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cellules
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "analyse_niveau1.ipynb")
print("analyse_niveau1.ipynb généré.")
