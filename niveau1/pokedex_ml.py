"""
Cœur "Machine Learning" du Pokédex — Niveau 1 (données tabulaires).

Ce module regroupe TOUT ce qui touche aux données et aux modèles :
    1. chargement du CSV ;
    2. clustering (K-Means) pour trouver des groupes de Pokémon et les interpréter ;
    3. comparaison rigoureuse de plusieurs modèles (validation croisée + jeu de test) pour :
         - prédire si un Pokémon est légendaire (classification binaire, classes déséquilibrées),
         - prédire son type principal "Type 1" (classification multi-classes, 18 classes) ;
    4. une classe `PokedexIA` qui entraîne les modèles retenus et fait les prédictions
       utilisées par l'application (app_pokedex_niveau1.py).

Le notebook (analyse_niveau1.ipynb) appelle ces fonctions pour produire les tableaux
et les figures du rapport ; l'application les appelle pour faire les prédictions.
On évite ainsi de dupliquer le code entre le notebook et l'application.

Bibliothèques : numpy, pandas, scikit-learn, matplotlib (voir requirements.txt).
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
    top_k_accuracy_score,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Le CSV est à la racine du dépôt (un dossier au-dessus de niveau1/).
CHEMIN_CSV = Path(__file__).resolve().parent.parent / "pokemon.csv"

# Les 6 statistiques de base = les "features" (variables d'entrée) de tous nos modèles.
# La consigne dit : "prend en entrée les statistiques de base d'un Pokémon".
STATS = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]

# Noms français, pour l'affichage uniquement.
STATS_FR = {
    "HP": "PV",
    "Attack": "Attaque",
    "Defense": "Défense",
    "Sp. Atk": "Att. Spé.",
    "Sp. Def": "Déf. Spé.",
    "Speed": "Vitesse",
}

# Graine aléatoire fixe => mêmes résultats à chaque exécution (reproductibilité).
GRAINE = 42

# 20 % des données sont mises de côté et ne servent QU'À l'évaluation finale.
PROPORTION_TEST = 0.2

# Nombre de groupes du K-Means. Choisi après analyse (voir le notebook) :
#  - le silhouette score est maximal pour k=2, mais ce découpage est trivial
#    ("faibles" contre "forts") ;
#  - la courbe du coude n'a pas de cassure nette ;
#  - k=5 a un silhouette un peu meilleur que k=3, 4 et 6 (petit pic local) et donne
#    des groupes faciles à interpréter (faibles / défensifs / endurants /
#    rapides / surpuissants).
# => C'est un CHOIX à justifier dans le rapport, pas une vérité absolue.
NB_CLUSTERS = 5


# ---------------------------------------------------------------------------
# 1. Données
# ---------------------------------------------------------------------------

def charger_donnees(chemin=CHEMIN_CSV):
    """Charge pokemon.csv dans un DataFrame pandas et ajoute la colonne 'Total'."""
    df = pd.read_csv(chemin)
    # "Legendary" peut être lu comme texte selon le fichier : on force un booléen.
    df["Legendary"] = df["Legendary"].astype(str).str.lower().eq("true")
    # Total des stats : pratique pour interpréter les groupes (PAS utilisé comme feature,
    # car c'est juste la somme des 6 stats => information redondante).
    df["Total"] = df[STATS].sum(axis=1)
    return df


def separer_train_test(df, cible):
    """
    Découpage 80 % entraînement / 20 % test, STRATIFIÉ sur la cible.

    Stratifié = on garde les mêmes proportions de classes dans les deux parties.
    Indispensable ici : il n'y a que ~8 % de légendaires ; sans stratification,
    le jeu de test pourrait n'en contenir presque aucun.
    """
    X = df[STATS]
    y = df[cible]
    return train_test_split(
        X, y, test_size=PROPORTION_TEST, stratify=y, random_state=GRAINE
    )


# ---------------------------------------------------------------------------
# 2. Clustering (non supervisé) : trouver des groupes dans les statistiques
# ---------------------------------------------------------------------------

def scores_choix_k(df, k_min=2, k_max=10):
    """
    Pour chaque k, entraîne un K-Means et renvoie :
      - l'inertie (somme des distances² au centre de son groupe) -> méthode du coude ;
      - le silhouette score (entre -1 et 1, plus c'est haut, mieux les groupes sont séparés).
    Les stats sont standardisées avant (moyenne 0, écart-type 1) pour que chaque
    statistique pèse autant dans la distance.
    """
    X = StandardScaler().fit_transform(df[STATS])
    lignes = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=GRAINE).fit(X)
        lignes.append({
            "k": k,
            "inertie": km.inertia_,
            "silhouette": silhouette_score(X, km.labels_),
        })
    return pd.DataFrame(lignes).set_index("k")


def nommer_cluster(profil_z, total_moyen, total_global):
    """
    Donne un nom lisible à un groupe à partir de son "profil" :
    profil_z = moyenne de chaque stat dans le groupe, en écart-types par rapport
    à la moyenne de tous les Pokémon (z-score).

    Règle simple (et assumée comme telle) :
      - niveau global d'après le Total moyen du groupe ;
      - stat la plus au-dessus de la moyenne => "spécialité" du groupe.
    """
    ratio = total_moyen / total_global
    if ratio < 0.8:
        niveau = "Faibles (souvent non évolués)"
        return niveau  # tout est bas : pas de spécialité qui ait du sens
    if ratio > 1.25:
        niveau = "Surpuissants (profil légendaire)"
    else:
        niveau = "Intermédiaires"

    stat_forte = profil_z.idxmax()
    if stat_forte in ("Defense", "Sp. Def"):
        specialite = "défensifs"
    elif stat_forte == "Speed":
        specialite = "rapides / offensifs"
    elif stat_forte in ("Attack", "Sp. Atk"):
        specialite = "offensifs"
    else:
        specialite = "endurants (PV élevés)"
    return f"{niveau} – {specialite}"


class Clustering:
    """K-Means sur les 6 stats standardisées, avec interprétation des groupes."""

    def __init__(self, df, k=NB_CLUSTERS):
        self.k = k
        # Pipeline = standardisation puis K-Means, appliqués toujours dans cet ordre.
        self.modele = make_pipeline(
            StandardScaler(), KMeans(n_clusters=k, n_init=10, random_state=GRAINE)
        )
        self.labels = self.modele.fit_predict(df[STATS])

        # Tableau descriptif de chaque groupe : moyennes des stats, taille, % de légendaires.
        tmp = df.assign(Groupe=self.labels)
        self.description = tmp.groupby("Groupe").agg(
            **{s: (s, "mean") for s in STATS},
            Total=("Total", "mean"),
            Effectif=("Name", "count"),
            Part_legendaires=("Legendary", "mean"),
        ).round(2)

        # Profil en z-scores (utile pour comparer les groupes entre eux).
        moy, ecart = df[STATS].mean(), df[STATS].std()
        self.profils_z = (self.description[STATS] - moy) / ecart

        total_global = df["Total"].mean()
        self.noms = {
            g: nommer_cluster(self.profils_z.loc[g], self.description.loc[g, "Total"], total_global)
            for g in self.description.index
        }
        self.description.insert(0, "Interprétation", pd.Series(self.noms))

        # Exemples représentatifs : les 3 Pokémon les plus proches du centre de chaque groupe.
        distances = self.modele.transform(df[STATS])  # distance à chaque centre
        self.exemples = {}
        for g in range(k):
            dans_g = np.where(self.labels == g)[0]
            proches = dans_g[np.argsort(distances[dans_g, g])[:3]]
            self.exemples[g] = df.iloc[proches]["Name"].tolist()

    def predire(self, stats):
        """Renvoie (numéro du groupe, nom du groupe) pour une liste de 6 stats."""
        x = pd.DataFrame([stats], columns=STATS)
        g = int(self.modele.predict(x)[0])
        return g, self.noms[g]

    def projection_2d(self, df):
        """Projection ACP (PCA) en 2 dimensions, pour visualiser les groupes sur un graphe."""
        X = self.modele[0].transform(df[STATS])  # stats standardisées
        acp = PCA(n_components=2, random_state=GRAINE)
        coords = acp.fit_transform(X)
        return coords, acp.explained_variance_ratio_


# ---------------------------------------------------------------------------
# 3. Modèles supervisés à comparer
# ---------------------------------------------------------------------------

def modeles_candidats(cible):
    """
    Renvoie un dictionnaire {nom: modèle} des modèles testés.

    - Chaque modèle sensible à l'échelle des données (régression logistique, k-NN, SVM)
      est précédé d'un StandardScaler dans un pipeline. Le scaler est alors appris
      UNIQUEMENT sur la partie entraînement de chaque pli => pas de fuite d'information.
    - Pour "Legendary", classes très déséquilibrées (65 légendaires / 800) :
      class_weight="balanced" donne plus de poids aux légendaires pendant l'apprentissage,
      sinon le modèle a intérêt à répondre "non légendaire" tout le temps.
    - Le "Dummy" (prédit toujours la classe majoritaire) sert de référence minimale :
      un modèle qui ne fait pas mieux que lui n'apprend rien.
    """
    poids = "balanced" if cible == "Legendary" else None
    return {
        "Référence (classe majoritaire)": DummyClassifier(strategy="prior"),
        "Régression logistique": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight=poids),
        ),
        "k-NN (k=7)": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=7)
        ),
        "Naïf bayésien gaussien": GaussianNB(),
        "Arbre de décision": DecisionTreeClassifier(
            max_depth=5, class_weight=poids, random_state=GRAINE
        ),
        "Forêt aléatoire": RandomForestClassifier(
            n_estimators=300, class_weight=poids, random_state=GRAINE
        ),
        # Un SVM ne donne pas de probabilités "naturellement" : CalibratedClassifierCV
        # les estime (calibration de Platt, par validation croisée interne).
        "SVM (noyau RBF)": make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(
                SVC(kernel="rbf", class_weight=poids, random_state=GRAINE),
                ensemble=False,
            ),
        ),
    }


def metriques(cible, classes=None):
    """
    Métriques adaptées à chaque problème (passées à cross_validate).

    Légendaire (binaire, déséquilibré) — l'accuracy seule est TROMPEUSE :
    répondre toujours "non" donne déjà ~92 % d'accuracy !
      - précision : parmi ceux prédits légendaires, combien le sont vraiment ;
      - rappel    : parmi les vrais légendaires, combien on en retrouve ;
      - F1        : moyenne harmonique précision/rappel ;
      - ROC-AUC   : qualité du classement des probabilités (0,5 = hasard, 1 = parfait) ;
      - PR-AUC (average precision) : comme ROC-AUC mais plus exigeant quand la classe
        positive est rare => c'est notre critère principal, car l'application affiche
        une PROBABILITÉ d'être légendaire.

    Type 1 (18 classes, déséquilibrées : 112 Water contre 4 Flying) :
      - accuracy ;
      - accuracy équilibrée : moyenne des rappels de chaque type (chaque type compte pareil) ;
      - F1 macro : moyenne des F1 de chaque type => critère principal ;
      - top-3 accuracy : le vrai type est-il dans les 3 types les plus probables ?
    """
    if cible == "Legendary":
        return {
            "accuracy": "accuracy",
            "accuracy_equilibree": "balanced_accuracy",
            "precision": make_scorer(precision_score, zero_division=0),
            "rappel": "recall",
            "f1": make_scorer(f1_score, zero_division=0),
            "roc_auc": "roc_auc",
            "pr_auc": "average_precision",
        }
    return {
        "accuracy": "accuracy",
        "accuracy_equilibree": "balanced_accuracy",
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
        "top3_accuracy": make_scorer(
            top_k_accuracy_score, response_method="predict_proba", k=3, labels=classes
        ),
    }


# Critère utilisé pour choisir automatiquement le "meilleur" modèle.
CRITERE = {"Legendary": "pr_auc", "Type 1": "f1_macro"}


def comparer_modeles(df, cible, nb_plis=5):
    """
    Compare tous les modèles candidats par validation croisée stratifiée à 5 plis
    sur la partie ENTRAÎNEMENT uniquement.

    Validation croisée : on coupe l'entraînement en 5 morceaux ; on entraîne sur 4,
    on évalue sur le 5e, et on recommence 5 fois en changeant de morceau.
    => 5 scores par modèle : on donne la moyenne ET l'écart-type (stabilité du modèle).
    Le jeu de test n'est pas touché ici : il sert à l'évaluation finale du modèle retenu.

    Renvoie un DataFrame (une ligne par modèle) trié par le critère principal.
    """
    X_train, _, y_train, _ = separer_train_test(df, cible)
    classes = sorted(df[cible].unique())
    plis = StratifiedKFold(n_splits=nb_plis, shuffle=True, random_state=GRAINE)

    lignes = []
    for nom, modele in modeles_candidats(cible).items():
        with warnings.catch_warnings():
            # Le type "Flying" n'a que 3 exemples dans l'entraînement (< 5 plis) :
            # scikit-learn prévient, mais ça fonctionne. On masque cet avertissement.
            warnings.simplefilter("ignore", UserWarning)
            res = cross_validate(modele, X_train, y_train, cv=plis, scoring=metriques(cible, classes))
        ligne = {"Modèle": nom}
        for m in metriques(cible, classes):
            scores = res[f"test_{m}"]
            ligne[m] = scores.mean()
            ligne[f"{m} (écart-type)"] = scores.std()
        lignes.append(ligne)

    tableau = pd.DataFrame(lignes).set_index("Modèle")
    return tableau.sort_values(CRITERE[cible], ascending=False)


def evaluer_sur_test(df, cible, nom_modele):
    """
    Entraîne `nom_modele` sur les 80 % d'entraînement et l'évalue UNE fois sur les 20 % de test.
    C'est l'estimation "honnête" des performances du modèle retenu.
    Renvoie (dictionnaire de scores, y_test, y_pred, modèle entraîné).
    """
    X_train, X_test, y_train, y_test = separer_train_test(df, cible)
    modele = modeles_candidats(cible)[nom_modele].fit(X_train, y_train)
    y_pred = modele.predict(X_test)
    proba = modele.predict_proba(X_test)

    if cible == "Legendary":
        p_leg = proba[:, list(modele.classes_).index(True)]
        scores = {
            "accuracy": accuracy_score(y_test, y_pred),
            "accuracy_equilibree": balanced_accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "rappel": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, p_leg),
            "pr_auc": average_precision_score(y_test, p_leg),
        }
    else:
        scores = {
            "accuracy": accuracy_score(y_test, y_pred),
            "accuracy_equilibree": balanced_accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "top3_accuracy": top_k_accuracy_score(
                y_test, proba, k=3, labels=modele.classes_
            ),
        }
    return scores, y_test, y_pred, modele


# ---------------------------------------------------------------------------
# 4. Le "cerveau" du Pokédex utilisé par l'application
# ---------------------------------------------------------------------------

# Modèles retenus PAR DÉFAUT, d'après la comparaison du notebook (validation croisée) :
#  - Légendaire : régression logistique. PR-AUC quasi à égalité avec la forêt aléatoire
#    (~0,77 toutes les deux), mais meilleur rappel (~0,96 : elle rate très peu de
#    légendaires), modèle bien plus simple et interprétable (un poids par stat).
#  - Type 1 : forêt aléatoire, meilleur F1 macro (même si ça reste faible, voir notebook).
# PokedexIA(choix_auto=True) refait le choix automatiquement selon CRITERE.
MODELE_RETENU = {
    "Legendary": "Régression logistique",
    "Type 1": "Forêt aléatoire",
}


class PokedexIA:
    """
    Entraîne le clustering + les deux classifieurs, puis prédit pour un nouveau Pokémon :
      - son groupe (cluster) et l'interprétation du groupe ;
      - la probabilité qu'il soit légendaire ;
      - ses types les plus probables.

    Les classifieurs finaux sont ré-entraînés sur TOUTES les données (train + test) :
    l'évaluation a déjà été faite dans le notebook, autant utiliser le maximum
    d'exemples pour l'application.
    """

    def __init__(self, df=None, choix_auto=False):
        self.df = charger_donnees() if df is None else df
        self.clustering = Clustering(self.df)

        self.noms_modeles = dict(MODELE_RETENU)
        if choix_auto:  # plus lent (~quelques dizaines de secondes)
            for cible in self.noms_modeles:
                self.noms_modeles[cible] = comparer_modeles(self.df, cible).index[0]

        X = self.df[STATS]
        self.modele_legendaire = modeles_candidats("Legendary")[self.noms_modeles["Legendary"]]
        self.modele_legendaire.fit(X, self.df["Legendary"])
        self.modele_type = modeles_candidats("Type 1")[self.noms_modeles["Type 1"]]
        self.modele_type.fit(X, self.df["Type 1"])

    def predire(self, stats):
        """
        stats : liste de 6 nombres dans l'ordre de STATS (PV, Attaque, Défense,
        Att. Spé., Déf. Spé., Vitesse).
        Renvoie un dictionnaire avec toutes les informations à afficher.
        """
        if len(stats) != len(STATS):
            raise ValueError(f"Il faut exactement {len(STATS)} statistiques.")
        x = pd.DataFrame([[float(v) for v in stats]], columns=STATS)

        groupe, nom_groupe = self.clustering.predire(x.iloc[0].tolist())

        classes_leg = list(self.modele_legendaire.classes_)
        p_legendaire = float(self.modele_legendaire.predict_proba(x)[0][classes_leg.index(True)])

        proba_types = self.modele_type.predict_proba(x)[0]
        ordre = np.argsort(proba_types)[::-1][:3]
        top_types = [(self.modele_type.classes_[i], float(proba_types[i])) for i in ordre]

        return {
            "groupe": groupe,
            "nom_groupe": nom_groupe,
            "exemples_groupe": self.clustering.exemples[groupe],
            "proba_legendaire": p_legendaire,
            "type_predit": top_types[0][0],
            "top3_types": top_types,
        }

    def stats_de(self, nom):
        """Renvoie les 6 stats d'un Pokémon du jeu de données (recherche sans casse)."""
        ligne = self.df[self.df["Name"].str.lower() == nom.strip().lower()]
        if ligne.empty:
            return None
        return ligne.iloc[0][STATS].astype(float).tolist()
