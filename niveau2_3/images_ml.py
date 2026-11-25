"""
Cœur "images" du Pokédex — Niveaux 2 et 3.

Idée générale (TRANSFER LEARNING) :
    On n'entraîne PAS un réseau de neurones de zéro (trop long sur un PC sans GPU,
    et 17 000 images c'est peu pour ça). On prend un réseau déjà entraîné sur
    ImageNet (MobileNetV2, ~1,2 million d'images) et on l'utilise comme
    "extracteur de caractéristiques" : chaque image devient un vecteur de 1280
    nombres qui décrit son contenu (formes, couleurs, textures...).
    Ensuite, des modèles classiques (régression logistique, k-NN, forêt, MLP...)
    apprennent à partir de ces vecteurs :
        - Niveau 2 : prédire le ou les TYPES du Pokémon (multi-label : 1 ou 2 types) ;
        - Niveau 3 : IDENTIFIER le Pokémon (classification à ~150 classes).

    Référence "sans deep learning" pour comparer : un histogramme de couleurs.

Ce module contient : chargement des images, extraction des caractéristiques,
modèles candidats, métriques, et la classe `ReconnaissanceImage` utilisée par l'appli.
"""

from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

DOSSIER = Path(__file__).resolve().parent
CHEMIN_CSV = DOSSIER.parent / "pokemon.csv"
DOSSIER_MODELES = DOSSIER / "modeles"
DOSSIER_MINIATURES = DOSSIER / "miniatures"
DOSSIER_RESULTATS = DOSSIER / "resultats"

TAILLE = 224          # taille d'entrée attendue par MobileNetV2 (224 x 224 pixels)
GRAINE = 42
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


# ---------------------------------------------------------------------------
# 1. Correspondance dossiers d'images <-> lignes du CSV
# ---------------------------------------------------------------------------

def normaliser_nom(nom):
    """
    Met un nom sous une forme comparable : 'Mr. Mime' -> 'mrmime', 'Nidoran♀' -> 'nidoranf'.
    Utile car les noms des dossiers du dataset et ceux du CSV ne sont pas écrits pareil.
    """
    nom = str(nom).replace("♀", "f").replace("♂", "m")
    nom = unicodedata.normalize("NFKD", nom).encode("ascii", "ignore").decode()
    return "".join(c for c in nom.lower() if c.isalnum())


def charger_csv():
    """CSV des Pokémon, sans les Méga-évolutions (elles n'ont pas d'images à part)."""
    df = pd.read_csv(CHEMIN_CSV)
    df["Legendary"] = df["Legendary"].astype(str).str.lower().eq("true")
    df = df[~df["Name"].str.startswith("Mega ")].copy()
    df["cle"] = df["Name"].map(normaliser_nom)
    return df.drop_duplicates("cle").set_index("cle")


def types_de(ligne):
    """Liste des types d'une ligne du CSV : ['Fire'] ou ['Fire', 'Flying']."""
    return [ligne["Type 1"]] + ([ligne["Type 2"]] if isinstance(ligne["Type 2"], str) else [])


def lister_images(dossier_images):
    """
    Parcourt le dataset : un sous-dossier par Pokémon, contenant ses images.
    (C'est l'organisation du dataset Kaggle "Pokemon Images First Generation".)
    Renvoie un DataFrame : chemin, nom du Pokémon (comme dans le CSV), types.
    Affiche les dossiers qu'on n'a pas su relier au CSV.
    """
    csv = charger_csv()
    lignes, inconnus = [], []
    for sous_dossier in sorted(Path(dossier_images).iterdir()):
        if not sous_dossier.is_dir():
            continue
        cle = normaliser_nom(sous_dossier.name)
        if cle not in csv.index:
            inconnus.append(sous_dossier.name)
            continue
        info = csv.loc[cle]
        for f in sorted(sous_dossier.rglob("*")):
            if f.suffix.lower() in EXTENSIONS:
                lignes.append({"chemin": str(f), "pokemon": info["Name"], "types": types_de(info)})
    if inconnus:
        print(f"[!] {len(inconnus)} dossier(s) sans correspondance dans le CSV, ignorés :", inconnus)
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------------------
# 2. Images -> vecteurs de caractéristiques
# ---------------------------------------------------------------------------

def preparer_image(source):
    """
    Ouvre une image (chemin, objet PIL ou tableau numpy RGB) et la met au format du réseau :
    RGB, fond blanc si l'image est transparente, carré 224x224 sans déformation
    (on complète avec du blanc plutôt que d'étirer l'image).
    Renvoie un tableau numpy uint8 (224, 224, 3).
    """
    if isinstance(source, np.ndarray):
        img = Image.fromarray(source)
    elif isinstance(source, Image.Image):
        img = source
    else:
        img = Image.open(source)
        img.seek(0)  # GIF animé : on prend la 1re image
    img = img.convert("RGBA")
    fond = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(fond, img).convert("RGB")
    img = ImageOps.pad(img, (TAILLE, TAILLE), color=(255, 255, 255))
    return np.asarray(img, dtype=np.uint8)


def histogramme_couleurs(image):
    """
    Caractéristique "naïve" de référence : histogramme 3D des couleurs en HSV
    (16 teintes x 4 saturations x 4 luminosités = 256 valeurs, normalisé).
    Décrit QUELLES couleurs il y a, mais pas les formes.
    """
    hsv = np.asarray(Image.fromarray(image).convert("HSV")).reshape(-1, 3)
    hist, _ = np.histogramdd(hsv, bins=(16, 4, 4), range=((0, 256), (0, 256), (0, 256)))
    hist = hist.ravel()
    return (hist / hist.sum()).astype(np.float32)


class ExtracteurCNN:
    """
    MobileNetV2 pré-entraîné sur ImageNet, sans sa dernière couche ("include_top=False").
    pooling="avg" : on fait la moyenne de la dernière carte de caractéristiques
    -> un vecteur de 1280 nombres par image.
    Les poids (~9 Mo) sont téléchargés automatiquement au premier lancement.
    """

    def __init__(self):
        import os
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # moins de messages TensorFlow
        import keras
        from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

        self._preprocess = preprocess_input
        self.reseau = MobileNetV2(
            weights="imagenet", include_top=False, pooling="avg",
            input_shape=(TAILLE, TAILLE, 3),
        )
        self.reseau.trainable = False

    def __call__(self, images):
        """images : tableau (n, 224, 224, 3) uint8 -> tableau (n, 1280) float32."""
        x = self._preprocess(np.asarray(images, dtype=np.float32))
        return self.reseau.predict(x, verbose=0, batch_size=64).astype(np.float32)


def extraire_caracteristiques(df, chemin_cache, taille_lot=128):
    """
    Calcule (ou relit depuis le cache) les caractéristiques de toutes les images.
    Le calcul prend plusieurs minutes : on le sauvegarde dans un fichier .npz
    pour ne pas le refaire à chaque fois.
    Renvoie (X_cnn, X_hist, df_filtré) — les images illisibles sont retirées.
    """
    chemin_cache = Path(chemin_cache)
    if chemin_cache.exists():
        d = np.load(chemin_cache, allow_pickle=True)
        if list(d["chemins"]) == df["chemin"].tolist():
            ok = d["ok"]
            return d["X_cnn"], d["X_hist"], df[ok].reset_index(drop=True)
        print("Cache obsolète (les images ont changé) : recalcul.")

    extracteur = ExtracteurCNN()
    n = len(df)
    X_cnn = np.zeros((n, 1280), np.float32)
    X_hist = np.zeros((n, 256), np.float32)
    ok = np.zeros(n, bool)
    for debut in range(0, n, taille_lot):
        idx, images = [], []
        for i in range(debut, min(debut + taille_lot, n)):
            try:
                images.append(preparer_image(df["chemin"].iat[i]))
                idx.append(i)
            except Exception as e:  # fichier corrompu, format exotique...
                print(f"  image ignorée ({e.__class__.__name__}) : {df['chemin'].iat[i]}")
        if images:
            X_cnn[idx] = extracteur(np.stack(images))
            X_hist[idx] = [histogramme_couleurs(im) for im in images]
            ok[idx] = True
        print(f"  {min(debut + taille_lot, n)}/{n} images traitées", end="\r")
    print()
    np.savez_compressed(chemin_cache, X_cnn=X_cnn[ok], X_hist=X_hist[ok], ok=ok,
                        chemins=np.array(df["chemin"].tolist(), dtype=object))
    return X_cnn[ok], X_hist[ok], df[ok].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Modèles candidats (appliqués sur les vecteurs de caractéristiques)
# ---------------------------------------------------------------------------

def modeles_candidats():
    """
    Modèles comparés. Tous acceptent aussi le multi-label (niveau 2) : les k-NN,
    forêts et MLP nativement ; la régression logistique via OneVsRest (voir plus bas).
    - Régression logistique : modèle linéaire, simple, donne de bonnes probabilités.
    - k-NN (distance cosinus) : "à quelles images d'entraînement ressemble-t-elle le plus ?".
    - Forêt aléatoire : ensemble d'arbres de décision.
    - MLP : petit réseau de neurones (1 couche cachée de 512 neurones) entraîné
      par-dessus le CNN gelé.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "Régression logistique": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, C=0.5)
        ),
        "k-NN (k=5, cosinus)": KNeighborsClassifier(n_neighbors=5, metric="cosine", weights="distance"),
        "Forêt aléatoire": RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=GRAINE
        ),
        "MLP (1 couche cachée)": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(512,), alpha=1e-3, early_stopping=True,
                          max_iter=200, random_state=GRAINE),
        ),
    }


def version_multilabel(nom, modele):
    """La régression logistique ne gère pas le multi-label seule : un classifieur par type."""
    from sklearn.multiclass import OneVsRestClassifier
    if nom.startswith("Régression logistique"):
        return OneVsRestClassifier(modele, n_jobs=-1)
    return modele


# ---------------------------------------------------------------------------
# 4. Métriques
# ---------------------------------------------------------------------------

def proba_vers_types(proba, classes, seuil=0.5, max_types=2):
    """
    Transforme les probabilités de chaque type en une prédiction de 1 ou 2 types :
    on garde les types dont la probabilité dépasse `seuil` (au plus 2, comme dans
    les jeux), et au minimum le type le plus probable.
    Renvoie une matrice binaire (n, nb_types).
    """
    proba = np.asarray(proba)
    pred = np.zeros_like(proba, dtype=int)
    ordre = np.argsort(-proba, axis=1)
    for i in range(len(proba)):
        pred[i, ordre[i, 0]] = 1
        for j in ordre[i, 1:max_types]:
            if proba[i, j] >= seuil:
                pred[i, j] = 1
    return pred


def proba_multilabel(modele, X):
    """predict_proba renvoie une liste de tableaux pour les forêts / k-NN multi-label : on uniformise."""
    p = modele.predict_proba(X)
    if isinstance(p, list):
        return np.column_stack([pi[:, 1] if pi.shape[1] == 2 else np.zeros(len(X)) for pi in p])
    return p


def scores_types(Y_vrai, Y_pred, pokemons):
    """
    Métriques du niveau 2 (multi-label).
      - exact      : les types prédits sont EXACTEMENT les bons (les 2 s'il y en a 2) ;
      - au moins 1 : au moins un type prédit est correct ;
      - F1 micro   : sur l'ensemble de toutes les décisions (type présent / absent) ;
      - F1 macro   : moyenne des F1 de chaque type => chaque type compte autant,
                     même les rares (Ice, Fairy...) ;
      - couverture : % des Pokémon pour lesquels au moins 50 % de leurs images
                     reçoivent exactement les bons types (exigence de la consigne :
                     ne pas être bon seulement sur quelques Pokémon faciles).
    """
    from sklearn.metrics import f1_score
    exact = (Y_vrai == Y_pred).all(axis=1)
    au_moins_un = ((Y_vrai == 1) & (Y_pred == 1)).any(axis=1)
    par_pokemon = pd.Series(exact).groupby(np.asarray(pokemons)).mean()
    return {
        "exact": exact.mean(),
        "au_moins_1_type_correct": au_moins_un.mean(),
        "f1_micro": f1_score(Y_vrai, Y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(Y_vrai, Y_pred, average="macro", zero_division=0),
        "couverture_50%": (par_pokemon >= 0.5).mean(),
    }, par_pokemon


def scores_identification(y_vrai, proba, classes):
    """
    Métriques du niveau 3 (identification, ~150 classes).
      - accuracy, accuracy équilibrée (= moyenne des rappels de chaque Pokémon),
        F1 macro, top-5 (le bon Pokémon est dans les 5 propositions) ;
      - couverture 50 % / 80 % : % des Pokémon reconnus sur au moins 50 % / 80 % de leurs images.
    """
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                                 top_k_accuracy_score)
    y_pred = np.asarray(classes)[np.argmax(proba, axis=1)]
    rappel = pd.Series(y_pred == np.asarray(y_vrai)).groupby(np.asarray(y_vrai)).mean()
    return {
        "accuracy": accuracy_score(y_vrai, y_pred),
        "accuracy_equilibree": balanced_accuracy_score(y_vrai, y_pred),
        "f1_macro": f1_score(y_vrai, y_pred, average="macro", zero_division=0),
        "top5": top_k_accuracy_score(y_vrai, proba, k=5, labels=classes),
        "couverture_50%": (rappel >= 0.5).mean(),
        "couverture_80%": (rappel >= 0.8).mean(),
    }, rappel


# ---------------------------------------------------------------------------
# 5. Utilisation dans l'application
# ---------------------------------------------------------------------------

class ReconnaissanceImage:
    """
    Charge les modèles entraînés (fichiers .joblib créés par entrainer.py) et
    analyse une image : identification du Pokémon (niveau 3) + types (niveau 2).
    """

    def __init__(self):
        import joblib
        f_id = DOSSIER_MODELES / "identification.joblib"
        f_ty = DOSSIER_MODELES / "types.joblib"
        if not (f_id.exists() and f_ty.exists()):
            raise FileNotFoundError(
                "Modèles absents : lance d'abord  python niveau2_3/entrainer.py --images <dossier>"
            )
        self.identification = joblib.load(f_id)   # dict : modele, nom, classes
        self.types = joblib.load(f_ty)            # dict : modele, nom, classes (types)
        self.extracteur = ExtracteurCNN()

    def analyser(self, source):
        """
        source : chemin d'image, image PIL ou tableau numpy RGB (image caméra).
        Renvoie un dict : top-5 Pokémon (nom, proba) et types prédits (type, proba).
        """
        x = self.extracteur(preparer_image(source)[None])

        m = self.identification
        p_id = m["modele"].predict_proba(x)[0]
        ordre = np.argsort(-p_id)[:5]
        top5 = [(m["classes"][i], float(p_id[i])) for i in ordre]

        t = self.types
        p_ty = proba_multilabel(t["modele"], x)
        pred = proba_vers_types(p_ty, t["classes"])[0]
        types_predits = [(t["classes"][j], float(p_ty[0, j])) for j in np.argsort(-p_ty[0]) if pred[j]]
        tous_types = sorted(zip(t["classes"], p_ty[0].tolist()), key=lambda c: -c[1])

        return {"top5": top5, "types": types_predits, "probas_types": tous_types}
