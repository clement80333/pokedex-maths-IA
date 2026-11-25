"""
Entraîne et compare les modèles des niveaux 2 et 3, puis sauvegarde les modèles retenus.

Usage (depuis la racine du dépôt) :
    python niveau2_3/entrainer.py --images "C:/chemin/vers/dataset/pokemon"

où le dossier contient un sous-dossier par Pokémon (Abra/, Aerodactyl/, ...).

Étapes :
  1. extraction des caractéristiques de toutes les images (MobileNetV2 + histogramme couleur),
     mise en cache dans niveau2_3/cache_caracteristiques.npz (quelques minutes la 1re fois) ;
  2. NIVEAU 3 — identification : comparaison de modèles (jeu de test = 20 % des images) ;
  3. NIVEAU 2 — types : comparaison de modèles
       a) sur des images de Pokémon déjà vus (découpage par image),
       b) sur des Pokémon JAMAIS vus (découpage par Pokémon, GroupKFold) ;
  4. ré-entraînement des modèles retenus sur toutes les images + sauvegarde (.joblib),
     tableaux de résultats (.csv) et figures (.png) dans niveau2_3/resultats/,
     une miniature par Pokémon dans niveau2_3/miniatures/ (pour l'appli).
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # pas de fenêtre : on enregistre les figures en .png
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# Permet "from niveau2_3 import ..." même quand on lance ce fichier directement
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from niveau2_3 import images_ml as im  # noqa: E402

warnings.filterwarnings("ignore")


def chrono(t0):
    return f"{time.time() - t0:.0f} s"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, help="dossier du dataset (un sous-dossier par Pokémon)")
    parser.add_argument("--rapide", action="store_true", help="saute la forêt aléatoire (la plus lente)")
    args = parser.parse_args()

    for d in (im.DOSSIER_MODELES, im.DOSSIER_MINIATURES, im.DOSSIER_RESULTATS):
        d.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ données
    df = im.lister_images(args.images)
    print(f"{len(df)} images, {df['pokemon'].nunique()} Pokémon")
    t0 = time.time()
    X_cnn, X_hist, df = im.extraire_caracteristiques(df, im.DOSSIER / "cache_caracteristiques.npz")
    print(f"Caractéristiques prêtes ({chrono(t0)}) : CNN {X_cnn.shape}, histogrammes {X_hist.shape}")

    y_id = df["pokemon"].to_numpy()
    mlb = MultiLabelBinarizer()
    Y_types = mlb.fit_transform(df["types"])
    types = list(mlb.classes_)

    # Effectifs par Pokémon (pour le rapport : le dataset est-il équilibré ?)
    effectifs = df["pokemon"].value_counts()
    effectifs.to_csv(im.DOSSIER_RESULTATS / "effectifs_par_pokemon.csv")
    print(f"Images par Pokémon : min {effectifs.min()}, médiane {effectifs.median():.0f}, max {effectifs.max()}")

    # Découpage 80/20 stratifié par Pokémon (chaque Pokémon garde ~20 % de ses images en test)
    idx = np.arange(len(df))
    i_train, i_test = train_test_split(idx, test_size=0.2, stratify=y_id, random_state=im.GRAINE)

    candidats = im.modeles_candidats()
    if args.rapide:
        candidats.pop("Forêt aléatoire")

    # ------------------------------------------------------- NIVEAU 3 : identification
    print("\n=== NIVEAU 3 : identification du Pokémon ===")
    res_id, rappels_id, modeles_id = [], {}, {}
    essais = [("Histogramme couleurs", X_hist, n) for n in ("Régression logistique", "Forêt aléatoire")
              if n in candidats]
    essais += [("CNN MobileNetV2", X_cnn, n) for n in candidats]
    for nom_feat, X, nom_mod in essais:
        t0 = time.time()
        modele = clone(candidats[nom_mod]).fit(X[i_train], y_id[i_train])
        proba = modele.predict_proba(X[i_test])
        scores, rappel = im.scores_identification(y_id[i_test], proba, modele.classes_)
        cle = f"{nom_feat} + {nom_mod}"
        res_id.append({"Approche": cle, **scores, "temps (s)": time.time() - t0})
        rappels_id[cle] = rappel
        modeles_id[cle] = (nom_feat, nom_mod)
        print(f"  {cle:<50} acc={scores['accuracy']:.3f}  acc_équi={scores['accuracy_equilibree']:.3f}  ({chrono(t0)})")
    res_id = pd.DataFrame(res_id).set_index("Approche").sort_values("accuracy_equilibree", ascending=False)
    res_id.to_csv(im.DOSSIER_RESULTATS / "niveau3_identification.csv")
    # Modèle retenu = le meilleur parmi ceux basés sur le CNN (c'est ce que l'appli utilise ;
    # l'histogramme de couleurs n'est là que comme référence de comparaison).
    meilleur_id = [a for a in res_id.index if a.startswith("CNN")][0]
    print(res_id.round(3).to_string())

    # Pokémon les moins bien reconnus (pour l'analyse des limites)
    pire = rappels_id[meilleur_id].sort_values().head(15)
    pire.to_csv(im.DOSSIER_RESULTATS / "niveau3_pokemon_les_moins_reconnus.csv")

    fig, ax = plt.subplots(figsize=(8, 4))
    rappels_id[meilleur_id].plot.hist(bins=20, ax=ax)
    ax.set_xlabel("Taux de reconnaissance du Pokémon (rappel)")
    ax.set_ylabel("Nombre de Pokémon")
    ax.set_title(f"Niveau 3 – couverture : {meilleur_id}")
    fig.tight_layout(); fig.savefig(im.DOSSIER_RESULTATS / "niveau3_couverture.png", dpi=120)

    # ---------------------------------------------------------------- NIVEAU 2 : types
    print("\n=== NIVEAU 2 : types à partir de l'image ===")
    res_ty, par_pok_ty = [], {}

    # a) Classifieurs multi-label directs (image -> types)
    for nom_mod, modele_base in candidats.items():
        t0 = time.time()
        modele = im.version_multilabel(nom_mod, clone(modele_base)).fit(X_cnn[i_train], Y_types[i_train])
        p = im.proba_multilabel(modele, X_cnn[i_test])
        scores, par_pok = im.scores_types(Y_types[i_test], im.proba_vers_types(p, types), y_id[i_test])
        cle = f"Direct : CNN + {nom_mod}"
        res_ty.append({"Approche": cle, **scores, "temps (s)": time.time() - t0})
        par_pok_ty[cle] = par_pok
        print(f"  {cle:<50} exact={scores['exact']:.3f}  f1_macro={scores['f1_macro']:.3f}  ({chrono(t0)})")

    # b) En passant par l'identification : P(type) = somme des P(Pokémon) ayant ce type
    t0 = time.time()
    nom_feat, nom_mod = modeles_id[meilleur_id]
    X_feat = X_cnn if nom_feat.startswith("CNN") else X_hist
    modele = clone(candidats[nom_mod]).fit(X_feat[i_train], y_id[i_train])
    p_types = proba_types_via_identification(modele.predict_proba(X_feat[i_test]), modele.classes_, df, mlb)
    scores, par_pok = im.scores_types(Y_types[i_test], im.proba_vers_types(p_types, types), y_id[i_test])
    cle = f"Via identification : {meilleur_id}"
    res_ty.append({"Approche": cle, **scores, "temps (s)": time.time() - t0})
    par_pok_ty[cle] = par_pok
    print(f"  {cle:<50} exact={scores['exact']:.3f}  f1_macro={scores['f1_macro']:.3f}")

    res_ty = pd.DataFrame(res_ty).set_index("Approche").sort_values("f1_macro", ascending=False)
    res_ty.to_csv(im.DOSSIER_RESULTATS / "niveau2_types_pokemon_connus.csv")
    print(res_ty.round(3).to_string())

    # c) Généralisation à des Pokémon JAMAIS vus : GroupKFold (toutes les images d'un
    #    même Pokémon sont dans le même pli). L'approche "via identification" est
    #    impossible ici (on ne peut pas reconnaître un Pokémon jamais vu).
    print("\n--- Types sur des Pokémon jamais vus (GroupKFold, 5 plis) ---")
    res_inconnus = []
    for nom_mod in ("Régression logistique", "k-NN (k=5, cosinus)", "MLP (1 couche cachée)"):
        t0 = time.time()
        Y_pred = np.zeros_like(Y_types)
        for tr, te in GroupKFold(n_splits=5).split(X_cnn, Y_types, groups=y_id):
            modele = im.version_multilabel(nom_mod, clone(candidats[nom_mod])).fit(X_cnn[tr], Y_types[tr])
            Y_pred[te] = im.proba_vers_types(im.proba_multilabel(modele, X_cnn[te]), types)
        scores, _ = im.scores_types(Y_types, Y_pred, y_id)
        res_inconnus.append({"Approche": f"Direct : CNN + {nom_mod}", **scores, "temps (s)": time.time() - t0})
        print(f"  {nom_mod:<30} exact={scores['exact']:.3f}  f1_macro={scores['f1_macro']:.3f}  ({chrono(t0)})")
    # Référence : prédire toujours le type le plus fréquent
    Y_ref = np.zeros_like(Y_types); Y_ref[:, Y_types.sum(0).argmax()] = 1
    scores, _ = im.scores_types(Y_types, Y_ref, y_id)
    res_inconnus.append({"Approche": "Référence (type le plus fréquent)", **scores, "temps (s)": 0})
    res_inconnus = pd.DataFrame(res_inconnus).set_index("Approche")
    res_inconnus.to_csv(im.DOSSIER_RESULTATS / "niveau2_types_pokemon_inconnus.csv")
    print(res_inconnus.round(3).to_string())

    # ----------------------------------------------------- modèles finaux + sauvegarde
    print("\n=== Entraînement final sur toutes les images ===")
    # Identification : meilleur modèle du niveau 3
    nom_feat, nom_mod = modeles_id[meilleur_id]
    final_id = clone(candidats[nom_mod]).fit(X_cnn, y_id)
    joblib.dump({"modele": final_id, "nom": meilleur_id, "classes": list(final_id.classes_)},
                im.DOSSIER_MODELES / "identification.joblib", compress=3)

    # Types : meilleur classifieur DIRECT (il marche aussi pour un Pokémon inconnu,
    # contrairement à l'approche via identification).
    directs = [a for a in res_ty.index if a.startswith("Direct")]
    meilleur_ty = directs[0]
    nom_mod_ty = meilleur_ty.split(" + ", 1)[1]
    final_ty = im.version_multilabel(nom_mod_ty, clone(candidats[nom_mod_ty])).fit(X_cnn, Y_types)
    joblib.dump({"modele": final_ty, "nom": meilleur_ty, "classes": types},
                im.DOSSIER_MODELES / "types.joblib", compress=3)
    print(f"  identification : {meilleur_id}\n  types          : {meilleur_ty}")

    # Miniatures : pour chaque Pokémon, l'image la plus "typique" (la plus proche du centre)
    for nom, groupe in df.groupby("pokemon"):
        vecteurs = X_cnn[groupe.index]
        i = groupe.index[np.argmin(np.linalg.norm(vecteurs - vecteurs.mean(0), axis=1))]
        img = im.Image.fromarray(im.preparer_image(df.at[i, "chemin"])).resize((160, 160))
        img.save(im.DOSSIER_MINIATURES / f"{im.normaliser_nom(nom)}.png", optimize=True)
    print("Terminé. Résultats dans niveau2_3/resultats/, modèles dans niveau2_3/modeles/.")


def proba_types_via_identification(proba_id, classes_id, df, mlb):
    """
    P(type t) = somme des probabilités des Pokémon qui ont le type t.
    Ex. : P(Pikachu) = 0,7 et P(Raichu) = 0,2 => P(Electric) = 0,9.
    """
    types_par_pokemon = df.drop_duplicates("pokemon").set_index("pokemon")["types"]
    M = mlb.transform([types_par_pokemon[c] for c in classes_id])  # (nb_pokemon, nb_types)
    return proba_id @ M


if __name__ == "__main__":
    main()
