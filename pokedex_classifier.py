"""Orchestre le partage des données et l'évaluation des modèles."""

import math
import random
from collections import defaultdict

from pokedex_data import STAT_COLUMNS
from pokedex_models import MODELE_CLASSES


class PokemonClassifier:
    """Gère le partage des données, l'entraînement et l'évaluation."""

    CIBLES = {
        "Légendaire": None,
        "Type principal": "Type 1",
    }
    MODELES = {
        nom: modele.EXPLICATION for nom, modele in MODELE_CLASSES.items()
    }
    PROPORTION_TEST = 0.2
    GRAINE_ALEATOIRE = 42

    def __init__(self, pokemon):
        self.pokemon = list(pokemon)
        self.graine_aleatoire = self.GRAINE_ALEATOIRE
        self._decoupages = {}
        self._modeles_entraines = {}

    def entrainer_et_evaluer(self, cible, nom_modele):
        """Entraîne un modèle sur 80 % des données et l'évalue sur les 20 % restants."""
        if nom_modele not in self.MODELES:
            raise ValueError(f"Modèle de classification inconnu : {nom_modele}")
        if cible not in self.CIBLES:
            raise ValueError(f"Cible de classification inconnue : {cible}")

        cle = (cible, nom_modele)
        if cle in self._modeles_entraines:
            return self._modeles_entraines[cle]

        entrainement, test = self.obtenir_decoupage(cible)
        moyennes, ecarts_types = self._parametres_normalisation(entrainement)
        exemples_entrainement = [
            (
                self._normaliser(
                    self._extraire_stats(pokemon), moyennes, ecarts_types
                ),
                self.etiquette(pokemon, cible),
            )
            for pokemon in entrainement
        ]
        modele_entraine = self._entrainer_modele(
            nom_modele, exemples_entrainement, cible
        )

        etiquettes_reelles = [
            self.etiquette(pokemon, cible) for pokemon in test
        ]
        predictions = [
            modele_entraine.predire(
                self._normaliser(
                    self._extraire_stats(pokemon), moyennes, ecarts_types
                )
            )
            for pokemon in test
        ]
        exactitude = (
            sum(
                prediction == reelle
                for prediction, reelle in zip(predictions, etiquettes_reelles)
            )
            / len(test)
            if test else 0.0
        )
        resultat = {
            "modele": modele_entraine,
            "moyennes": moyennes,
            "ecarts_types": ecarts_types,
            "entrainement": entrainement,
            "test": test,
            "nombre_entrainement": len(entrainement),
            "nombre_test": len(test),
            "exactitude": exactitude,
            "f1_macro": self._calculer_f1_macro(
                etiquettes_reelles, predictions
            ),
            "predictions": predictions,
            "etiquettes_reelles": etiquettes_reelles,
        }
        if cible == "Légendaire":
            precision, rappel = self._calculer_precision_rappel(
                "Légendaire", etiquettes_reelles, predictions
            )
            resultat["precision_legendaire"] = precision
            resultat["rappel_legendaire"] = rappel
        self._modeles_entraines[cle] = resultat
        return resultat

    def predire(self, statistiques, cible, nom_modele):
        """Prédit une classe après normalisation selon les données d'entraînement."""
        resultat = self.entrainer_et_evaluer(cible, nom_modele)
        vecteur = self._normaliser(
            [float(valeur) for valeur in statistiques],
            resultat["moyennes"],
            resultat["ecarts_types"],
        )
        return resultat["modele"].predire(vecteur)

    def obtenir_decoupage(self, cible):
        """Retourne le même partage stratifié à chaque appel pour cette cible."""
        if cible not in self.CIBLES:
            raise ValueError(f"Cible de classification inconnue : {cible}")
        if cible not in self._decoupages:
            self._decoupages[cible] = self._separer_groupes(cible)
        return self._decoupages[cible]

    def changer_decoupage(self):
        """Change la graine et invalide les découpages et modèles mémorisés."""
        self.graine_aleatoire += 1
        self._decoupages.clear()
        self._modeles_entraines.clear()
        return self.graine_aleatoire

    def _separer_groupes(self, cible):
        """Produit un partage 80/20 en gardant les proportions de chaque classe."""
        groupes = defaultdict(list)
        for pokemon in self.pokemon:
            groupes[self.etiquette(pokemon, cible)].append(pokemon)

        generateur = random.Random(
            self.graine_aleatoire + sum(ord(caractere) for caractere in cible)
        )
        quotas_test = {
            etiquette: len(groupe) * self.PROPORTION_TEST
            for etiquette, groupe in groupes.items()
        }
        nombres_test = {
            etiquette: int(quotas_test[etiquette])
            for etiquette in groupes
        }
        for etiquette, groupe in groupes.items():
            if len(groupe) > 1:
                nombres_test[etiquette] = max(nombres_test[etiquette], 1)

        nombre_test_vise = round(len(self.pokemon) * self.PROPORTION_TEST)
        while sum(nombres_test.values()) < nombre_test_vise:
            candidats = [
                etiquette for etiquette, groupe in groupes.items()
                if nombres_test[etiquette] < len(groupe) - 1
            ]
            if not candidats:
                break
            etiquette = max(
                candidats,
                key=lambda nom: (quotas_test[nom] - nombres_test[nom], nom),
            )
            nombres_test[etiquette] += 1
        while sum(nombres_test.values()) > nombre_test_vise:
            candidats = [
                etiquette for etiquette, groupe in groupes.items()
                if nombres_test[etiquette] > (1 if len(groupe) > 1 else 0)
            ]
            if not candidats:
                break
            etiquette = min(
                candidats,
                key=lambda nom: (quotas_test[nom] - nombres_test[nom], nom),
            )
            nombres_test[etiquette] -= 1

        entrainement, test = [], []
        for etiquette in sorted(groupes):
            groupe = groupes[etiquette][:]
            generateur.shuffle(groupe)
            nombre_test = nombres_test[etiquette]
            test.extend(groupe[:nombre_test])
            entrainement.extend(groupe[nombre_test:])
        generateur.shuffle(entrainement)
        generateur.shuffle(test)
        return entrainement, test

    @classmethod
    def etiquette(self, pokemon, cible):
        """Retourne l'étiquette de type principal ou de statut légendaire."""
        colonne = self.CIBLES[cible]
        if colonne == "Type 1":
            return pokemon[colonne]
        est_legendaire = pokemon["Legendary"].strip().casefold() == "true"
        return "Légendaire" if est_legendaire else "Non légendaire"

    @staticmethod
    def _extraire_stats(pokemon):
        """Lit le vecteur d'entrée dans le même ordre que les champs du formulaire."""
        return [float(pokemon[colonne]) for _, colonne in STAT_COLUMNS]

    @staticmethod
    def _parametres_normalisation(pokemon):
        """Calcule moyenne et écart-type par statistique depuis l'entraînement."""
        colonnes = [
            [PokemonClassifier._extraire_stats(individu)[index]
             for individu in pokemon]
            for index in range(len(STAT_COLUMNS))
        ]
        moyennes = [sum(colonne) / len(colonne) for colonne in colonnes]
        ecarts_types = [
            math.sqrt(
                sum((valeur - moyenne) ** 2 for valeur in colonne)
                / len(colonne)
            )
            for colonne, moyenne in zip(colonnes, moyennes)
        ]
        return moyennes, [ecart or 1.0 for ecart in ecarts_types]

    @staticmethod
    def _normaliser(statistiques, moyennes, ecarts_types):
        """Met les stats sur une échelle comparable pour les modèles."""
        return [
            (valeur - moyenne) / ecart
            for valeur, moyenne, ecart in zip(
                statistiques, moyennes, ecarts_types
            )
        ]

    def _entrainer_modele(self, nom_modele, exemples, cible):
        """Crée et entraîne l'implémentation dédiée au modèle demandé."""
        classe_modele = MODELE_CLASSES[nom_modele]
        graine = self.graine_aleatoire + sum(
            ord(caractere) for caractere in cible
        )
        modele = classe_modele(seed=graine)
        modele.entrainer(exemples)
        return modele

    @staticmethod
    def _calculer_f1_macro(etiquettes_reelles, predictions):
        """Calcule le F1 moyen par classe, sans favoriser les classes majoritaires."""
        classes = set(etiquettes_reelles) | set(predictions)
        scores = []
        for etiquette in classes:
            precision, rappel = PokemonClassifier._calculer_precision_rappel(
                etiquette, etiquettes_reelles, predictions
            )
            scores.append(
                2 * precision * rappel / (precision + rappel)
                if precision + rappel else 0.0
            )
        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def _calculer_precision_rappel(etiquette, etiquettes_reelles, predictions):
        """Mesure la précision et le rappel d'une étiquette particulière."""
        vrais_positifs = sum(
            reelle == etiquette and prediction == etiquette
            for reelle, prediction in zip(etiquettes_reelles, predictions)
        )
        faux_positifs = sum(
            reelle != etiquette and prediction == etiquette
            for reelle, prediction in zip(etiquettes_reelles, predictions)
        )
        faux_negatifs = sum(
            reelle == etiquette and prediction != etiquette
            for reelle, prediction in zip(etiquettes_reelles, predictions)
        )
        precision = (
            vrais_positifs / (vrais_positifs + faux_positifs)
            if vrais_positifs + faux_positifs else 0.0
        )
        rappel = (
            vrais_positifs / (vrais_positifs + faux_negatifs)
            if vrais_positifs + faux_negatifs else 0.0
        )
        return precision, rappel
