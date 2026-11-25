import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict

from pokedex_data import STAT_COLUMNS


class ClassificationModel(ABC):
    """Interface commune : chaque modèle apprend des exemples puis prédit."""

    NOM = ""
    EXPLICATION = ""

    def __init__(self, seed=42):
        """Conserve une graine pour rendre les modèles aléatoires reproductibles."""
        self.seed = seed

    @abstractmethod
    def entrainer(self, exemples):
        """Apprend un modèle depuis des couples (stats_normalisées, étiquette)."""

    @abstractmethod
    def predire(self, statistiques):
        """Retourne l'étiquette prédite pour un vecteur de statistiques."""

    def predire_probabilites(self, statistiques):
        """Retourne une estimation approximative des probabilités par classe."""
        prediction = self.predire(statistiques)
        return {prediction: 1.0}


class KNearestNeighborsModel(ClassificationModel):
    """Vote pour la classe la plus fréquente parmi les voisins les plus proches."""

    NOM = "k-NN (5 voisins)"
    EXPLICATION = (
        "Le k-NN compare les six statistiques saisies aux lignes "
        "d'entraînement. Avant la comparaison, chaque statistique est "
        "normalisée à partir des moyennes et écarts-types de l'entraînement : "
        "ainsi, une statistique avec de grandes valeurs ne domine pas "
        "seulement à cause de son échelle. Le modèle calcule une distance "
        "entre le Pokémon saisi et chaque exemple, puis retient les cinq "
        "exemples les plus proches.\n\n"
        "Exemple simplifié : si les cinq voisins ont les types Électrik, "
        "Électrik, Eau, Électrik et Normal, le vote majoritaire donne "
        "Électrik. Pour la cible Légendaire, le vote porte sur les statuts.\n\n"
        "Le modèle conserve les exemples d'entraînement et les compare à "
        "chaque nouvelle saisie, plutôt que de construire des règles."
    )
    NOMBRE_VOISINS = 5

    def entrainer(self, exemples):
        """Garde les vecteurs d'entraînement à consulter lors des prédictions."""
        self.exemples = exemples

    def predire(self, statistiques):
        """Retourne la classe majoritaire des cinq exemples les plus proches."""
        distances = []
        for vecteur, etiquette in self.exemples:
            distance = self._distance(statistiques, vecteur)
            distances.append((distance, etiquette))

        voisins = sorted(distances)[:self.NOMBRE_VOISINS]
        votes = defaultdict(int)
        distances_par_classe = defaultdict(float)
        for distance, etiquette in voisins:
            votes[etiquette] += 1
            distances_par_classe[etiquette] += distance
        return min(
            votes,
            key=lambda etiquette: (
                -votes[etiquette],
                distances_par_classe[etiquette],
                etiquette,
            ),
        )

    def predire_probabilites(self, statistiques):
        distances = [
            (self._distance(statistiques, vecteur), etiquette)
            for vecteur, etiquette in self.exemples
        ]
        voisins = sorted(distances)[:self.NOMBRE_VOISINS]
        poids = {
            etiquette: 0.0
            for _distance, etiquette in voisins
        }
        for distance, etiquette in voisins:
            poids[etiquette] += 1.0 / (1.0 + distance)
        total = sum(poids.values())
        return {
            etiquette: poids[etiquette] / total
            for etiquette in sorted(poids)
        }

    @staticmethod
    def _distance(premier, second):
        """Calcule la distance euclidienne au carré entre deux vecteurs."""
        return sum(
            (valeur - autre) ** 2 for valeur, autre in zip(premier, second)
        )


class GaussianNaiveBayesModel(ClassificationModel):
    """Évalue la probabilité des stats dans chaque classe avec une loi normale."""

    NOM = "Naïf bayésien gaussien"
    EXPLICATION = (
        "Le Bayésien gaussien apprend, pour chaque classe et chaque "
        "statistique, une moyenne et une variance à partir des lignes "
        "d'entraînement. Il suppose que les six statistiques sont "
        "indépendantes une fois la classe connue et suivent une distribution "
        "gaussienne. Il choisit ensuite la classe la plus probable.\n\n"
        "Exemple fictif : si la moyenne d'Attaque apprise pour Eau est 55 "
        "et celle pour Électrik 100, une Attaque saisie de 58 favorise Eau "
        "pour cette statistique. Le modèle combine ensuite les probabilités "
        "des six stats. « Naïf » décrit son hypothèse d'indépendance, qui "
        "peut être imparfaite en pratique."
    )

    def entrainer(self, exemples):
        """Calcule les probabilités a priori, moyennes et variances par classe."""
        groupes = defaultdict(list)
        for statistiques, etiquette in exemples:
            groupes[etiquette].append(statistiques)

        self.parametres_classes = {}
        for etiquette, vecteurs in groupes.items():
            moyennes = [
                sum(vecteur[index] for vecteur in vecteurs) / len(vecteurs)
                for index in range(len(STAT_COLUMNS))
            ]
            variances = [
                max(
                    sum(
                        (vecteur[index] - moyenne) ** 2 for vecteur in vecteurs
                    ) / len(vecteurs),
                    1e-9,
                )
                for index, moyenne in enumerate(moyennes)
            ]
            self.parametres_classes[etiquette] = (
                len(vecteurs) / len(exemples),
                moyennes,
                variances,
            )

    def predire(self, statistiques):
        """Choisit la classe de plus grande probabilité a posteriori."""
        probabilites = {}
        for etiquette, (prior, moyennes, variances) in self.parametres_classes.items():
            log_probabilite = math.log(prior)
            for valeur, moyenne, variance in zip(
                statistiques, moyennes, variances
            ):
                log_probabilite -= 0.5 * (
                    math.log(2 * math.pi * variance)
                    + (valeur - moyenne) ** 2 / variance
                )
            probabilites[etiquette] = log_probabilite
        return max(
            probabilites, key=lambda etiquette: (probabilites[etiquette], etiquette)
        )

    def predire_probabilites(self, statistiques):
        scores = {}
        for etiquette, (prior, moyennes, variances) in self.parametres_classes.items():
            log_probabilite = math.log(prior)
            for valeur, moyenne, variance in zip(statistiques, moyennes, variances):
                log_probabilite -= 0.5 * (
                    math.log(2 * math.pi * variance)
                    + (valeur - moyenne) ** 2 / variance
                )
            scores[etiquette] = log_probabilite
        maximum = max(scores.values())
        exponentiels = {
            etiquette: math.exp(score - maximum)
            for etiquette, score in scores.items()
        }
        total = sum(exponentiels.values())
        return {
            etiquette: valeur / total
            for etiquette, valeur in exponentiels.items()
        }


class CentroidModel(ClassificationModel):
    """Attribue la classe dont le centre moyen est le plus proche."""

    NOM = "Centroïdes"
    EXPLICATION = (
        "Le modèle calcule un point moyen (centroïde) pour chaque classe "
        "à partir des statistiques normalisées des Pokémon d'entraînement. "
        "Une nouvelle saisie est attribuée au centroïde le plus proche.\n\n"
        "Exemple : si le profil est plus proche du centroïde Eau que des "
        "autres centroïdes, la prédiction est Eau. Contrairement au k-NN, "
        "il compare aux moyennes de classe, pas à chaque Pokémon."
    )

    def entrainer(self, exemples):
        """Calcule la moyenne de chaque statistique pour toutes les classes."""
        groupes = defaultdict(list)
        for statistiques, etiquette in exemples:
            groupes[etiquette].append(statistiques)
        self.centres = {
            etiquette: [
                sum(vecteur[index] for vecteur in vecteurs) / len(vecteurs)
                for index in range(len(STAT_COLUMNS))
            ]
            for etiquette, vecteurs in groupes.items()
        }

    def predire(self, statistiques):
        """Retourne l'étiquette du centroïde le plus proche."""
        return min(
            self.centres,
            key=lambda etiquette: (
                KNearestNeighborsModel._distance(
                    statistiques, self.centres[etiquette]
                ),
                etiquette,
            ),
        )

    def predire_probabilites(self, statistiques):
        distances = {
            etiquette: KNearestNeighborsModel._distance(statistiques, centre)
            for etiquette, centre in self.centres.items()
        }
        minimum = min(distances.values())
        poids = {
            etiquette: math.exp(-(distance - minimum))
            for etiquette, distance in distances.items()
        }
        total = sum(poids.values())
        return {
            etiquette: poids[etiquette] / total
            for etiquette in sorted(poids)
        }


class DecisionTreeModel(ClassificationModel):
    """Pose des seuils successifs sur les stats pour atteindre une classe."""

    NOM = "Arbre de décision"
    EXPLICATION = (
        "L'arbre apprend une suite de questions, par exemple « Attaque "
        "inférieure ou égale à un seuil ? ». Il choisit des seuils qui "
        "séparent au mieux les classes; la prédiction suit les réponses "
        "jusqu'à une feuille.\n\n"
        "Exemple simplifié : Attaque > 90 ? puis Vitesse > 80 ? La classe "
        "majoritaire dans la feuille finale est prédite. La profondeur "
        "est limitée pour éviter de trop mémoriser les exemples."
    )
    PROFONDEUR_MAX = 7
    EXEMPLES_MIN_PAR_FEUILLE = 8

    def __init__(self, seed=42, nombre_stats_par_coupure=None):
        """Configure l'arbre et, pour une forêt, le tirage aléatoire des stats."""
        super().__init__(seed)
        self.nombre_stats_par_coupure = nombre_stats_par_coupure
        self.generateur = random.Random(seed)

    def entrainer(self, exemples):
        """Construit l'arbre en choisissant les coupures avec le critère Gini."""
        self.racine = self._construire(exemples, profondeur=0)

    def predire(self, statistiques):
        """Suit les coupures de l'arbre et retourne l'étiquette de la feuille."""
        noeud = self.racine
        while noeud[0] == "noeud":
            _, index_stat, seuil, gauche, droite = noeud
            noeud = gauche if statistiques[index_stat] <= seuil else droite
        return noeud[1]

    def predire_probabilites(self, statistiques):
        noeud = self.racine
        while noeud[0] == "noeud":
            _, index_stat, seuil, gauche, droite = noeud
            noeud = gauche if statistiques[index_stat] <= seuil else droite
        if len(noeud) >= 3:
            frequences = noeud[2]
            total = sum(frequences.values())
            return {
                etiquette: nombre / total
                for etiquette, nombre in sorted(frequences.items())
            }
        return {noeud[1]: 1.0}

    def _construire(self, exemples, profondeur):
        """Crée récursivement une feuille ou sépare les exemples en deux."""
        frequences = defaultdict(int)
        for _, etiquette in exemples:
            frequences[etiquette] += 1
        classe_majoritaire = min(
            frequences, key=lambda etiquette: (-frequences[etiquette], etiquette)
        )
        if (
            len(frequences) == 1
            or profondeur >= self.PROFONDEUR_MAX
            or len(exemples) < self.EXEMPLES_MIN_PAR_FEUILLE * 2
        ):
            return ("feuille", classe_majoritaire, dict(frequences))

        indices_stats = list(range(len(STAT_COLUMNS)))
        if self.nombre_stats_par_coupure is not None:
            indices_stats = self.generateur.sample(
                indices_stats,
                min(self.nombre_stats_par_coupure, len(indices_stats)),
            )
        coupure = self._meilleure_coupure(exemples, indices_stats)
        if coupure is None:
            return ("feuille", classe_majoritaire, dict(frequences))

        index_stat, seuil = coupure
        gauche = [
            exemple for exemple in exemples
            if exemple[0][index_stat] <= seuil
        ]
        droite = [
            exemple for exemple in exemples
            if exemple[0][index_stat] > seuil
        ]
        return (
            "noeud",
            index_stat,
            seuil,
            self._construire(gauche, profondeur + 1),
            self._construire(droite, profondeur + 1),
        )

    def _meilleure_coupure(self, exemples, indices_stats):
        """Trouve le seuil qui réduit le plus le mélange des étiquettes."""
        total = defaultdict(int)
        for _, etiquette in exemples:
            total[etiquette] += 1
        nombre_exemples = len(exemples)
        meilleure_impurete = self._gini(total, nombre_exemples)
        meilleure_coupure = None

        for index_stat in indices_stats:
            tries = sorted(exemples, key=lambda exemple: exemple[0][index_stat])
            gauche = defaultdict(int)
            droite = total.copy()
            for index in range(nombre_exemples - 1):
                valeur, etiquette = tries[index][0][index_stat], tries[index][1]
                gauche[etiquette] += 1
                droite[etiquette] -= 1
                taille_gauche = index + 1
                taille_droite = nombre_exemples - taille_gauche
                if (
                    taille_gauche < self.EXEMPLES_MIN_PAR_FEUILLE
                    or taille_droite < self.EXEMPLES_MIN_PAR_FEUILLE
                ):
                    continue
                valeur_suivante = tries[index + 1][0][index_stat]
                if valeur == valeur_suivante:
                    continue
                impurete = (
                    taille_gauche * self._gini(gauche, taille_gauche)
                    + taille_droite * self._gini(droite, taille_droite)
                ) / nombre_exemples
                if impurete < meilleure_impurete:
                    meilleure_impurete = impurete
                    meilleure_coupure = (
                        index_stat, (valeur + valeur_suivante) / 2
                    )
        return meilleure_coupure

    @staticmethod
    def _gini(frequences, total):
        """Mesure le mélange des étiquettes dans un groupe d'exemples."""
        return 1 - sum(
            (nombre / total) ** 2 for nombre in frequences.values()
        )


class RandomForestModel(ClassificationModel):
    """Combine le vote de plusieurs arbres entraînés de façons aléatoires."""

    NOM = "Forêt aléatoire"
    EXPLICATION = (
        "La forêt entraîne neuf arbres sur des échantillons tirés avec remise "
        "dans les données d'entraînement. À chaque coupure, chaque arbre ne "
        "considère que deux statistiques tirées au hasard.\n\n"
        "Exemple : si cinq arbres votent Eau et quatre votent Normal, la "
        "forêt prédit Eau. Le vote de plusieurs arbres réduit la dépendance "
        "à une seule suite de questions."
    )
    NOMBRE_ARBRES = 9
    NOMBRE_STATS_PAR_ARBRE = 2

    def entrainer(self, exemples):
        """Entraîne les arbres sur des échantillons bootstrap de même taille."""
        generateur = random.Random(self.seed)
        self.arbres = []
        for _ in range(self.NOMBRE_ARBRES):
            echantillon = [
                exemples[generateur.randrange(len(exemples))]
                for _ in exemples
            ]
            arbre = DecisionTreeModel(
                seed=generateur.randrange(2**32),
                nombre_stats_par_coupure=self.NOMBRE_STATS_PAR_ARBRE,
            )
            arbre.entrainer(echantillon)
            self.arbres.append(arbre)

    def predire(self, statistiques):
        """Retourne l'étiquette qui reçoit le plus de votes des arbres."""
        votes = defaultdict(int)
        for arbre in self.arbres:
            votes[arbre.predire(statistiques)] += 1
        return min(votes, key=lambda etiquette: (-votes[etiquette], etiquette))

    def predire_probabilites(self, statistiques):
        moyenne = defaultdict(float)
        for arbre in self.arbres:
            for etiquette, probabilite in arbre.predire_probabilites(statistiques).items():
                moyenne[etiquette] += probabilite / len(self.arbres)
        return dict(sorted(moyenne.items()))


MODELE_CLASSES = {
    modele.NOM: modele
    for modele in (
        KNearestNeighborsModel,
        GaussianNaiveBayesModel,
        CentroidModel,
        DecisionTreeModel,
        RandomForestModel,
    )
}
