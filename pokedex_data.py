import csv
from pathlib import Path


STAT_COLUMNS = [
    ("PV", "HP"),
    ("Attaque", "Attack"),
    ("Défense", "Defense"),
    ("Att. Spé.", "Sp. Atk"),
    ("Déf. Spé.", "Sp. Def"),
    ("Vitesse", "Speed"),
]

SORT_COLUMNS = {
    "Numéro Pokédex": "#",
    "PV": "HP",
    "Attaque": "Attack",
    "Défense": "Defense",
    "Att. Spé.": "Sp. Atk",
    "Déf. Spé.": "Sp. Def",
    "Vitesse": "Speed",
}


class PokedexData:
    """Charge les données et fournit les opérations de recherche du Pokédex."""

    COLONNES_REQUISES = {
        "#", "Name", "Type 1", "Type 2", "HP", "Attack", "Defense",
        "Sp. Atk", "Sp. Def", "Speed", "Generation", "Legendary",
    }

    def __init__(self, chemin_csv=None):
        self.chemin_csv = chemin_csv or Path(__file__).with_name("pokemon.csv")
        self.pokedex = self._charger_pokedex()

    def _charger_pokedex(self):
        with self.chemin_csv.open(encoding="utf-8-sig", newline="") as fichier:
            lecteur = csv.DictReader(fichier)
            colonnes = set(lecteur.fieldnames or [])
            manquantes = self.COLONNES_REQUISES - colonnes
            if manquantes:
                raise ValueError(
                    "Colonnes absentes du CSV : "
                    + ", ".join(sorted(manquantes))
                )
            return list(lecteur)

    @staticmethod
    def calculer_moyennes_statistiques(pokemon):
        nombre = len(pokemon)
        moyennes = [
            sum(int(individu[colonne]) for individu in pokemon) / nombre
            for _, colonne in STAT_COLUMNS
        ]
        moyenne_totale = sum(
            sum(int(individu[colonne]) for _, colonne in STAT_COLUMNS)
            for individu in pokemon
        ) / nombre
        return nombre, *moyennes, moyenne_totale

    def calculer_moyennes_par_type(self):
        types = {}
        for pokemon in self.pokedex:
            for colonne in ("Type 1", "Type 2"):
                type_pokemon = pokemon[colonne]
                if type_pokemon:
                    types.setdefault(type_pokemon.casefold(), type_pokemon)

        moyennes_par_type = []
        for cle_type, nom_type in sorted(
            types.items(), key=lambda element: element[1].casefold()
        ):
            membres = [
                pokemon for pokemon in self.pokedex
                if cle_type in (
                    pokemon["Type 1"].casefold(),
                    pokemon["Type 2"].casefold(),
                )
            ]
            moyennes_par_type.append(
                (nom_type, *self.calculer_moyennes_statistiques(membres))
            )
        return moyennes_par_type

    @staticmethod
    def indices_stats_extremes(statistiques):
        classement = sorted(
            range(len(STAT_COLUMNS)),
            key=lambda index: statistiques[index],
            reverse=True,
        )
        return set(classement[:2]), set(classement[-2:])

    def filtrer(self, recherche, types, generations, legendaires):
        resultats = []
        for pokemon in self.pokedex:
            if recherche not in pokemon["Name"].casefold():
                continue

            types_pokemon = {
                pokemon["Type 1"].casefold(),
                pokemon["Type 2"].casefold(),
            }
            if types and not types.intersection(types_pokemon):
                continue
            if generations and int(pokemon["Generation"]) not in generations:
                continue

            est_legendaire = pokemon["Legendary"].strip().casefold() == "true"
            if legendaires and est_legendaire not in legendaires:
                continue
            resultats.append(pokemon)
        return resultats

    @staticmethod
    def trier(pokedex, critere, ordre):
        if critere == "Total stats":
            cle_tri = lambda pokemon: sum(
                int(pokemon[colonne]) for _, colonne in STAT_COLUMNS
            )
        else:
            colonne = SORT_COLUMNS[critere]
            cle_tri = lambda pokemon: int(pokemon[colonne])
        return sorted(pokedex, key=cle_tri, reverse=ordre == "Décroissant")
