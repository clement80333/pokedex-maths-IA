import tkinter as tk
from tkinter import ttk

from pokedex_data import STAT_COLUMNS
from pokedex_classifier import PokemonClassifier


BG = "#f1f3f5"
WHITE = "#ffffff"
NAVY = "#173b3b"
TEXT = "#202b32"
MUTED = "#64747b"


class AverageStatsView(tk.Frame):
    """Affiche les moyennes statistiques par type et pour les légendaires."""

    def __init__(self, parent, donnees):
        super().__init__(parent, bg=BG, padx=22, pady=20)
        self.donnees = donnees
        self._creer_interface()

    def _creer_interface(self):
        tk.Label(
            self, text="LES TYPES EN CHIFFRES",
            font=("Segoe UI", 20, "bold"), fg=NAVY, bg=BG
        ).pack(anchor="w")
        tk.Label(
            self,
            text="Statistiques de base moyennes des Pokémon de chaque type.",
            font=("Segoe UI", 10), fg=MUTED, bg=BG
        ).pack(anchor="w", pady=(4, 16))

        cadre = tk.Frame(self, bg=WHITE, padx=10, pady=10)
        cadre.pack(fill="both", expand=True)
        cadre.grid_rowconfigure(0, weight=1)
        cadre.grid_columnconfigure(0, weight=1)
        self.table_canvas = tk.Canvas(
            cadre, bg=WHITE, highlightthickness=0
        )
        self.table_canvas.grid(row=0, column=0, sticky="nsew")
        barre_verticale = ttk.Scrollbar(
            cadre, orient="vertical", command=self.table_canvas.yview
        )
        barre_verticale.grid(row=0, column=1, sticky="ns")
        barre_horizontale = ttk.Scrollbar(
            cadre, orient="horizontal", command=self.table_canvas.xview
        )
        barre_horizontale.grid(row=1, column=0, sticky="ew")
        self.table_canvas.configure(
            yscrollcommand=barre_verticale.set,
            xscrollcommand=barre_horizontale.set,
        )
        self.tableau = tk.Frame(self.table_canvas, bg=WHITE)
        self.table_canvas.create_window(
            (0, 0), window=self.tableau, anchor="nw"
        )
        self.tableau.bind(
            "<Configure>",
            lambda _event: self.table_canvas.configure(
                scrollregion=self.table_canvas.bbox("all")
            ),
        )
        self.entetes = (
            ("Type / groupe", 16),
            ("Nb.", 5),
            ("PV", 7),
            ("Attaque", 9),
            ("Défense", 9),
            ("Att. Spé.", 8),
            ("Déf. Spé.", 8),
            ("Vitesse", 8),
            ("Total stats", 10),
        )
        self._creer_ligne_entetes()
        self.lignes = tk.Frame(self.tableau, bg=WHITE)
        self.lignes.pack(anchor="w")
        self._remplir()

        tk.Label(
            self,
            text="Vert : deux meilleures stats    •    "
                 "Rouge : deux stats les plus faibles",
            font=("Segoe UI", 9, "bold"), fg=NAVY, bg=BG
        ).pack(anchor="w", pady=(10, 0))
        tk.Label(
            self,
            text="Les doubles types comptent dans chaque type ; "
                 "les quatre lignes « Groupe statistique » viennent du "
                 "clustering k-moyennes ; « Légendaires » regroupe les "
                 "Pokémon légendaires.",
            font=("Segoe UI", 9, "italic"), fg=MUTED, bg=BG
        ).pack(anchor="w", pady=(12, 0))

    @staticmethod
    def _creer_cellule(parent, colonne, texte, largeur, fond, couleur, gras=False):
        tk.Label(
            parent, text=texte, width=largeur, anchor="center",
            font=("Segoe UI", 9, "bold" if gras else "normal"),
            fg=couleur, bg=fond, padx=4, pady=6,
            relief="solid", bd=1,
        ).grid(row=0, column=colonne, sticky="nsew", padx=1, pady=1)

    def _creer_ligne_entetes(self):
        ligne = tk.Frame(self.tableau, bg=WHITE)
        ligne.pack(anchor="w")
        for colonne, (titre, largeur) in enumerate(self.entetes):
            self._creer_cellule(
                ligne, colonne, titre, largeur, NAVY, WHITE, gras=True
            )

    def _remplir(self):
        moyennes_types = self.donnees.calculer_moyennes_par_type()
        groupes = PokemonClassifier(self.donnees.pokedex).obtenir_clusters(4)
        for groupe in groupes:
            moyennes_types.append(
                (
                    f"Groupe statistique {groupe['numero']}",
                    *self.donnees.calculer_moyennes_statistiques(
                        groupe["membres"]
                    ),
                )
            )
        legendaires = [
            pokemon for pokemon in self.donnees.pokedex
            if pokemon["Legendary"].strip().casefold() == "true"
        ]
        if legendaires:
            moyennes_types.append(
                (
                    "Légendaires",
                    *self.donnees.calculer_moyennes_statistiques(legendaires),
                )
            )

        for index, (nom, nombre, *moyennes) in enumerate(moyennes_types):
            ligne = tk.Frame(self.lignes, bg=WHITE)
            ligne.pack(anchor="w")
            speciale = nom == "Légendaires"
            meilleures, pires = self.donnees.indices_stats_extremes(moyennes)
            valeurs = (
                nom,
                nombre,
                *(f"{moyenne:.1f}" for moyenne in moyennes),
            )
            for colonne, (valeur, (_titre, largeur)) in enumerate(
                zip(valeurs, self.entetes)
            ):
                if speciale:
                    fond, couleur, gras = "#fff0d6", "#704800", True
                elif 2 <= colonne < 2 + len(STAT_COLUMNS):
                    indice_stat = colonne - 2
                    if indice_stat in meilleures:
                        fond, couleur, gras = "#d8f3dc", "#205c34", True
                    elif indice_stat in pires:
                        fond, couleur, gras = "#fde2e2", "#8a2929", True
                    else:
                        fond = "#f3f7f1" if index % 2 == 0 else WHITE
                        couleur, gras = TEXT, False
                else:
                    fond = "#f3f7f1" if index % 2 == 0 else WHITE
                    couleur, gras = TEXT, False
                self._creer_cellule(
                    ligne, colonne, valeur, largeur, fond, couleur, gras
                )
