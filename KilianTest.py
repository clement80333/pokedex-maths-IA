import csv
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

CSV_PATH = Path(__file__).with_name("pokemon.csv")

BG = "#f3f6fc"
WHITE = "#ffffff"
NAVY = "#172554"
BLUE = "#2563eb"
TEXT = "#1e293b"
MUTED = "#64748b"


def charger_pokedex():
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as fichier:
        lecteur = csv.DictReader(fichier)
        colonnes_requises = {
            "#", "Name", "Type 1", "Type 2", "HP", "Attack", "Defense",
            "Sp. Atk", "Sp. Def", "Speed", "Generation", "Legendary",
        }

        colonnes = set(lecteur.fieldnames or [])
        manquantes = colonnes_requises - colonnes
        if manquantes:
            raise ValueError(
                "Colonnes absentes du CSV : " + ", ".join(sorted(manquantes))
            )

        return list(lecteur)


class PokedexApp:
    def __init__(self, fenetre, pokedex):
        self.fenetre = fenetre
        self.pokedex = pokedex
        self.resultats = []
        self.index_actuel = 0

        fenetre.title("Pokédex")
        fenetre.geometry("1000x680")
        fenetre.minsize(800, 560)
        fenetre.configure(bg=BG)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Stat.Horizontal.TProgressbar",
            troughcolor="#e2e8f0",
            background=BLUE,
            thickness=10,
        )

        self.creer_interface()
        self.actualiser_liste()

    def creer_interface(self):
        entete = tk.Frame(self.fenetre, bg=NAVY, padx=26, pady=18)
        entete.pack(fill="x")

        tk.Label(
            entete, text="POKÉDEX", font=("Segoe UI", 24, "bold"),
            fg=WHITE, bg=NAVY
        ).pack(side="left")

        self.compteur = tk.Label(
            entete, font=("Segoe UI", 11), fg="#cbd5e1", bg=NAVY
        )
        self.compteur.pack(side="right", pady=8)

        filtres = tk.Frame(self.fenetre, bg=WHITE, padx=18, pady=14)
        filtres.pack(fill="x", padx=20, pady=(16, 12))

        tk.Label(
            filtres, text="Rechercher", font=("Segoe UI", 10, "bold"),
            fg=TEXT, bg=WHITE
        ).pack(side="left", padx=(0, 10))

        self.recherche = tk.StringVar()
        champ = tk.Entry(
            filtres, textvariable=self.recherche, font=("Segoe UI", 11),
            relief="solid", bd=1
        )
        champ.pack(side="left", fill="x", expand=True, ipady=6)
        self.recherche.trace_add("write", lambda *_: self.actualiser_liste())

        tk.Label(
            filtres, text="Type", font=("Segoe UI", 10, "bold"),
            fg=TEXT, bg=WHITE
        ).pack(side="left", padx=(18, 8))

        types = sorted({
            pokemon[colonne]
            for pokemon in self.pokedex
            for colonne in ("Type 1", "Type 2")
            if pokemon[colonne]
        })
        self.type_choisi = tk.StringVar(value="Tous")
        choix_type = ttk.Combobox(
            filtres,
            textvariable=self.type_choisi,
            values=["Tous"] + types,
            state="readonly",
            width=15,
        )
        choix_type.pack(side="left", ipady=4)
        choix_type.bind(
            "<<ComboboxSelected>>", lambda _event: self.actualiser_liste()
        )

        contenu = tk.Frame(self.fenetre, bg=BG, padx=20, pady=4)
        contenu.pack(fill="both", expand=True)
        contenu.grid_rowconfigure(0, weight=1)
        contenu.grid_columnconfigure(1, weight=1)

        panneau_liste = tk.Frame(contenu, bg=WHITE, padx=14, pady=14, width=310)
        panneau_liste.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        panneau_liste.grid_propagate(False)

        tk.Label(
            panneau_liste, text="Pokémon", font=("Segoe UI", 15, "bold"),
            fg=TEXT, bg=WHITE
        ).pack(anchor="w", pady=(0, 10))

        cadre_liste = tk.Frame(panneau_liste, bg=WHITE)
        cadre_liste.pack(fill="both", expand=True)

        barre = ttk.Scrollbar(cadre_liste)
        barre.pack(side="right", fill="y")

        self.liste = tk.Listbox(
            cadre_liste,
            font=("Segoe UI", 10),
            bg=WHITE,
            fg=TEXT,
            selectbackground=BLUE,
            selectforeground=WHITE,
            activestyle="none",
            relief="flat",
            highlightthickness=0,
            yscrollcommand=barre.set,
        )
        self.liste.pack(side="left", fill="both", expand=True)
        barre.config(command=self.liste.yview)
        self.liste.bind("<<ListboxSelect>>", self.selectionner_pokemon)

        self.panneau_fiche = tk.Frame(contenu, bg=WHITE, padx=26, pady=22)
        self.panneau_fiche.grid(row=0, column=1, sticky="nsew")

        navigation = tk.Frame(self.panneau_fiche, bg=WHITE)
        navigation.pack(side="bottom", fill="x", pady=(18, 0))

        tk.Button(
            navigation, text="← Précédent", command=lambda: self.naviguer(-1),
            bg=NAVY, fg=WHITE, relief="flat", padx=12, pady=7,
            activebackground=BLUE, activeforeground=WHITE
        ).pack(side="left")

        self.position = tk.Label(
            navigation, text="", font=("Segoe UI", 10),
            fg=MUTED, bg=WHITE
        )
        self.position.pack(side="left", expand=True)

        tk.Button(
            navigation, text="Suivant →", command=lambda: self.naviguer(1),
            bg=NAVY, fg=WHITE, relief="flat", padx=12, pady=7,
            activebackground=BLUE, activeforeground=WHITE
        ).pack(side="right")

        self.details = tk.Frame(self.panneau_fiche, bg=WHITE)
        self.details.pack(fill="both", expand=True)

    def actualiser_liste(self):
        recherche = self.recherche.get().strip().casefold()
        type_choisi = self.type_choisi.get().casefold()
        pokemon_actuel = (
            self.resultats[self.index_actuel] if self.resultats else None
        )

        self.resultats = [
            pokemon for pokemon in self.pokedex
            if recherche in pokemon["Name"].casefold()
            and (
                type_choisi == "tous"
                or type_choisi in (
                    pokemon["Type 1"].casefold(),
                    pokemon["Type 2"].casefold(),
                )
            )
        ]

        self.liste.delete(0, tk.END)
        for pokemon in self.resultats:
            types = pokemon["Type 1"]
            if pokemon["Type 2"]:
                types += f" / {pokemon['Type 2']}"
            self.liste.insert(
                tk.END, f"#{pokemon['#']:>3}   {pokemon['Name']}  ·  {types}"
            )

        if pokemon_actuel in self.resultats:
            self.index_actuel = self.resultats.index(pokemon_actuel)
        else:
            self.index_actuel = 0

        self.compteur.config(text=f"{len(self.resultats)} Pokémon")
        self.afficher_fiche_actuelle()

    def selectionner_pokemon(self, _event):
        selection = self.liste.curselection()
        if selection:
            self.index_actuel = selection[0]
            self.afficher_fiche_actuelle()

    def naviguer(self, direction):
        if not self.resultats:
            return

        self.index_actuel = (
            self.index_actuel + direction
        ) % len(self.resultats)

        self.liste.selection_clear(0, tk.END)
        self.liste.selection_set(self.index_actuel)
        self.liste.see(self.index_actuel)
        self.afficher_fiche_actuelle()

    def afficher_fiche_actuelle(self):
        for widget in self.details.winfo_children():
            widget.destroy()

        if not self.resultats:
            tk.Label(
                self.details, text="Aucun Pokémon trouvé",
                font=("Segoe UI", 16), fg=MUTED, bg=WHITE
            ).pack(pady=60)
            self.position.config(text="0 / 0")
            return

        pokemon = self.resultats[self.index_actuel]
        self.position.config(
            text=f"{self.index_actuel + 1} / {len(self.resultats)}"
        )

        tk.Label(
            self.details, text=f"#{pokemon['#']}",
            font=("Segoe UI", 12, "bold"), fg=BLUE, bg=WHITE
        ).pack(anchor="w")

        tk.Label(
            self.details, text=pokemon["Name"],
            font=("Segoe UI", 28, "bold"), fg=TEXT, bg=WHITE
        ).pack(anchor="w", pady=(0, 12))

        types = pokemon["Type 1"]
        if pokemon["Type 2"]:
            types += f"  •  {pokemon['Type 2']}"

        tk.Label(
            self.details, text=types, font=("Segoe UI", 12, "bold"),
            fg=BLUE, bg="#eff6ff", padx=12, pady=7
        ).pack(anchor="w", pady=(0, 24))

        tk.Label(
            self.details, text="STATISTIQUES",
            font=("Segoe UI", 10, "bold"), fg=MUTED, bg=WHITE
        ).pack(anchor="w", pady=(0, 10))

        statistiques = [
            ("PV", "HP"),
            ("Attaque", "Attack"),
            ("Défense", "Defense"),
            ("Att. spéciale", "Sp. Atk"),
            ("Déf. spéciale", "Sp. Def"),
            ("Vitesse", "Speed"),
        ]

        for nom, colonne in statistiques:
            ligne = tk.Frame(self.details, bg=WHITE)
            ligne.pack(fill="x", pady=6)

            valeur = int(pokemon[colonne])
            tk.Label(
                ligne, text=nom, width=16, anchor="w",
                font=("Segoe UI", 10), fg=TEXT, bg=WHITE
            ).pack(side="left")

            ttk.Progressbar(
                ligne,
                style="Stat.Horizontal.TProgressbar",
                maximum=255,
                value=valeur,
            ).pack(side="left", fill="x", expand=True, padx=12)

            tk.Label(
                ligne, text=str(valeur), width=4, anchor="e",
                font=("Segoe UI", 10, "bold"), fg=TEXT, bg=WHITE
            ).pack(side="right")

        legendaire = (
            "Oui" if pokemon["Legendary"].casefold() == "true" else "Non"
        )
        tk.Label(
            self.details,
            text=f"Génération : {pokemon['Generation']}    •    "
                 f"Légendaire : {legendaire}",
            font=("Segoe UI", 10), fg=MUTED, bg=WHITE
        ).pack(anchor="w", pady=(20, 0))


def main():
    fenetre = tk.Tk()

    try:
        pokedex = charger_pokedex()
        if not pokedex:
            raise ValueError("Le fichier CSV ne contient aucun Pokémon.")
    except (OSError, UnicodeError, csv.Error, ValueError) as erreur:
        messagebox.showerror(
            "Impossible d'ouvrir le Pokédex",
            str(erreur),
            parent=fenetre,
        )
        fenetre.destroy()
        return

    PokedexApp(fenetre, pokedex)
    fenetre.mainloop()


if __name__ == "__main__":
    main()