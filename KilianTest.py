import base64
import csv
import json
import queue
import re
import threading
import tkinter as tk
import unicodedata
from pathlib import Path
from tkinter import messagebox, ttk
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

CSV_PATH = Path(__file__).with_name("pokemon.csv")
IMAGE_CACHE_PATH = (
    Path.home() / ".cache" / "pokedex-maths-ia" / "images"
)
POKEAPI_URL = "https://pokeapi.co/api/v2/pokemon/"

BG = "#f1f3f5"
WHITE = "#ffffff"
RED = "#d9363e"
DARK_RED = "#a51f2b"
NAVY = "#173b3b"
BLUE = "#27a9c2"
TEXT = "#202b32"
MUTED = "#64747b"
SCREEN = "#edf5e9"
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
TYPE_COLORS = {
    "bug": "#677a24",
    "dark": "#5b5365",
    "dragon": "#7560cf",
    "electric": "#8a6c00",
    "fairy": "#a3476b",
    "fighting": "#c45b42",
    "fire": "#e8783d",
    "flying": "#788fc7",
    "ghost": "#6d67a8",
    "grass": "#5b9e57",
    "ground": "#80602b",
    "ice": "#347983",
    "normal": "#626b67",
    "poison": "#9b64a4",
    "psychic": "#a23d5f",
    "rock": "#a08e54",
    "steel": "#738b9b",
    "water": "#4c85c7",
}


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


def calculer_moyennes_par_type(pokedex):
    types = {}
    for pokemon in pokedex:
        for colonne in ("Type 1", "Type 2"):
            type_pokemon = pokemon[colonne]
            if type_pokemon:
                types.setdefault(type_pokemon.casefold(), type_pokemon)

    moyennes_par_type = []
    for cle_type, nom_type in sorted(
        types.items(), key=lambda element: element[1].casefold()
    ):
        membres = [
            pokemon for pokemon in pokedex
            if cle_type in (
                pokemon["Type 1"].casefold(),
                pokemon["Type 2"].casefold(),
            )
        ]
        moyennes_par_type.append(
            (nom_type, *calculer_moyennes_statistiques(membres))
        )
    return moyennes_par_type


def filtrer_pokedex(pokedex, recherche, types, generations, legendaires):
    resultats = []
    for pokemon in pokedex:
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


def trier_pokedex(pokedex, critere, ordre):
    if critere == "Total stats":
        cle_tri = lambda pokemon: sum(
            int(pokemon[colonne]) for _, colonne in STAT_COLUMNS
        )
    else:
        colonne = SORT_COLUMNS[critere]
        cle_tri = lambda pokemon: int(pokemon[colonne])
    return sorted(pokedex, key=cle_tri, reverse=ordre == "Décroissant")


def slugs_pokemon(nom):
    nom = nom.strip().casefold()
    if not nom:
        return []
    aliases = {
        "farfetch'd": "farfetchd",
        "wormadam plant cloak": "wormadam-plant",
        "wormadam sandy cloak": "wormadam-sandy",
        "wormadam trash cloak": "wormadam-trash",
        "heat rotom": "rotom-heat",
        "wash rotom": "rotom-wash",
        "frost rotom": "rotom-frost",
        "fan rotom": "rotom-fan",
        "mow rotom": "rotom-mow",
        "basculin": "basculin-red-striped",
        "darmanitan standard mode": "darmanitan-standard",
        "darmanitan zen mode": "darmanitan-zen",
        "frillish": "frillish-male",
        "jellicent": "jellicent-male",
        "pyroar": "pyroar-male",
        "kyurem black kyurem": "kyurem-black",
        "kyurem white kyurem": "kyurem-white",
        "hoopa confined": "hoopa",
    }
    if nom in aliases:
        return [aliases[nom]]
    if nom == "nidoran♀":
        return ["nidoran-f"]
    if nom == "nidoran♂":
        return ["nidoran-m"]
    if nom == "deoxysattack forme":
        return ["deoxys-attack", "deoxys"]
    if nom == "zygarde half forme":
        return ["zygarde-50", "zygarde"]

    if nom.startswith("mega "):
        morceaux = nom[5:].split()
        suffixe = ""
        if morceaux and morceaux[-1] in ("x", "y"):
            suffixe = "-" + morceaux.pop()
        base = "-".join(morceaux)
        return [f"{base}-mega{suffixe}", base]
    if nom.startswith("primal "):
        base = re.sub(r"[^a-z0-9]+", "-", nom[7:]).strip("-")
        return [f"{base}-primal", base]

    if nom.endswith(" forme") or nom.endswith(" size"):
        nom = nom.rsplit(" ", 1)[0]
    nom = unicodedata.normalize("NFKD", nom)
    nom = nom.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", nom).strip("-")
    return [slug] if slug else []


def charger_image_pokemon(nom):
    for slug in slugs_pokemon(nom):
        chemin_cache = IMAGE_CACHE_PATH / f"{slug}.png"
        if chemin_cache.is_file():
            return chemin_cache.read_bytes()

        requete = Request(
            POKEAPI_URL + quote(slug),
            headers={"User-Agent": "PokedexMathsIA/1.0"},
        )
        try:
            with urlopen(requete, timeout=12) as reponse:
                pokemon = json.loads(reponse.read().decode("utf-8"))
        except HTTPError as erreur:
            if erreur.code == 404:
                continue
            raise

        sprites = pokemon.get("sprites") or {}
        officiel = (sprites.get("other") or {}).get("official-artwork") or {}
        url_image = officiel.get("front_default") or sprites.get("front_default")
        if not url_image:
            continue

        requete_image = Request(
            url_image, headers={"User-Agent": "PokedexMathsIA/1.0"}
        )
        with urlopen(requete_image, timeout=12) as reponse:
            contenu_image = reponse.read()
        IMAGE_CACHE_PATH.mkdir(parents=True, exist_ok=True)
        chemin_cache.write_bytes(contenu_image)
        return contenu_image
    return None


class PokedexApp:
    def __init__(self, fenetre, pokedex):
        self.fenetre = fenetre
        self.pokedex = pokedex
        self.resultats = []
        self.index_actuel = 0
        self._file_images = queue.Queue(maxsize=1)
        self._resultats_images = queue.Queue()
        self._jeton_image = 0
        self._image_actuelle = None
        self.image_label = None

        fenetre.title("Pokédex")
        fenetre.geometry("1100x740")
        fenetre.minsize(850, 600)
        fenetre.configure(bg=BG)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "TNotebook", background=RED, borderwidth=0, tabmargins=(0, 0, 0, 0)
        )
        style.configure(
            "TNotebook.Tab",
            background="#f5b5b7",
            foreground=DARK_RED,
            padding=(18, 10),
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", WHITE)],
            foreground=[("selected", DARK_RED)],
        )
        style.configure(
            "Stat.Horizontal.TProgressbar",
            troughcolor="#d6e3d2",
            background="#4b9b68",
            thickness=10,
        )
        style.configure(
            "Stats.Treeview",
            background=WHITE,
            fieldbackground=WHITE,
            foreground=TEXT,
            rowheight=32,
            font=("Segoe UI", 10),
            borderwidth=0,
        )
        style.configure(
            "Stats.Treeview.Heading",
            background=NAVY,
            foreground=WHITE,
            font=("Segoe UI", 9, "bold"),
            padding=(7, 9),
        )

        self._travailleur_images = threading.Thread(
            target=self._charger_images_en_arriere_plan, daemon=True
        )
        self._travailleur_images.start()
        self.creer_interface()
        self.actualiser_liste()
        self.fenetre.after(100, self._actualiser_image)

    def _demander_image(self, nom):
        self._jeton_image += 1
        self.image_label.config(
            image="", text="Chargement…", fg=MUTED
        )
        self._image_actuelle = None
        demande = (self._jeton_image, nom)
        try:
            self._file_images.put_nowait(demande)
        except queue.Full:
            try:
                self._file_images.get_nowait()
            except queue.Empty:
                pass
            self._file_images.put_nowait(demande)

    def _charger_images_en_arriere_plan(self):
        while True:
            jeton, nom = self._file_images.get()
            try:
                contenu = charger_image_pokemon(nom)
            except (OSError, ValueError) as erreur:
                self._resultats_images.put((jeton, None, str(erreur)))
            else:
                self._resultats_images.put((jeton, contenu, None))

    def _actualiser_image(self):
        while True:
            try:
                jeton, contenu, erreur = self._resultats_images.get_nowait()
            except queue.Empty:
                break

            if jeton != self._jeton_image or self.image_label is None:
                continue
            if erreur:
                self.image_label.config(
                    image="", text="Connexion indisponible",
                    fg=MUTED,
                )
                continue
            if contenu is None:
                self.image_label.config(
                    image="", text="Image indisponible",
                    fg=MUTED,
                )
                continue

            try:
                image = tk.PhotoImage(
                    master=self.fenetre,
                    data=base64.b64encode(contenu).decode("ascii"),
                )
            except tk.TclError:
                self.image_label.config(
                    image="", text="Format d’image non pris en charge",
                    fg=MUTED,
                )
                continue
            facteur = max(1, (max(image.width(), image.height()) + 119) // 120)
            if facteur > 1:
                image = image.subsample(facteur, facteur)
            self._image_actuelle = image
            self.image_label.config(image=image, text="")

        self.fenetre.after(100, self._actualiser_image)

    def creer_interface(self):
        boitier = tk.Frame(self.fenetre, bg=RED, padx=14, pady=14)
        boitier.pack(fill="both", expand=True, padx=18, pady=18)

        entete = tk.Frame(boitier, bg=RED, padx=10, pady=6)
        entete.pack(fill="x", pady=(0, 12))
        lentille = tk.Canvas(
            entete, width=58, height=58, bg=RED, highlightthickness=0
        )
        lentille.create_oval(
            3, 3, 55, 55, fill="#f3f5f4", outline=DARK_RED, width=2
        )
        lentille.create_oval(
            9, 9, 49, 49, fill=BLUE, outline="#16768c", width=2
        )
        lentille.create_oval(16, 15, 28, 27, fill="#b9f5ff", outline="")
        lentille.pack(side="left", padx=(0, 14))

        titres = tk.Frame(entete, bg=RED)
        titres.pack(side="left")
        tk.Label(
            titres, text="POKÉDEX", font=("Segoe UI", 24, "bold"),
            fg=WHITE, bg=RED
        ).pack(anchor="w")
        tk.Label(
            titres, text="Encyclopédie des Pokémon",
            font=("Segoe UI", 10), fg="#ffe3e3", bg=RED
        ).pack(anchor="w")
        self.compteur = tk.Label(
            entete, font=("Segoe UI", 11, "bold"), fg=WHITE, bg=RED
        )
        self.compteur.pack(side="right", padx=8)

        self.onglets = ttk.Notebook(boitier)
        self.onglets.pack(fill="both", expand=True)
        page_pokedex = tk.Frame(self.onglets, bg=BG, padx=14, pady=14)
        page_stats = tk.Frame(self.onglets, bg=BG, padx=22, pady=20)
        self.onglets.add(page_pokedex, text="  Pokédex  ")
        self.onglets.add(page_stats, text="  Moyennes par type  ")

        filtres = tk.Frame(page_pokedex, bg=WHITE, padx=18, pady=14)
        filtres.pack(fill="x", pady=(0, 12))
        tk.Label(
            filtres, text="Rechercher", font=("Segoe UI", 10, "bold"),
            fg=TEXT, bg=WHITE
        ).pack(side="left", padx=(0, 10))

        self.recherche = tk.StringVar()
        champ = tk.Entry(
            filtres, textvariable=self.recherche, font=("Segoe UI", 11),
            relief="solid", bd=1, highlightthickness=1,
            highlightbackground="#d6dfd8", highlightcolor=BLUE
        )
        champ.pack(side="left", fill="x", expand=True, ipady=6)
        self.recherche.trace_add("write", lambda *_: self.actualiser_liste())

        self.types_disponibles = sorted({
            pokemon[colonne]
            for pokemon in self.pokedex
            for colonne in ("Type 1", "Type 2")
            if pokemon[colonne]
        })
        self.generations_disponibles = sorted({
            int(pokemon["Generation"]) for pokemon in self.pokedex
        })
        self.filtres_types = {
            type_pokemon: tk.BooleanVar(value=False)
            for type_pokemon in self.types_disponibles
        }
        self.filtres_generations = {
            generation: tk.BooleanVar(value=False)
            for generation in self.generations_disponibles
        }
        self.filtres_legendaire = {
            est_legendaire: tk.BooleanVar(value=False)
            for est_legendaire in (True, False)
        }
        self.fenetre_filtres = None
        self.bouton_filtrer = tk.Button(
            filtres, text="Filtrer ▾", command=self.ouvrir_filtres,
            font=("Segoe UI", 10, "bold"), bg="#f8fbf6", fg=TEXT,
            activebackground="#e2ece0", activeforeground=TEXT,
            relief="solid", bd=1, padx=10, pady=6, cursor="hand2"
        )
        self.bouton_filtrer.pack(side="left", padx=(18, 0))

        tk.Label(
            filtres, text="Trier par", font=("Segoe UI", 10, "bold"),
            fg=TEXT, bg=WHITE
        ).pack(side="left", padx=(18, 8))
        self.tri_choisi = tk.StringVar(value="Numéro Pokédex")
        choix_tri = ttk.Combobox(
            filtres, textvariable=self.tri_choisi,
            values=list(SORT_COLUMNS) + ["Total stats"],
            state="readonly", width=13,
        )
        choix_tri.pack(side="left", ipady=4)
        choix_tri.bind(
            "<<ComboboxSelected>>", lambda _event: self.actualiser_liste()
        )

        self.ordre_tri = tk.StringVar(value="Croissant")
        choix_ordre = ttk.Combobox(
            filtres, textvariable=self.ordre_tri,
            values=("Croissant", "Décroissant"),
            state="readonly", width=12,
        )
        choix_ordre.pack(side="left", padx=(8, 0), ipady=4)
        choix_ordre.bind(
            "<<ComboboxSelected>>", lambda _event: self.actualiser_liste()
        )

        contenu = tk.Frame(page_pokedex, bg=BG)
        contenu.pack(fill="both", expand=True)
        contenu.grid_rowconfigure(0, weight=1)
        contenu.grid_columnconfigure(1, weight=1, uniform="panneaux")
        contenu.grid_columnconfigure(0, weight=0)

        panneau_liste = tk.Frame(
            contenu, bg=SCREEN, padx=14, pady=14, width=300,
            highlightbackground="#c8d7c3", highlightthickness=1
        )
        panneau_liste.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        panneau_liste.grid_propagate(False)
        tk.Label(
            panneau_liste, text="INDEX DES POKÉMON",
            font=("Segoe UI", 11, "bold"), fg=NAVY, bg=SCREEN
        ).pack(anchor="w", pady=(0, 10))

        cadre_liste = tk.Frame(panneau_liste, bg=SCREEN)
        cadre_liste.pack(fill="both", expand=True)
        barre = ttk.Scrollbar(cadre_liste)
        barre.pack(side="right", fill="y")
        self.liste = tk.Listbox(
            cadre_liste, font=("Segoe UI", 10),
            bg="#f8fbf6", fg=TEXT, selectbackground="#397653",
            selectforeground=WHITE, activestyle="none", relief="flat",
            highlightthickness=1, highlightbackground="#cad8c6", bd=0,
            yscrollcommand=barre.set,
        )
        self.liste.pack(side="left", fill="both", expand=True)
        barre.config(command=self.liste.yview)
        self.liste.bind("<<ListboxSelect>>", self.selectionner_pokemon)

        self.panneau_fiche = tk.Frame(
            contenu, bg=SCREEN, padx=26, pady=22,
            highlightbackground="#c8d7c3", highlightthickness=1
        )
        self.panneau_fiche.grid(row=0, column=1, sticky="nsew")
        navigation = tk.Frame(self.panneau_fiche, bg=SCREEN)
        navigation.pack(side="bottom", fill="x", pady=(18, 0))
        tk.Button(
            navigation, text="← Précédent", command=lambda: self.naviguer(-1),
            bg=NAVY, fg=WHITE, relief="flat", padx=14, pady=8,
            activebackground="#285b54", activeforeground=WHITE, cursor="hand2"
        ).pack(side="left")
        self.position = tk.Label(
            navigation, text="", font=("Segoe UI", 10), fg=MUTED, bg=SCREEN
        )
        self.position.pack(side="left", expand=True)
        tk.Button(
            navigation, text="Suivant →", command=lambda: self.naviguer(1),
            bg=NAVY, fg=WHITE, relief="flat", padx=14, pady=8,
            activebackground="#285b54", activeforeground=WHITE, cursor="hand2"
        ).pack(side="right")
        self.details = tk.Frame(self.panneau_fiche, bg=SCREEN)
        self.details.pack(fill="both", expand=True)
        self.creer_page_moyennes(page_stats)

    def ouvrir_filtres(self):
        if (
            self.fenetre_filtres is not None
            and self.fenetre_filtres.winfo_exists()
        ):
            self.fenetre_filtres.deiconify()
            self.fenetre_filtres.lift()
            return

        fenetre = tk.Toplevel(self.fenetre)
        self.fenetre_filtres = fenetre
        fenetre.title("Filtrer le Pokédex")
        fenetre.transient(self.fenetre)
        fenetre.resizable(False, False)
        fenetre.configure(bg=BG)
        fenetre.geometry("720x430")
        fenetre.protocol("WM_DELETE_WINDOW", self.fermer_filtres)

        tk.Label(
            fenetre, text="Filtrer les Pokémon",
            font=("Segoe UI", 17, "bold"), fg=NAVY, bg=BG
        ).pack(anchor="w", padx=20, pady=(18, 3))
        tk.Label(
            fenetre,
            text="Les choix d’une même catégorie sont combinés entre eux.",
            font=("Segoe UI", 9), fg=MUTED, bg=BG
        ).pack(anchor="w", padx=20, pady=(0, 14))

        contenu = tk.Frame(fenetre, bg=BG)
        contenu.pack(fill="both", expand=True, padx=20)
        cadre_types = tk.LabelFrame(
            contenu, text="Types", font=("Segoe UI", 10, "bold"),
            fg=NAVY, bg=WHITE, padx=10, pady=8
        )
        cadre_types.pack(side="left", fill="both", expand=True, padx=(0, 12))
        for index, (type_pokemon, variable) in enumerate(
            self.filtres_types.items()
        ):
            tk.Checkbutton(
                cadre_types, text=type_pokemon, variable=variable,
                command=self.actualiser_liste, font=("Segoe UI", 9),
                fg=TEXT, bg=WHITE, activebackground=WHITE,
                selectcolor=SCREEN, anchor="w",
            ).grid(
                row=index // 3, column=index % 3,
                sticky="w", padx=4, pady=2,
            )

        colonne_droite = tk.Frame(contenu, bg=BG, width=220)
        colonne_droite.pack(side="left", fill="both")
        colonne_droite.pack_propagate(False)
        cadre_generations = tk.LabelFrame(
            colonne_droite, text="Générations",
            font=("Segoe UI", 10, "bold"), fg=NAVY, bg=WHITE, padx=10, pady=6
        )
        cadre_generations.pack(fill="x", pady=(0, 12))
        for generation, variable in self.filtres_generations.items():
            tk.Checkbutton(
                cadre_generations, text=f"Génération {generation}",
                variable=variable, command=self.actualiser_liste,
                font=("Segoe UI", 9), fg=TEXT, bg=WHITE,
                activebackground=WHITE, selectcolor=SCREEN, anchor="w",
            ).pack(anchor="w")

        cadre_legendaire = tk.LabelFrame(
            colonne_droite, text="Légendaire",
            font=("Segoe UI", 10, "bold"), fg=NAVY, bg=WHITE, padx=10, pady=6
        )
        cadre_legendaire.pack(fill="x")
        for libelle, est_legendaire in (("Oui", True), ("Non", False)):
            tk.Checkbutton(
                cadre_legendaire, text=libelle,
                variable=self.filtres_legendaire[est_legendaire],
                command=self.actualiser_liste, font=("Segoe UI", 9),
                fg=TEXT, bg=WHITE, activebackground=WHITE,
                selectcolor=SCREEN, anchor="w",
            ).pack(anchor="w")

        actions = tk.Frame(fenetre, bg=BG)
        actions.pack(fill="x", padx=20, pady=16)
        tk.Button(
            actions, text="Effacer les filtres", command=self.effacer_filtres,
            bg=WHITE, fg=TEXT, relief="solid", bd=1,
            padx=12, pady=6, cursor="hand2"
        ).pack(side="left")
        tk.Button(
            actions, text="Fermer", command=self.fermer_filtres,
            bg=NAVY, fg=WHITE, relief="flat",
            padx=16, pady=7, cursor="hand2"
        ).pack(side="right")

    def fermer_filtres(self):
        if self.fenetre_filtres is not None:
            if self.fenetre_filtres.winfo_exists():
                self.fenetre_filtres.destroy()
            self.fenetre_filtres = None

    def effacer_filtres(self):
        for filtres in (
            self.filtres_types,
            self.filtres_generations,
            self.filtres_legendaire,
        ):
            for variable in filtres.values():
                variable.set(False)
        self.actualiser_liste()

    def creer_page_moyennes(self, parent):
        tk.Label(
            parent, text="LES TYPES EN CHIFFRES",
            font=("Segoe UI", 20, "bold"), fg=NAVY, bg=BG
        ).pack(anchor="w")
        tk.Label(
            parent,
            text="Statistiques de base moyennes des Pokémon de chaque type.",
            font=("Segoe UI", 10), fg=MUTED, bg=BG
        ).pack(anchor="w", pady=(4, 16))

        cadre = tk.Frame(parent, bg=WHITE, padx=10, pady=10)
        cadre.pack(fill="both", expand=True)
        colonnes = (
            "type", "nombre", "hp", "attack", "defense",
            "sp_atk", "sp_def", "speed", "total",
        )
        self.table_moyennes = ttk.Treeview(
            cadre, columns=colonnes, show="headings", style="Stats.Treeview"
        )
        entetes = (
            ("type", "Type / groupe", 105, "w"),
            ("nombre", "Nb.", 55, "center"),
            ("hp", "PV", 60, "center"),
            ("attack", "Attaque", 78, "center"),
            ("defense", "Défense", 78, "center"),
            ("sp_atk", "Att. Spé.", 75, "center"),
            ("sp_def", "Déf. Spé.", 75, "center"),
            ("speed", "Vitesse", 75, "center"),
            ("total", "Total stats", 90, "center"),
        )
        for colonne, titre, largeur, ancrage in entetes:
            self.table_moyennes.heading(colonne, text=titre)
            self.table_moyennes.column(
                colonne, width=largeur, minwidth=55, anchor=ancrage,
                stretch=colonne in ("type", "nombre"),
            )
        self.table_moyennes.tag_configure("pair", background="#f3f7f1")
        self.table_moyennes.tag_configure("impair", background=WHITE)
        self.table_moyennes.tag_configure(
            "legendaire", background="#fff0d6", foreground="#704800",
            font=("Segoe UI", 10, "bold"),
        )
        barre = ttk.Scrollbar(
            cadre, orient="vertical", command=self.table_moyennes.yview
        )
        self.table_moyennes.configure(yscrollcommand=barre.set)
        self.table_moyennes.pack(side="left", fill="both", expand=True)
        barre.pack(side="right", fill="y")
        self.remplir_moyennes()

        tk.Label(
            parent,
            text="Les doubles types comptent dans chaque type ; "
                 "la ligne « Légendaires » regroupe les Pokémon légendaires.",
            font=("Segoe UI", 9, "italic"), fg=MUTED, bg=BG
        ).pack(anchor="w", pady=(12, 0))

    def remplir_moyennes(self):
        for element in self.table_moyennes.get_children():
            self.table_moyennes.delete(element)

        for index, (type_pokemon, nombre, *moyennes) in enumerate(
            calculer_moyennes_par_type(self.pokedex)
        ):
            self.table_moyennes.insert(
                "", tk.END,
                values=(
                    type_pokemon,
                    nombre,
                    *(f"{moyenne:.1f}" for moyenne in moyennes),
                ),
                tags=("pair" if index % 2 == 0 else "impair",),
            )

        legendaire = [
            pokemon for pokemon in self.pokedex
            if pokemon["Legendary"].strip().casefold() == "true"
        ]
        if legendaire:
            nombre, *moyennes = calculer_moyennes_statistiques(legendaire)
            self.table_moyennes.insert(
                "", tk.END,
                values=(
                    "Légendaires",
                    nombre,
                    *(f"{moyenne:.1f}" for moyenne in moyennes),
                ),
                tags=("legendaire",),
            )

    def actualiser_liste(self):
        recherche = self.recherche.get().strip().casefold()
        types = {
            type_pokemon.casefold()
            for type_pokemon, variable in self.filtres_types.items()
            if variable.get()
        }
        generations = {
            generation
            for generation, variable in self.filtres_generations.items()
            if variable.get()
        }
        legendaires = {
            est_legendaire
            for est_legendaire, variable in self.filtres_legendaire.items()
            if variable.get()
        }
        filtres_actifs = sum(bool(choix) for choix in (
            types, generations, legendaires
        ))
        self.bouton_filtrer.config(
            text=(
                f"Filtrer ({filtres_actifs}) ▾"
                if filtres_actifs else "Filtrer ▾"
            )
        )
        pokemon_actuel = (
            self.resultats[self.index_actuel] if self.resultats else None
        )

        resultats = filtrer_pokedex(
            self.pokedex, recherche, types, generations, legendaires
        )
        self.resultats = trier_pokedex(
            resultats, self.tri_choisi.get(), self.ordre_tri.get()
        )

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

        self.liste.selection_clear(0, tk.END)
        if self.resultats:
            self.liste.selection_set(self.index_actuel)
            self.liste.see(self.index_actuel)

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
        self._jeton_image += 1
        self.image_label = None
        for widget in self.details.winfo_children():
            widget.destroy()

        if not self.resultats:
            tk.Label(
                self.details, text="Aucun Pokémon trouvé",
                font=("Segoe UI", 16), fg=MUTED, bg=SCREEN
            ).pack(pady=60)
            self.position.config(text="0 / 0")
            return

        pokemon = self.resultats[self.index_actuel]
        self.position.config(
            text=f"{self.index_actuel + 1} / {len(self.resultats)}"
        )

        entete = tk.Frame(self.details, bg=SCREEN)
        entete.pack(fill="x", pady=(0, 8))
        informations = tk.Frame(entete, bg=SCREEN)
        informations.pack(side="left", fill="x", expand=True)
        tk.Label(
            informations, text=f"#{pokemon['#']}",
            font=("Segoe UI", 10, "bold"), fg=NAVY, bg=SCREEN
        ).pack(anchor="w")

        tk.Label(
            informations, text=pokemon["Name"],
            font=("Segoe UI", 22, "bold"), fg=TEXT, bg=SCREEN
        ).pack(anchor="w", pady=(0, 4))

        types = [pokemon["Type 1"]]
        if pokemon["Type 2"]:
            types.append(pokemon["Type 2"])

        badges_types = tk.Frame(informations, bg=SCREEN)
        badges_types.pack(anchor="w")
        for type_pokemon in types:
            tk.Label(
                badges_types, text=type_pokemon,
                font=("Segoe UI", 10, "bold"), fg=WHITE,
                bg=TYPE_COLORS.get(type_pokemon.casefold(), NAVY),
                padx=10, pady=4
            ).pack(side="left", padx=(0, 6))

        cadre_image = tk.Frame(
            entete, bg="#f8fbf6", width=128, height=128,
            highlightbackground="#c8d7c3", highlightthickness=1,
        )
        cadre_image.pack(side="right", padx=(12, 0))
        cadre_image.pack_propagate(False)
        self.image_label = tk.Label(
            cadre_image, text="Chargement…", bg="#f8fbf6", fg=MUTED,
            font=("Segoe UI", 8), wraplength=112,
        )
        self.image_label.pack(fill="both", expand=True)
        self._demander_image(pokemon["Name"])

        tk.Label(
            self.details, text="STATISTIQUES",
            font=("Segoe UI", 10, "bold"), fg=MUTED, bg=SCREEN
        ).pack(anchor="w", pady=(0, 4))

        for nom, colonne in STAT_COLUMNS:
            ligne = tk.Frame(self.details, bg=SCREEN)
            ligne.pack(fill="x", pady=1)

            valeur = int(pokemon[colonne])
            tk.Label(
                ligne, text=nom, width=16, anchor="w",
                font=("Segoe UI", 9), fg=TEXT, bg=SCREEN
            ).pack(side="left")

            ttk.Progressbar(
                ligne,
                style="Stat.Horizontal.TProgressbar",
                maximum=255,
                value=valeur,
            ).pack(side="left", fill="x", expand=True, padx=12)

            tk.Label(
                ligne, text=str(valeur), width=4, anchor="e",
                font=("Segoe UI", 9, "bold"), fg=TEXT, bg=SCREEN
            ).pack(side="right")

        legendaire = (
            "Oui" if pokemon["Legendary"].strip().casefold() == "true" else "Non"
        )
        tk.Label(
            self.details,
            text=f"Génération : {pokemon['Generation']}    •    "
                 f"Légendaire : {legendaire}",
            font=("Segoe UI", 9), fg=MUTED, bg=SCREEN
        ).pack(anchor="w", pady=(8, 0))


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