import tkinter as tk
from tkinter import ttk

from pokedex_classifier import PokemonClassifier
from pokedex_data import STAT_COLUMNS


BG = "#f1f3f5"
WHITE = "#ffffff"
NAVY = "#173b3b"
TEXT = "#202b32"
MUTED = "#64747b"
GREEN = "#205c34"
RED = "#a32929"


class ClassificationView(tk.Frame):
    """Permet de tester le modèle sur des stats saisies par l'utilisateur."""

    def __init__(self, parent, pokedex):
        """Crée le classificateur, ses modèles évalués et le formulaire."""
        super().__init__(parent, bg=BG, padx=22, pady=20)
        self.classificateur = PokemonClassifier(pokedex)
        self.cible = tk.StringVar(value="Légendaire")
        self.champs_stats = {}
        self.evaluations = {}
        self.evaluation_labels = {}
        self.resultat_labels = {}
        self._creer_interface()
        self._afficher_evaluations()

    def _creer_interface(self):
        """Construit les commandes, la saisie des stats et la zone de résultat."""
        tk.Label(
            self, text="CLASSIFICATION PAR STATISTIQUES",
            font=("Segoe UI", 20, "bold"), fg=NAVY, bg=BG
        ).pack(anchor="w")
        tk.Label(
            self,
            text="Saisissez les six statistiques de base pour estimer une classe.",
            font=("Segoe UI", 10), fg=MUTED, bg=BG
        ).pack(anchor="w", pady=(4, 16))

        cadre_cible = tk.Frame(self, bg=WHITE, padx=16, pady=14)
        cadre_cible.pack(fill="x", pady=(0, 12))
        tk.Label(
            cadre_cible, text="À deviner",
            font=("Segoe UI", 10, "bold"), fg=TEXT, bg=WHITE
        ).pack(side="left", padx=(0, 12))
        choix_cible = ttk.Combobox(
            cadre_cible,
            textvariable=self.cible,
            values=list(PokemonClassifier.CIBLES),
            state="readonly",
            width=22,
        )
        choix_cible.pack(side="left")
        choix_cible.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._afficher_evaluations(),
        )
        tk.Button(
            cadre_cible,
            text="Changer le jeu d'entraînement",
            command=self._changer_jeu_entrainement,
            font=("Segoe UI", 9, "bold"), bg="#f8fbf6", fg=TEXT,
            relief="solid", bd=1, padx=10, pady=6, cursor="hand2",
        ).pack(side="left", padx=(12, 0))
        graine = self.classificateur.graine_aleatoire
        self.graine_label = tk.Label(
            cadre_cible,
            text=(
                f"Répartition {graine - self.classificateur.GRAINE_ALEATOIRE + 1} "
                f"(graine {graine})"
            ),
            font=("Segoe UI", 9), fg=MUTED, bg=WHITE,
        )
        self.graine_label.pack(side="right", padx=(8, 0))

        cadre_stats = tk.LabelFrame(
            self, text="Statistiques de base (0 à 255)",
            font=("Segoe UI", 10, "bold"), fg=NAVY, bg=WHITE,
            padx=18, pady=16
        )
        cadre_stats.pack(fill="x")
        for index, (nom, _colonne) in enumerate(STAT_COLUMNS):
            ligne, colonne = divmod(index, 2)
            tk.Label(
                cadre_stats, text=nom, width=14, anchor="w",
                font=("Segoe UI", 10), fg=TEXT, bg=WHITE
            ).grid(row=ligne, column=colonne * 2, sticky="w", padx=(0, 8), pady=8)
            variable = tk.StringVar()
            self.champs_stats[nom] = variable
            tk.Entry(
                cadre_stats, textvariable=variable, width=12,
                font=("Segoe UI", 11), justify="center",
            ).grid(row=ligne, column=colonne * 2 + 1, sticky="w", padx=(0, 24))

        self.onglets_modeles = ttk.Notebook(self)
        self.onglets_modeles.pack(fill="both", expand=True, pady=(14, 0))
        for nom_modele in PokemonClassifier.MODELES:
            page = tk.Frame(
                self.onglets_modeles, bg=WHITE, padx=16, pady=14
            )
            self.onglets_modeles.add(page, text=nom_modele)
            actions = tk.Frame(page, bg=WHITE)
            actions.pack(anchor="w", pady=(0, 8))
            tk.Button(
                actions, text="Expliquer le modèle",
                command=lambda modele=nom_modele: self._expliquer(modele),
                font=("Segoe UI", 9), bg="#f8fbf6", fg=TEXT,
                relief="solid", bd=1, padx=12, pady=6, cursor="hand2",
            ).pack(side="left", padx=(0, 8))
            tk.Button(
                actions, text="Voir les données train/test",
                command=lambda modele=nom_modele: self._afficher_decoupage(
                    modele
                ),
                font=("Segoe UI", 9), bg="#f8fbf6", fg=TEXT,
                relief="solid", bd=1, padx=12, pady=6, cursor="hand2",
            ).pack(side="left", padx=(0, 8))
            tk.Button(
                actions, text="Tester avec ce modèle",
                command=lambda modele=nom_modele: self._predire(modele),
                font=("Segoe UI", 9, "bold"), bg=NAVY, fg=WHITE,
                activebackground="#285b54", activeforeground=WHITE,
                relief="flat", padx=12, pady=7, cursor="hand2",
            ).pack(side="left")
            self.evaluation_labels[nom_modele] = tk.Label(
                page, text="", font=("Segoe UI", 9, "bold"),
                fg=MUTED, bg=WHITE, wraplength=760, justify="left",
            )
            self.evaluation_labels[nom_modele].pack(
                anchor="w", pady=(12, 4)
            )
            self.resultat_labels[nom_modele] = tk.Label(
                page, text="", font=("Segoe UI", 12, "bold"),
                fg=TEXT, bg=WHITE, wraplength=760, justify="left",
            )
            self.resultat_labels[nom_modele].pack(anchor="w")
        tk.Label(
            self,
            text=(
                "« Changer le jeu d'entraînement » tire une nouvelle "
                "répartition stratifiée : 80 % entraînement et 20 % test. "
                "Tous les modèles partagent les mêmes lignes. Le type prédit "
                "est le type principal (Type 1)."
            ),
            font=("Segoe UI", 9, "italic"), fg=MUTED, bg=BG,
            wraplength=760, justify="left",
        ).pack(anchor="w", pady=(10, 0))

    def _changer_jeu_entrainement(self):
        """Tire un nouveau partage puis réentraîne les modèles de la cible."""
        graine = self.classificateur.changer_decoupage()
        self.evaluations.clear()
        self._afficher_evaluations()
        numero_repartition = (
            graine - self.classificateur.GRAINE_ALEATOIRE + 1
        )
        self.graine_label.config(
            text=f"Répartition {numero_repartition} (graine {graine})"
        )

    def _afficher_evaluations(self):
        """Montre les métriques du test pour chaque modèle et la cible choisie."""
        cible = self.cible.get()
        for nom_modele in PokemonClassifier.MODELES:
            cle = (cible, nom_modele)
            if cle not in self.evaluations:
                self.evaluations[cle] = (
                    self.classificateur.entrainer_et_evaluer(
                        cible, nom_modele
                    )
                )
            evaluation = self.evaluations[cle]
            texte = (
                f"Test : exactitude {evaluation['exactitude']:.1%}, "
                f"F1 macro {evaluation['f1_macro']:.1%} "
                f"({evaluation['nombre_entrainement']} entraînement / "
                f"{evaluation['nombre_test']} test)"
            )
            if "precision_legendaire" in evaluation:
                texte += (
                    f" · précision légendaire "
                    f"{evaluation['precision_legendaire']:.1%}, rappel "
                    f"{evaluation['rappel_legendaire']:.1%}"
                )
            self.evaluation_labels[nom_modele].config(text=texte)
            self.resultat_labels[nom_modele].config(text="")

    def _predire(self, nom_modele):
        """Valide les saisies puis lance le modèle correspondant à son onglet."""
        statistiques = []
        for nom, _colonne in STAT_COLUMNS:
            texte = self.champs_stats[nom].get().strip()
            try:
                valeur = int(texte)
            except ValueError:
                self._afficher_message(
                    nom_modele, f"Entrez un nombre entier pour « {nom} ».",
                    erreur=True,
                )
                return
            if not 0 <= valeur <= 255:
                self._afficher_message(
                    nom_modele,
                    f"La valeur de « {nom} » doit être comprise entre 0 et 255.",
                    erreur=True,
                )
                return
            statistiques.append(valeur)

        cible = self.cible.get()
        try:
            prediction = self.classificateur.predire(
                statistiques, cible, nom_modele
            )
        except ValueError as erreur:
            self._afficher_message(nom_modele, str(erreur), erreur=True)
            return

        if cible == "Légendaire":
            message = f"Statut estimé : {prediction}"
        else:
            message = f"Type principal estimé : {prediction}"
        self._afficher_message(nom_modele, message)

    def _expliquer(self, nom_modele):
        """Ouvre une explication détaillée et un exemple de fonctionnement."""
        cible = self.cible.get()
        entrainement, test = self.classificateur.obtenir_decoupage(cible)
        texte = (
            f"Objectif choisi : {cible}\n"
            "Entrées du modèle : les six statistiques de base "
            "(PV, Attaque, Défense, Att. Spé., Déf. Spé. et Vitesse).\n"
            f"Découpage : {len(entrainement)} Pokémon pour l'entraînement "
            f"et {len(test)} pour le test. Tous les modèles utilisent "
            "exactement les mêmes lignes pour cette cible.\n\n"
            "Fonctionnement\n"
            f"{PokemonClassifier.MODELES[nom_modele]}\n\n"
            "Le jeu d'entraînement sert à construire le modèle. Le jeu de "
            "test reste à part pendant cet apprentissage et sert à mesurer "
            "ses prédictions sur des Pokémon qu'il n'a pas appris."
        )
        fenetre = tk.Toplevel(self)
        fenetre.title(f"Fonctionnement : {nom_modele}")
        fenetre.transient(self.winfo_toplevel())
        fenetre.geometry("700x420")
        fenetre.minsize(520, 320)

        zone = tk.Frame(fenetre, padx=14, pady=14)
        zone.pack(fill="both", expand=True)
        zone.grid_rowconfigure(0, weight=1)
        zone.grid_columnconfigure(0, weight=1)
        explication = tk.Text(
            zone, wrap="word", font=("Segoe UI", 10),
            padx=10, pady=8, relief="solid", bd=1,
        )
        explication.insert("1.0", texte)
        explication.configure(state="disabled")
        explication.grid(row=0, column=0, sticky="nsew")
        barre = ttk.Scrollbar(zone, orient="vertical", command=explication.yview)
        barre.grid(row=0, column=1, sticky="ns")
        explication.configure(yscrollcommand=barre.set)
        tk.Button(
            zone, text="Fermer", command=fenetre.destroy,
            padx=14, pady=6,
        ).grid(row=1, column=0, sticky="e", pady=(10, 0))

    def _afficher_decoupage(self, nom_modele):
        """Montre les lignes et les étiquettes des deux ensembles utilisés."""
        cible = self.cible.get()
        entrainement, test = self.classificateur.obtenir_decoupage(cible)
        fenetre = tk.Toplevel(self)
        fenetre.title(f"Données utilisées : {nom_modele}")
        fenetre.transient(self.winfo_toplevel())
        fenetre.geometry("1050x620")
        fenetre.minsize(800, 480)

        tk.Label(
            fenetre,
            text=(
                f"Modèle : {nom_modele}    •    Cible : {cible}    •    "
                f"Graine aléatoire : "
                f"{self.classificateur.graine_aleatoire}"
            ),
            font=("Segoe UI", 10, "bold"), anchor="w",
        ).pack(fill="x", padx=14, pady=(14, 4))
        tk.Label(
            fenetre,
            text=(
                "Seules les six statistiques sont fournies au modèle. "
                "La cible réelle sert d'étiquette; le test n'est pas utilisé "
                "pour entraîner le modèle. Les données sont communes aux "
                "deux modèles pour une même cible."
            ),
            font=("Segoe UI", 9), anchor="w", justify="left", wraplength=990,
        ).pack(fill="x", padx=14, pady=(0, 10))

        onglets_donnees = ttk.Notebook(fenetre)
        onglets_donnees.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        self._creer_table_donnees(
            onglets_donnees, "Entraînement", entrainement, cible
        )
        self._creer_table_donnees(
            onglets_donnees, "Test", test, cible
        )

    def _creer_table_donnees(self, onglets, nom_ensemble, pokemon, cible):
        """Ajoute un tableau consultable pour un ensemble de données."""
        page = tk.Frame(onglets, padx=8, pady=8)
        onglets.add(page, text=f"{nom_ensemble} ({len(pokemon)})")
        colonnes_stats = [colonne for _, colonne in STAT_COLUMNS]
        colonnes = ("numero", "nom", *colonnes_stats, "etiquette")
        entetes = (
            "#", "Pokémon", *(nom for nom, _ in STAT_COLUMNS),
            f"Cible réelle : {cible}",
        )
        largeurs = (55, 150, 65, 75, 75, 75, 75, 75, 150)

        tableau = ttk.Treeview(
            page, columns=colonnes, show="headings", height=16
        )
        for colonne, entete, largeur in zip(colonnes, entetes, largeurs):
            tableau.heading(colonne, text=entete)
            tableau.column(colonne, width=largeur, minwidth=50, anchor="center")

        barre_verticale = ttk.Scrollbar(
            page, orient="vertical", command=tableau.yview
        )
        barre_horizontale = ttk.Scrollbar(
            page, orient="horizontal", command=tableau.xview
        )
        tableau.configure(
            yscrollcommand=barre_verticale.set,
            xscrollcommand=barre_horizontale.set,
        )
        page.grid_rowconfigure(0, weight=1)
        page.grid_columnconfigure(0, weight=1)
        tableau.grid(row=0, column=0, sticky="nsew")
        barre_verticale.grid(row=0, column=1, sticky="ns")
        barre_horizontale.grid(row=1, column=0, sticky="ew")

        for individu in sorted(pokemon, key=lambda ligne: int(ligne["#"])):
            valeurs = (
                individu["#"],
                individu["Name"],
                *(individu[colonne] for colonne in colonnes_stats),
                self.classificateur.etiquette(individu, cible),
            )
            tableau.insert("", "end", values=valeurs)

    def _afficher_message(self, nom_modele, message, erreur=False):
        """Affiche le résultat ou une erreur sous l'onglet du modèle choisi."""
        self.resultat_labels[nom_modele].config(
            text=message,
            fg=RED if erreur else GREEN,
        )
