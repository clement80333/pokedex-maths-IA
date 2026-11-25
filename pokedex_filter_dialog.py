import tkinter as tk


BG = "#f1f3f5"
WHITE = "#ffffff"
NAVY = "#173b3b"
TEXT = "#202b32"
MUTED = "#64747b"
SCREEN = "#edf5e9"


class FilterDialog:
    """Affiche les filtres et transmet leurs changements à l'application.

    Les variables de filtre restent gérées par l'application principale :
    cette classe ne fait que les présenter dans une fenêtre séparée.
    """

    def __init__(
        self, parent, filtres_types, filtres_generations,
        filtres_legendaire, on_change, on_clear,
    ):
        """Mémorise les filtres existants et les actions à appeler.

        ``on_change`` recalcule la liste après un changement de case.
        ``on_clear`` décoche les filtres et actualise cette liste.
        """
        self.parent = parent
        self.filtres_types = filtres_types
        self.filtres_generations = filtres_generations
        self.filtres_legendaire = filtres_legendaire
        self.on_change = on_change
        self.on_clear = on_clear
        self.fenetre = None

    def ouvrir(self):
        """Ouvre la fenêtre des filtres, ou ramène au premier plan l'existante."""
        if self.fenetre is not None and self.fenetre.winfo_exists():
            self.fenetre.deiconify()
            self.fenetre.lift()
            return

        fenetre = tk.Toplevel(self.parent)
        self.fenetre = fenetre
        fenetre.title("Filtrer le Pokédex")
        fenetre.transient(self.parent)
        fenetre.resizable(False, False)
        fenetre.configure(bg=BG)
        fenetre.geometry("720x430")
        fenetre.protocol("WM_DELETE_WINDOW", self.fermer)

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
            # Répartit les types sur trois colonnes pour limiter la hauteur.
            tk.Checkbutton(
                cadre_types, text=type_pokemon, variable=variable,
                command=self.on_change, font=("Segoe UI", 9),
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
                variable=variable, command=self.on_change,
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
                command=self.on_change, font=("Segoe UI", 9),
                fg=TEXT, bg=WHITE, activebackground=WHITE,
                selectcolor=SCREEN, anchor="w",
            ).pack(anchor="w")

        actions = tk.Frame(fenetre, bg=BG)
        actions.pack(fill="x", padx=20, pady=16)
        tk.Button(
            actions, text="Effacer les filtres", command=self.on_clear,
            bg=WHITE, fg=TEXT, relief="solid", bd=1,
            padx=12, pady=6, cursor="hand2"
        ).pack(side="left")
        tk.Button(
            actions, text="Fermer", command=self.fermer,
            bg=NAVY, fg=WHITE, relief="flat",
            padx=16, pady=7, cursor="hand2"
        ).pack(side="right")

    def fermer(self):
        """Ferme la fenêtre si elle existe et libère sa référence."""
        if self.fenetre is not None:
            if self.fenetre.winfo_exists():
                self.fenetre.destroy()
            self.fenetre = None
