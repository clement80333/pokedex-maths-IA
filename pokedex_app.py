"""
POKÉDEX INTELLIGENT — application graphique qui regroupe les 3 niveaux du projet.

Lancement (depuis la racine du dépôt) :
    python pokedex_app.py

Onglets :
    1. Pokédex  : consulter les Pokémon, les filtrer / regrouper / trier par type (niveau 3)
    2. Stats    : saisir 6 statistiques -> groupe, probabilité légendaire, types (niveau 1)
    3. Image    : charger une image -> types prédits + Pokémon reconnu (niveaux 2 et 3)
    4. Caméra   : montrer un Pokémon à la webcam -> identification + fiche (niveau 3)

Les onglets Image et Caméra ont besoin des modèles entraînés par
niveau2_3/entrainer.py (fichiers niveau2_3/modeles/*.joblib) et de TensorFlow.
S'ils manquent, l'appli marche quand même, sans ces deux onglets.
"""

import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

RACINE = Path(__file__).resolve().parent
sys.path.insert(0, str(RACINE / "niveau1"))
sys.path.insert(0, str(RACINE / "niveau2_3"))

import pandas as pd  # noqa: E402

from pokedex_ml import STATS, STATS_FR, PokedexIA  # noqa: E402  (niveau 1)

try:  # Pillow sert à afficher les images ; sans lui, pas d'images (mais le reste marche)
    from PIL import Image, ImageTk
except ImportError:
    Image = ImageTk = None

# ---------------------------------------------------------------------------
# Apparence "Pokédex"
# ---------------------------------------------------------------------------
ROUGE, ROUGE_FONCE = "#DC0A2D", "#8B0000"
ECRAN, ECRAN_TEXTE = "#98CB98", "#0F380F"   # écran vert façon console portable
GRIS = "#DEDEDE"
POLICE = ("Segoe UI", 10)
POLICE_TITRE = ("Segoe UI", 16, "bold")

COULEURS_TYPES = {
    "Normal": "#A8A77A", "Fire": "#EE8130", "Water": "#6390F0", "Electric": "#F7D02C",
    "Grass": "#7AC74C", "Ice": "#96D9D6", "Fighting": "#C22E28", "Poison": "#A33EA1",
    "Ground": "#E2BF65", "Flying": "#A98FF3", "Psychic": "#F95587", "Bug": "#A6B91A",
    "Rock": "#B6A136", "Ghost": "#735797", "Dragon": "#6F35FC", "Dark": "#705746",
    "Steel": "#B7B7CE", "Fairy": "#D685AD",
}
TYPES_FR = {
    "Normal": "Normal", "Fire": "Feu", "Water": "Eau", "Electric": "Électrik", "Grass": "Plante",
    "Ice": "Glace", "Fighting": "Combat", "Poison": "Poison", "Ground": "Sol", "Flying": "Vol",
    "Psychic": "Psy", "Bug": "Insecte", "Rock": "Roche", "Ghost": "Spectre", "Dragon": "Dragon",
    "Dark": "Ténèbres", "Steel": "Acier", "Fairy": "Fée",
}
TAILLE_ECRAN = 300  # côté (pixels) de l'image affichée sur l'écran de gauche


def nom_fichier(nom):
    """Même normalisation que niveau2_3/images_ml.py (sans importer TensorFlow)."""
    import unicodedata
    nom = str(nom).replace("♀", "f").replace("♂", "m")
    nom = unicodedata.normalize("NFKD", nom).encode("ascii", "ignore").decode()
    return "".join(c for c in nom.lower() if c.isalnum())


class PokedexApp:
    def __init__(self, fenetre):
        self.fenetre = fenetre
        fenetre.title("Pokédex intelligent")
        fenetre.configure(bg=ROUGE)
        fenetre.minsize(1050, 650)

        # Niveau 1 : entraîné au lancement (~2 s)
        self.ia = PokedexIA()
        self.df = self.ia.df.copy()
        self.df["Groupe"] = self.ia.clustering.labels

        # Niveaux 2-3 : chargés en arrière-plan (TensorFlow est long à démarrer)
        self.reco = None
        self.erreur_reco = "chargement des modèles d'images..."
        threading.Thread(target=self._charger_reco, daemon=True).start()

        self.camera = None
        self._image_ecran = None   # garder une référence, sinon Tk efface l'image
        self._derniere_image = None

        self._construire_interface()
        self._remplir_liste()
        self.fenetre.after(500, self._verifier_reco)

    # ------------------------------------------------------------------ interface
    def _construire_interface(self):
        # Bandeau du haut : grosse lentille bleue + 3 voyants
        haut = tk.Canvas(self.fenetre, height=80, bg=ROUGE, highlightthickness=0)
        haut.pack(fill="x")
        haut.create_oval(15, 8, 79, 72, fill="white", outline=ROUGE_FONCE, width=3)
        haut.create_oval(22, 15, 72, 65, fill="#29A8E0", outline="")
        haut.create_oval(32, 22, 47, 37, fill="#B3E5FC", outline="")
        for i, c in enumerate(("#FF3B3B", "#FFD93B", "#4CD964")):
            haut.create_oval(95 + 28 * i, 12, 113 + 28 * i, 30, fill=c, outline=ROUGE_FONCE, width=2)
        haut.create_text(200, 52, text="POKÉDEX", anchor="w", fill="white", font=("Segoe UI", 24, "bold"))
        self.statut = haut.create_text(1030, 52, text="", anchor="e", fill="white", font=POLICE)
        self.canvas_haut = haut
        haut.bind("<Configure>", lambda e: haut.coords(self.statut, e.width - 15, 52))

        corps = tk.Frame(self.fenetre, bg=ROUGE)
        corps.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        # Écran de gauche (image + nom)
        gauche = tk.Frame(corps, bg=GRIS, bd=4, relief="ridge")
        gauche.pack(side="left", fill="y", padx=(0, 12))
        self.ecran = tk.Label(gauche, bg=ECRAN, width=TAILLE_ECRAN, height=TAILLE_ECRAN,
                              text="", fg=ECRAN_TEXTE, font=POLICE_TITRE, compound="center")
        self.ecran.pack(padx=10, pady=10)
        self.legende_ecran = tk.Label(gauche, text="", bg=GRIS, font=("Segoe UI", 12, "bold"),
                                      wraplength=TAILLE_ECRAN)
        self.legende_ecran.pack(pady=(0, 6))
        self.types_ecran = tk.Frame(gauche, bg=GRIS)
        self.types_ecran.pack(pady=(0, 10))

        # Partie droite : onglets
        style = ttk.Style()
        style.configure("TNotebook", background=ROUGE)
        style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=(12, 4))
        self.onglets = ttk.Notebook(corps)
        self.onglets.pack(side="left", fill="both", expand=True)
        self._onglet_pokedex()
        self._onglet_stats()
        self._onglet_image()
        self._onglet_camera()

    def _afficher_ecran(self, image_pil=None, texte="", types=()):
        """Affiche une image (ou un texte) sur l'écran de gauche, + nom et badges de types."""
        if image_pil is not None and ImageTk is not None:
            img = image_pil.copy()
            img.thumbnail((TAILLE_ECRAN, TAILLE_ECRAN))
            self._image_ecran = ImageTk.PhotoImage(img)
            self.ecran.configure(image=self._image_ecran, text="", width=TAILLE_ECRAN, height=TAILLE_ECRAN)
        else:
            self._image_ecran = None
            self.ecran.configure(image="", text=texte or "?", width=24, height=12)
        self.legende_ecran.configure(text=texte if image_pil is not None else "")
        for w in self.types_ecran.winfo_children():
            w.destroy()
        for t in types:
            badge(self.types_ecran, t).pack(side="left", padx=3)

    # ----------------------------------------------------------- onglet 1 : Pokédex
    def _onglet_pokedex(self):
        f = ttk.Frame(self.onglets, padding=8)
        self.onglets.add(f, text="Pokédex")

        filtres = ttk.Frame(f)
        filtres.pack(fill="x")
        ttk.Label(filtres, text="Recherche :").pack(side="left")
        self.recherche = tk.StringVar()
        self.recherche.trace_add("write", lambda *a: self._remplir_liste())
        ttk.Entry(filtres, textvariable=self.recherche, width=16).pack(side="left", padx=(2, 10))

        ttk.Label(filtres, text="Type :").pack(side="left")
        self.filtre_type = tk.StringVar(value="Tous")
        types = ["Tous"] + sorted(COULEURS_TYPES, key=lambda t: TYPES_FR[t])
        cb = ttk.Combobox(filtres, textvariable=self.filtre_type, state="readonly", width=11,
                          values=[t if t == "Tous" else TYPES_FR[t] for t in types])
        cb.pack(side="left", padx=(2, 10))
        cb.bind("<<ComboboxSelected>>", lambda e: self._remplir_liste())

        ttk.Label(filtres, text="Trier par :").pack(side="left")
        self.tri = tk.StringVar(value="Numéro")
        cb2 = ttk.Combobox(filtres, textvariable=self.tri, state="readonly", width=9,
                           values=["Numéro", "Nom", "Total"] + [STATS_FR[s] for s in STATS])
        cb2.pack(side="left", padx=(2, 10))
        cb2.bind("<<ComboboxSelected>>", lambda e: self._remplir_liste())

        self.regrouper = tk.BooleanVar(value=False)
        ttk.Checkbutton(filtres, text="Regrouper par type", variable=self.regrouper,
                        command=self._remplir_liste).pack(side="left")
        self.leg_seul = tk.BooleanVar(value=False)
        ttk.Checkbutton(filtres, text="Légendaires", variable=self.leg_seul,
                        command=self._remplir_liste).pack(side="left", padx=6)

        bas = ttk.Frame(f)
        bas.pack(fill="both", expand=True, pady=(8, 0))

        colonnes = ("num", "types", "total")
        self.liste = ttk.Treeview(bas, columns=colonnes, show="tree headings", height=20)
        self.liste.heading("#0", text="Nom")
        self.liste.heading("num", text="N°")
        self.liste.heading("types", text="Types")
        self.liste.heading("total", text="Total")
        self.liste.column("#0", width=190)
        self.liste.column("num", width=45, anchor="e")
        self.liste.column("types", width=130)
        self.liste.column("total", width=50, anchor="e")
        barre = ttk.Scrollbar(bas, orient="vertical", command=self.liste.yview)
        self.liste.configure(yscrollcommand=barre.set)
        self.liste.pack(side="left", fill="y")
        barre.pack(side="left", fill="y")
        self.liste.bind("<<TreeviewSelect>>", self._selection)
        self.compteur = ttk.Label(f, text="")
        self.compteur.pack(anchor="w")

        self.fiche = tk.Frame(bas, bg="white", bd=2, relief="groove")
        self.fiche.pack(side="left", fill="both", expand=True, padx=(10, 0))

    def _pokemon_filtres(self):
        d = self.df
        if self.recherche.get().strip():
            d = d[d["Name"].str.lower().str.contains(self.recherche.get().strip().lower(), regex=False)]
        if self.filtre_type.get() != "Tous":
            t = next(k for k, v in TYPES_FR.items() if v == self.filtre_type.get())
            d = d[(d["Type 1"] == t) | (d["Type 2"] == t)]
        if self.leg_seul.get():
            d = d[d["Legendary"]]
        cle = {"Numéro": "#", "Nom": "Name", "Total": "Total"}.get(self.tri.get())
        if cle is None:
            cle = next(s for s in STATS if STATS_FR[s] == self.tri.get())
        return d.sort_values(cle, ascending=cle in ("#", "Name"))

    def _remplir_liste(self):
        self.liste.delete(*self.liste.get_children())
        d = self._pokemon_filtres()
        if self.regrouper.get():
            # Classement par type : un dossier par type (un Pokémon à 2 types apparaît 2 fois)
            for t in sorted(COULEURS_TYPES, key=lambda t: TYPES_FR[t]):
                sous = d[(d["Type 1"] == t) | (d["Type 2"] == t)]
                if sous.empty:
                    continue
                parent = self.liste.insert("", "end", text=f"{TYPES_FR[t]} ({len(sous)})", open=False,
                                           tags=(t,))
                self.liste.tag_configure(t, background=COULEURS_TYPES[t])
                for i, r in sous.iterrows():
                    self._inserer(parent, i, r)
        else:
            for i, r in d.iterrows():
                self._inserer("", i, r)
        self.compteur.configure(text=f"{len(d)} Pokémon affichés")

    def _inserer(self, parent, index, r):
        types = TYPES_FR[r["Type 1"]] + (f" / {TYPES_FR[r['Type 2']]}" if isinstance(r["Type 2"], str) else "")
        nom = ("★ " if r["Legendary"] else "") + r["Name"]
        self.liste.insert(parent, "end", iid=f"{parent}_{index}", text=nom,
                          values=(r["#"], types, r["Total"]))

    def _selection(self, _event=None):
        sel = self.liste.selection()
        if not sel or "_" not in sel[0] or not sel[0].split("_")[-1].isdigit():
            return
        self.afficher_fiche(int(sel[0].split("_")[-1]))

    def afficher_fiche(self, index):
        """Affiche toutes les infos d'un Pokémon (index = ligne du DataFrame)."""
        r = self.df.loc[index]
        for w in self.fiche.winfo_children():
            w.destroy()
        types = [r["Type 1"]] + ([r["Type 2"]] if isinstance(r["Type 2"], str) else [])

        tk.Label(self.fiche, text=f"#{r['#']:03d}  {r['Name']}", bg="white",
                 font=POLICE_TITRE).pack(anchor="w", padx=10, pady=(8, 2))
        ligne = tk.Frame(self.fiche, bg="white")
        ligne.pack(anchor="w", padx=10)
        for t in types:
            badge(ligne, t).pack(side="left", padx=(0, 4))
        infos = (f"Génération {r['Generation']}  ·  "
                 + ("LÉGENDAIRE ★" if r["Legendary"] else "Non légendaire")
                 + f"\nGroupe (clustering) : {self.ia.clustering.noms[r['Groupe']]}")
        tk.Label(self.fiche, text=infos, bg="white", justify="left", font=POLICE).pack(anchor="w", padx=10, pady=6)

        # Barres de statistiques (échelle : 0 à 255, le max du jeu de données)
        c = tk.Canvas(self.fiche, width=380, height=26 * len(STATS) + 30, bg="white", highlightthickness=0)
        c.pack(anchor="w", padx=10)
        for i, s in enumerate(STATS):
            y = 10 + 26 * i
            v = int(r[s])
            couleur = "#4CD964" if v >= 100 else "#FFD93B" if v >= 60 else "#FF6B6B"
            c.create_text(0, y + 8, text=STATS_FR[s], anchor="w", font=POLICE)
            c.create_rectangle(80, y, 80 + 250, y + 16, fill="#EEEEEE", outline="")
            c.create_rectangle(80, y, 80 + 250 * v / 255, y + 16, fill=couleur, outline="")
            c.create_text(340, y + 8, text=str(v), anchor="w", font=POLICE)
        c.create_text(0, 10 + 26 * len(STATS) + 8, text=f"Total : {int(r['Total'])}", anchor="w",
                      font=("Segoe UI", 10, "bold"))

        self._afficher_ecran(self._miniature(r["Name"]), r["Name"], types)

    def _miniature(self, nom):
        """Image du Pokémon si disponible (Pokémon de 1re génération, créée par entrainer.py)."""
        f = RACINE / "niveau2_3" / "miniatures" / f"{nom_fichier(nom)}.png"
        if Image is not None and f.exists():
            return Image.open(f)
        return None

    def montrer_pokemon(self, nom):
        """Ouvre l'onglet Pokédex sur la fiche du Pokémon `nom`."""
        lignes = self.df.index[self.df["Name"] == nom]
        if len(lignes):
            self.onglets.select(0)
            self.afficher_fiche(lignes[0])

    # ------------------------------------------------------ onglet 2 : stats (niv. 1)
    def _onglet_stats(self):
        f = ttk.Frame(self.onglets, padding=12)
        self.onglets.add(f, text="Stats → IA (niv. 1)")
        ttk.Label(f, text="Saisis les statistiques de base d'un Pokémon (connu ou inventé) :",
                  font=("Segoe UI", 10, "bold")).grid(row=0, column=0, columnspan=3, sticky="w")
        self.champs = {}
        for i, s in enumerate(STATS):
            ttk.Label(f, text=STATS_FR[s]).grid(row=1 + i, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=str(round(self.df[s].mean())))
            ttk.Spinbox(f, from_=1, to=255, textvariable=var, width=6).grid(row=1 + i, column=1, sticky="w")
            self.champs[s] = var
        ttk.Button(f, text="Analyser", command=self._analyser_stats).grid(row=8, column=0, columnspan=2,
                                                                          sticky="ew", pady=8)
        self.res_stats = tk.Text(f, width=60, height=16, font=POLICE, wrap="word", bg=ECRAN,
                                 fg=ECRAN_TEXTE, relief="flat")
        self.res_stats.grid(row=1, column=2, rowspan=9, padx=(20, 0), sticky="n")
        ttk.Label(f, foreground="gray", text=(
            f"Modèles : légendaire = {self.ia.noms_modeles['Legendary']}, "
            f"type = {self.ia.noms_modeles['Type 1']}, groupes = K-Means (k={self.ia.clustering.k})"
        )).grid(row=10, column=0, columnspan=3, sticky="w", pady=(10, 0))

    def _analyser_stats(self):
        try:
            stats = [float(self.champs[s].get().replace(",", ".")) for s in STATS]
        except ValueError:
            messagebox.showerror("Saisie invalide", "Les 6 statistiques doivent être des nombres.")
            return
        if not all(1 <= v <= 300 for v in stats):
            messagebox.showerror("Saisie invalide", "Les statistiques doivent être entre 1 et 300.")
            return
        r = self.ia.predire(stats)
        txt = [
            f"Total des stats : {int(sum(stats))}", "",
            f"GROUPE : {r['nom_groupe']}",
            f"   Pokémon typiques : {', '.join(r['exemples_groupe'])}", "",
            f"PROBABILITÉ D'ÊTRE LÉGENDAIRE : {r['proba_legendaire']:.1%}", "",
            "TYPES LES PLUS PROBABLES (d'après les stats seules) :",
        ] + [f"   {i}. {TYPES_FR[t]:<10} {p:.0%}" for i, (t, p) in enumerate(r["top3_types"], 1)]
        ecrire(self.res_stats, "\n".join(txt))
        self._afficher_ecran(None, f"Groupe {r['groupe']}", [r["type_predit"]])

    # ---------------------------------------------------- onglet 3 : image (niv. 2-3)
    def _onglet_image(self):
        f = ttk.Frame(self.onglets, padding=12)
        self.onglets.add(f, text="Image (niv. 2)")
        ttk.Button(f, text="Choisir une image…", command=self._choisir_image).pack(anchor="w")
        self.res_image = tk.Text(f, width=70, height=20, font=POLICE, bg=ECRAN, fg=ECRAN_TEXTE, relief="flat")
        self.res_image.pack(fill="both", expand=True, pady=8)
        self.bouton_fiche_img = ttk.Button(f, text="Voir la fiche du Pokémon reconnu", state="disabled")
        self.bouton_fiche_img.pack(anchor="w")

    def _choisir_image(self):
        if not self._reco_prete():
            return
        chemin = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.gif *.webp")])
        if chemin:
            self._analyser_image(Image.open(chemin), self.res_image, self.bouton_fiche_img)

    def _analyser_image(self, image_pil, zone_texte, bouton_fiche):
        """Analyse une image (niveau 2 : types ; niveau 3 : identification) et affiche le résultat."""
        r = self.reco.analyser(image_pil)
        nom, p = r["top5"][0]
        types_pred = [t for t, _ in r["types"]]
        sur = p >= 0.5
        lignes = [
            "TYPES PRÉDITS À PARTIR DE L'IMAGE (niveau 2) :",
            "   " + " / ".join(f"{TYPES_FR[t]} ({q:.0%})" for t, q in r["types"]),
            "   Autres types envisagés : " + ", ".join(
                f"{TYPES_FR[t]} {q:.0%}" for t, q in r["probas_types"][len(r['types']):len(r['types']) + 3]),
            "",
            "POKÉMON RECONNU (niveau 3) :",
            f"   {nom}  —  confiance {p:.0%}" + ("" if sur else "   (peu sûr !)"),
            "   Autres possibilités : " + ", ".join(f"{n} {q:.0%}" for n, q in r["top5"][1:]),
        ]
        lignes_df = self.df[self.df["Name"] == nom]
        if len(lignes_df):
            v = lignes_df.iloc[0]
            vrais = [v["Type 1"]] + ([v["Type 2"]] if isinstance(v["Type 2"], str) else [])
            lignes += ["", f"D'après le Pokédex, {nom} est de type " + " / ".join(TYPES_FR[t] for t in vrais)
                       + (", légendaire." if v["Legendary"] else ", non légendaire.")]
        ecrire(zone_texte, "\n".join(lignes))
        self._afficher_ecran(image_pil, f"{nom} ?" if not sur else nom, types_pred)
        bouton_fiche.configure(state="normal", command=lambda: self.montrer_pokemon(nom))

    # --------------------------------------------------- onglet 4 : caméra (niv. 3)
    def _onglet_camera(self):
        f = ttk.Frame(self.onglets, padding=12)
        self.onglets.add(f, text="Caméra (niv. 3)")
        boutons = ttk.Frame(f)
        boutons.pack(anchor="w")
        self.bouton_cam = ttk.Button(boutons, text="Allumer la caméra", command=self._basculer_camera)
        self.bouton_cam.pack(side="left")
        ttk.Button(boutons, text="Capturer et identifier", command=self._capturer).pack(side="left", padx=8)
        ttk.Label(f, foreground="gray", text=(
            "Montre une image de Pokémon (écran de téléphone, carte, figurine...) à la webcam, "
            "puis clique sur « Capturer ».")).pack(anchor="w", pady=4)
        self.res_cam = tk.Text(f, width=70, height=16, font=POLICE, bg=ECRAN, fg=ECRAN_TEXTE, relief="flat")
        self.res_cam.pack(fill="both", expand=True, pady=8)
        self.bouton_fiche_cam = ttk.Button(f, text="Voir la fiche du Pokémon reconnu", state="disabled")
        self.bouton_fiche_cam.pack(anchor="w")

    def _basculer_camera(self):
        if self.camera is not None:
            self._arreter_camera()
            return
        try:
            import cv2
        except ImportError:
            messagebox.showerror("Caméra", "OpenCV n'est pas installé : pip install opencv-python")
            return
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            self.camera = None
            messagebox.showerror("Caméra", "Aucune caméra trouvée (ou déjà utilisée par une autre appli).")
            return
        self.bouton_cam.configure(text="Éteindre la caméra")
        self._image_camera()

    def _image_camera(self):
        """Lit une image de la webcam et l'affiche sur l'écran, ~25 fois par seconde."""
        if self.camera is None:
            return
        import cv2
        ok, frame = self.camera.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV donne du BGR
            h, w = frame.shape[:2]
            cote = min(h, w)                                 # recadrage carré au centre
            frame = frame[(h - cote) // 2:(h + cote) // 2, (w - cote) // 2:(w + cote) // 2]
            self._derniere_image = Image.fromarray(frame)
            self._afficher_ecran(self._derniere_image, "Caméra")
        self.fenetre.after(40, self._image_camera)

    def _arreter_camera(self):
        if self.camera is not None:
            self.camera.release()
        self.camera = None
        self.bouton_cam.configure(text="Allumer la caméra")

    def _capturer(self):
        if not self._reco_prete():
            return
        if self._derniere_image is None:
            messagebox.showinfo("Caméra", "Allume d'abord la caméra.")
            return
        image = self._derniere_image.copy()
        self._arreter_camera()  # on fige l'image capturée
        self._analyser_image(image, self.res_cam, self.bouton_fiche_cam)

    # --------------------------------------------------------- modèles d'images
    def _charger_reco(self):
        try:
            from images_ml import ReconnaissanceImage
            self.reco = ReconnaissanceImage()
            self.erreur_reco = None
        except Exception as e:  # TensorFlow absent, modèles pas entraînés...
            self.erreur_reco = f"{e.__class__.__name__} : {e}"

    def _verifier_reco(self):
        if self.reco is not None:
            self.canvas_haut.itemconfigure(self.statut, text="Tous les modèles sont prêts")
        elif self.erreur_reco and not self.erreur_reco.startswith("chargement"):
            self.canvas_haut.itemconfigure(self.statut, text="Image/caméra indisponibles (voir onglet)")
            for z in (self.res_image, self.res_cam):
                ecrire(z, "Reconnaissance d'images indisponible :\n" + self.erreur_reco
                       + "\n\nVoir le README (installation de TensorFlow / entraînement).")
        else:
            self.canvas_haut.itemconfigure(self.statut, text="Chargement des modèles d'images…")
            self.fenetre.after(500, self._verifier_reco)

    def _reco_prete(self):
        if Image is None:
            messagebox.showerror("Pillow manquant", "pip install pillow")
            return False
        if self.reco is None:
            messagebox.showinfo("Patience", "Les modèles d'images ne sont pas disponibles :\n"
                                + (self.erreur_reco or ""))
            return False
        return True

    def fermer(self):
        self._arreter_camera()
        self.fenetre.destroy()


def badge(parent, type_):
    """Petite étiquette colorée pour un type."""
    return tk.Label(parent, text=TYPES_FR.get(type_, type_), bg=COULEURS_TYPES.get(type_, "gray"),
                    fg="white", font=("Segoe UI", 9, "bold"), padx=8, pady=1)


def ecrire(zone, texte):
    zone.configure(state="normal")
    zone.delete("1.0", "end")
    zone.insert("1.0", texte)
    zone.configure(state="disabled")


def main():
    fenetre = tk.Tk()
    app = PokedexApp(fenetre)
    fenetre.protocol("WM_DELETE_WINDOW", app.fermer)
    fenetre.mainloop()


if __name__ == "__main__":
    main()
