"""
Application Pokédex — Niveau 1.

L'utilisateur saisit les 6 statistiques de base d'un Pokémon (ou charge celles d'un
Pokémon existant par son nom), et le Pokédex affiche :
    - le GROUPE (cluster K-Means) auquel il appartient + son interprétation ;
    - la PROBABILITÉ qu'il soit légendaire ;
    - ses 3 TYPES les plus probables.

Lancement :  python app_pokedex_niveau1.py
(Interface tkinter, incluse dans Python : rien à installer en plus de requirements.txt.)
"""

import tkinter as tk
from tkinter import messagebox, ttk

from pokedex_ml import STATS, STATS_FR, PokedexIA


class AppPokedex:
    """Fenêtre principale : formulaire de saisie à gauche, résultats à droite."""

    def __init__(self, fenetre, ia):
        self.ia = ia
        fenetre.title("Pokédex IA — Niveau 1")
        fenetre.resizable(False, False)

        cadre = ttk.Frame(fenetre, padding=15)
        cadre.grid()

        # --- Zone de saisie ------------------------------------------------
        saisie = ttk.LabelFrame(cadre, text="Statistiques du Pokémon", padding=10)
        saisie.grid(row=0, column=0, sticky="n", padx=(0, 15))

        ttk.Label(saisie, text="Charger un Pokémon connu :").grid(row=0, column=0, columnspan=2, sticky="w")
        self.nom = tk.StringVar()
        # Liste déroulante avec tous les noms du CSV (on peut aussi taper un nom).
        ttk.Combobox(saisie, textvariable=self.nom, values=sorted(ia.df["Name"]), width=22).grid(
            row=1, column=0, sticky="w"
        )
        ttk.Button(saisie, text="Charger", command=self.charger).grid(row=1, column=1, padx=5)

        ttk.Separator(saisie).grid(row=2, column=0, columnspan=2, sticky="ew", pady=8)

        # Un champ de saisie par statistique, pré-rempli avec une valeur moyenne.
        self.champs = {}
        for i, stat in enumerate(STATS):
            ttk.Label(saisie, text=STATS_FR[stat]).grid(row=3 + i, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=str(round(ia.df[stat].mean())))
            ttk.Entry(saisie, textvariable=var, width=8).grid(row=3 + i, column=1, sticky="w")
            self.champs[stat] = var

        ttk.Button(saisie, text="Analyser", command=self.analyser).grid(
            row=3 + len(STATS), column=0, columnspan=2, pady=(10, 0), sticky="ew"
        )

        # --- Zone de résultats ---------------------------------------------
        resultats = ttk.LabelFrame(cadre, text="Analyse du Pokédex", padding=10)
        resultats.grid(row=0, column=1, sticky="n")

        self.texte = tk.Text(resultats, width=58, height=18, wrap="word", font=("Segoe UI", 10))
        self.texte.grid()
        self.texte.insert("1.0", "Saisis des statistiques puis clique sur « Analyser ».")
        self.texte.configure(state="disabled")

        # Info sur les modèles utilisés (transparence pour la démo).
        ttk.Label(
            cadre,
            text=(f"Modèles : légendaire = {ia.noms_modeles['Legendary']} · "
                  f"type = {ia.noms_modeles['Type 1']} · groupes = K-Means (k={ia.clustering.k})"),
            foreground="gray",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))

    def charger(self):
        """Remplit les champs avec les stats du Pokémon choisi dans le CSV."""
        stats = self.ia.stats_de(self.nom.get())
        if stats is None:
            messagebox.showwarning("Introuvable", f"« {self.nom.get()} » n'est pas dans le Pokédex.")
            return
        for stat, valeur in zip(STATS, stats):
            self.champs[stat].set(str(int(valeur)))
        self.analyser()

    def lire_stats(self):
        """Lit et vérifie les 6 champs ; renvoie None (avec un message) si une saisie est invalide."""
        valeurs = []
        for stat in STATS:
            try:
                v = float(self.champs[stat].get().replace(",", "."))
            except ValueError:
                messagebox.showerror("Saisie invalide", f"{STATS_FR[stat]} doit être un nombre.")
                return None
            if not 1 <= v <= 300:  # bornes larges autour des valeurs réelles (1 à 255)
                messagebox.showerror("Saisie invalide", f"{STATS_FR[stat]} doit être entre 1 et 300.")
                return None
            valeurs.append(v)
        return valeurs

    def analyser(self):
        """Lance les prédictions et affiche le résultat."""
        stats = self.lire_stats()
        if stats is None:
            return
        r = self.ia.predire(stats)

        lignes = [
            f"Total des stats : {int(sum(stats))}",
            "",
            f"GROUPE {r['groupe']} : {r['nom_groupe']}",
            f"   Pokémon typiques de ce groupe : {', '.join(r['exemples_groupe'])}",
            "",
            f"PROBABILITÉ D'ÊTRE LÉGENDAIRE : {r['proba_legendaire']:.1%}",
            f"   => {'Probablement LÉGENDAIRE' if r['proba_legendaire'] >= 0.5 else 'Probablement pas légendaire'}",
            "",
            "TYPES LES PLUS PROBABLES :",
        ]
        lignes += [f"   {i}. {t:<10} {p:.0%}" for i, (t, p) in enumerate(r["top3_types"], 1)]

        # Avertissement honnête si le Pokémon fait partie des données d'entraînement.
        nom = self.nom.get().strip()
        if nom and self.ia.stats_de(nom) == stats:
            vrai = self.ia.df[self.ia.df["Name"].str.lower() == nom.lower()].iloc[0]
            lignes += [
                "",
                f"Réalité ({vrai['Name']}) : type {vrai['Type 1']}"
                + (f"/{vrai['Type 2']}" if isinstance(vrai["Type 2"], str) else "")
                + f", {'légendaire' if vrai['Legendary'] else 'non légendaire'}.",
                "(Pokémon vu à l'entraînement : prédiction plus optimiste que pour un inconnu.)",
            ]

        self.texte.configure(state="normal")
        self.texte.delete("1.0", "end")
        self.texte.insert("1.0", "\n".join(lignes))
        self.texte.configure(state="disabled")


def main():
    # L'entraînement prend ~1-2 s (800 Pokémon) : on le fait au lancement,
    # pas besoin de sauvegarder les modèles dans un fichier.
    ia = PokedexIA()
    fenetre = tk.Tk()
    AppPokedex(fenetre, ia)
    fenetre.mainloop()


if __name__ == "__main__":
    main()
