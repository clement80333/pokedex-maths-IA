import csv
import tkinter as tk
from tkinter import messagebox

from pokedex_data import PokedexData
from pokedex_ui import PokedexApp


def main():
    fenetre = tk.Tk()

    try:
        donnees = PokedexData()
        if not donnees.pokedex:
            raise ValueError("Le fichier CSV ne contient aucun Pokémon.")
    except (OSError, UnicodeError, csv.Error, ValueError) as erreur:
        messagebox.showerror(
            "Impossible d'ouvrir le Pokédex",
            str(erreur),
            parent=fenetre,
        )
        fenetre.destroy()
        return

    PokedexApp(fenetre, donnees)
    fenetre.mainloop()


if __name__ == "__main__":
    main()
