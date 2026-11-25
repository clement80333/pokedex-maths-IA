import json
import re
import unicodedata
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


class PokemonImageService:
    """Récupère les illustrations officielles et les conserve dans le cache."""

    API_URL = "https://pokeapi.co/api/v2/pokemon/"
    CACHE_PATH = Path.home() / ".cache" / "pokedex-maths-ia" / "images"

    def charger(self, nom):
        for slug in self._slugs(nom):
            chemin_cache = self.CACHE_PATH / f"{slug}.png"
            if chemin_cache.is_file():
                return chemin_cache.read_bytes()

            requete = Request(
                self.API_URL + quote(slug),
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
            officiel = (
                (sprites.get("other") or {}).get("official-artwork") or {}
            )
            url_image = officiel.get("front_default") or sprites.get(
                "front_default"
            )
            if not url_image:
                continue

            requete_image = Request(
                url_image, headers={"User-Agent": "PokedexMathsIA/1.0"}
            )
            with urlopen(requete_image, timeout=12) as reponse:
                contenu_image = reponse.read()
            self.CACHE_PATH.mkdir(parents=True, exist_ok=True)
            chemin_cache.write_bytes(contenu_image)
            return contenu_image
        return None

    @staticmethod
    def _slugs(nom):
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
