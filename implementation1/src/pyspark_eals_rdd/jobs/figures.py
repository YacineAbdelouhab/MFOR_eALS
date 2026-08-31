"""Trace les deux figures du rapport a partir des JSON des experiences.

Aucun Spark n'intervient ici, le script relit seulement ce que xp1 et xp2 ont
ecrit. Il produit deux PDF vectoriels prets a inclure dans le document.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COULEURS = {
    "K=8": "#4C72B0",
    "K=16": "#DD8452",
    "K=32": "#55A868",
    "aleatoire": "#4C72B0",
    "svd": "#C44E52",
}
INTITULES = {"aleatoire": "Aleatoire", "svd": "SVD randomisee"}


def parseur() -> argparse.ArgumentParser:
    resultat = argparse.ArgumentParser(description="Figures 5.1 et 5.2 du rapport")
    resultat.add_argument("--xp1", default="/data/xp1.json")
    resultat.add_argument("--xp2", default="/data/xp2.json")
    resultat.add_argument("--sortie", default="/data/figures")
    return resultat


def lire(chemin: str) -> dict:
    with open(chemin, "r", encoding="utf-8") as flux:
        return json.load(flux)


def habiller(axe, abscisse: str, ordonnee: str) -> None:
    axe.set_xlabel(abscisse)
    axe.set_ylabel(ordonnee)
    axe.grid(True, linewidth=0.4, alpha=0.4)
    axe.spines["top"].set_visible(False)
    axe.spines["right"].set_visible(False)
    axe.legend(frameon=False)


def figure_cout(xp1: dict, chemin: Path) -> None:
    """Figure 5.1, le cout au fil des iterations pour les trois rangs."""
    figure, axe = plt.subplots(figsize=(6.4, 4.0))
    for etiquette, historique in xp1["historiques"].items():
        iterations = [ligne["iteration"] for ligne in historique]
        cout = [ligne["objective"] for ligne in historique]
        axe.plot(
            iterations,
            cout,
            marker="o",
            markersize=3.5,
            linewidth=1.6,
            color=COULEURS.get(etiquette),
            label=etiquette.replace("K=", "K = "),
        )
    habiller(axe, "Nombre d'iterations", "Fonction de cout $L(P, Q)$")
    figure.tight_layout()
    figure.savefig(chemin, format="pdf")
    plt.close(figure)
    print(f"Figure 5.1 ecrite dans {chemin}")


def figure_initialisation(xp2: dict, chemin: Path) -> None:
    """Figure 5.2, le cout en fonction du temps cumule, initialisation comprise."""
    figure, axe = plt.subplots(figsize=(6.4, 4.0))
    for mode, historique in xp2["historiques"].items():
        cumul, temps = xp2["init_seconds"][mode], []
        for ligne in historique:
            cumul += ligne["seconds"]
            temps.append(cumul)
        axe.plot(
            temps,
            [ligne["objective"] for ligne in historique],
            marker="o",
            markersize=3.5,
            linewidth=1.6,
            color=COULEURS.get(mode),
            label=INTITULES.get(mode, mode),
        )
    habiller(axe, "Temps cumule, initialisation comprise (s)", "Fonction de cout $L(P, Q)$")
    figure.tight_layout()
    figure.savefig(chemin, format="pdf")
    plt.close(figure)
    print(f"Figure 5.2 ecrite dans {chemin}")


def main() -> None:
    args = parseur().parse_args()
    dossier = Path(args.sortie)
    dossier.mkdir(parents=True, exist_ok=True)
    figure_cout(lire(args.xp1), dossier / "figure-cout-par-rang.pdf")
    figure_initialisation(lire(args.xp2), dossier / "figure-cout-par-temps.pdf")


if __name__ == "__main__":
    main()
