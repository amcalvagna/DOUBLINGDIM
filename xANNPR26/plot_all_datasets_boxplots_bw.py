#!/usr/bin/env python3
"""
Crea due immagini in bianco e nero, una per accuracy e una per loss,
confrontando tre dataset.

Ogni file .pt deve avere struttura simile a:
    data[fold]['coresets'][(selection_method, norm_method)]['accu' o 'loss'][test_key]

dove fold sono 0..5 e test_key sono stringhe tipo 'test 1.00%'.

Il grafico prodotto usa boxplot raggruppati:
- ogni gruppo sull'asse x è una percentuale di test;
- dentro ogni gruppo compaiono i tre dataset;
- ogni box riassume i 6 fold;
- la linea interna al box è la mediana;
- il box mostra il range interquartile Q1-Q3;
- i whisker sono forzati a min e max con whis=(0, 100);
- sopra i box viene anche tracciata la curva delle mediane, una per dataset.

Per evitare l'uso del colore, i dataset sono distinti con:
- diversi tratteggi/texture dei box;
- diversi stili di linea;
- diversi marker.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D


# Modifica qui se vuoi usare altri file o altre etichette.
DEFAULT_DATASETS = {
    "CIFAR10": "./save/LINEAR[512, 10]_CIFAR10_tests_60.pt",
    "WM38": "./save/LINEAR[512, 38]_WM38_tests_60.pt",
    "CIFAR100": "./save/LINEAR[512, 100]_CIFAR100_tests_60.pt",
}

# Chiave trovata nei tuoi file.
DEFAULT_CORESET_KEY = ("Wasserstein_fast", "Norm_Min")

# Stili in bianco e nero per distinguere i dataset.
# Aggiungi altri elementi se in futuro confronti più di tre dataset.
HATCHES = ["", "///", "\\\\"]
LINE_STYLES = ["-", "--", ":"]
MARKERS = ["o", "s", "^"]


def test_percentage(test_key: str) -> float:
    """Estrae la percentuale da una stringa tipo 'test 2.50%'."""
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", test_key)
    if match is None:
        raise ValueError(f"Non riesco a leggere la percentuale dalla chiave: {test_key!r}")
    return float(match.group(1))


def load_pt(path: str | Path) -> dict:
    """Carica un file .pt in modo compatibile con PyTorch recenti."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Per versioni più vecchie di PyTorch che non supportano weights_only.
        return torch.load(path, map_location="cpu")


def get_metric_matrix(
    data: dict,
    metric: str,
    coreset_key: Tuple[str, str] = DEFAULT_CORESET_KEY,
) -> Tuple[np.ndarray, List[str]]:
    """
    Restituisce una matrice shape = (n_fold, n_test_points).

    Righe: fold.
    Colonne: percentuali test ordinate crescenti.
    """
    folds = sorted(data.keys())
    if not folds:
        raise ValueError("Il dizionario dati non contiene fold.")

    first_fold = folds[0]
    metric_dict = data[first_fold]["coresets"][coreset_key][metric]
    test_keys = sorted(metric_dict.keys(), key=test_percentage)

    rows = []
    for fold in folds:
        fold_metric = data[fold]["coresets"][coreset_key][metric]
        rows.append([float(fold_metric[key]) for key in test_keys])

    return np.asarray(rows, dtype=float), test_keys


def grouped_boxplot_with_median_lines(
    datasets: Dict[str, dict],
    metric: str,
    output_path: str | Path,
    coreset_key: Tuple[str, str] = DEFAULT_CORESET_KEY,
) -> None:
    """Crea e salva un boxplot raggruppato in bianco e nero per accuracy oppure loss."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_names = list(datasets.keys())

    # Carica matrici metriche e verifica che tutti abbiano gli stessi punti x.
    matrices: Dict[str, np.ndarray] = {}
    reference_keys: List[str] | None = None

    for name, data in datasets.items():
        matrix, test_keys = get_metric_matrix(data, metric, coreset_key=coreset_key)
        matrices[name] = matrix
        if reference_keys is None:
            reference_keys = test_keys
        elif test_keys != reference_keys:
            raise ValueError(
                f"I punti test del dataset {name!r} non coincidono con quelli degli altri dataset."
            )

    assert reference_keys is not None
    x_labels = [f"{test_percentage(key):g}%" for key in reference_keys]
    x = np.arange(len(x_labels), dtype=float)

    n_datasets = len(dataset_names)
    group_width = 0.75
    box_width = group_width / max(n_datasets, 1) * 0.80
    offsets = np.linspace(-group_width / 2, group_width / 2, n_datasets)

    fig, ax = plt.subplots(figsize=(11, 6))

    legend_handles = []
    for i, (offset, name) in enumerate(zip(offsets, dataset_names)):
        matrix = matrices[name]
        positions = x + offset

        hatch = HATCHES[i % len(HATCHES)]
        line_style = LINE_STYLES[i % len(LINE_STYLES)]
        marker = MARKERS[i % len(MARKERS)]

        # Per ogni punto x, passiamo i 6 valori dei fold.
        values_per_test_point = [matrix[:, j] for j in range(matrix.shape[1])]

        ax.boxplot(
            values_per_test_point,
            positions=positions,
            widths=box_width,
            whis=(0, 100),          # whisker da minimo a massimo
            showfliers=False,       # con whisker min/max gli outlier non servono
            patch_artist=True,
            manage_ticks=False,
            boxprops={
                "facecolor": "white",
                "edgecolor": "black",
                "linewidth": 1.0,
                "hatch": hatch,
            },
            whiskerprops={"color": "black", "linewidth": 1.0},
            capprops={"color": "black", "linewidth": 1.0},
            medianprops={"color": "black", "linewidth": 1.4},
        )

        medians = np.median(matrix, axis=0)
        ax.plot(
            positions,
            medians,
            color="black",
            linestyle=line_style,
            marker=marker,
            linewidth=1.8,
            markersize=5,
            label=name,
        )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linestyle=line_style,
                marker=marker,
                linewidth=1.8,
                markersize=5,
                label=name,
            )
        )

    pretty_metric = "Accuracy" if metric == "accu" else "Loss"
    ylabel = "Accuracy (%)" if metric == "accu" else "Loss"

    ax.set_title(f"{pretty_metric}: Coreset Test Performance (data over 6 folds)")
    ax.set_xlabel("Coreset size as percentage of the original dataset")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)

    # Linee verticali tra gruppi di percentuali diverse.
    for sep in x[:-1] + 0.5:
        ax.axvline(
            sep,
            color="black",
            linestyle="--",
            linewidth=0.6,
            alpha=0.35,
            zorder=0,
        )

    ax.grid(True, axis="y", color="black", alpha=0.25, linewidth=0.6)
    ax.legend(handles=legend_handles, title="Dataset")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crea due immagini boxplot in bianco e nero, accuracy e loss, per tre dataset .pt."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./img"),
        help="Cartella in cui salvare le immagini.",
    )
    parser.add_argument(
        "--accuracy-name",
        default="accuracy_boxplot_all_datasets_bw.png",
        help="Nome file dell'immagine accuracy.",
    )
    parser.add_argument(
        "--loss-name",
        default="loss_boxplot_all_datasets_bw.png",
        help="Nome file dell'immagine loss.",
    )
    args = parser.parse_args()

    datasets = {name: load_pt(path) for name, path in DEFAULT_DATASETS.items()}

    grouped_boxplot_with_median_lines(
        datasets,
        metric="accu",
        output_path=args.output_dir / args.accuracy_name,
    )
    grouped_boxplot_with_median_lines(
        datasets,
        metric="loss",
        output_path=args.output_dir / args.loss_name,
    )

    print(f"Salvato: {args.output_dir / args.accuracy_name}")
    print(f"Salvato: {args.output_dir / args.loss_name}")


if __name__ == "__main__":
    main()
