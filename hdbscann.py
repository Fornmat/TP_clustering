#!/usr/bin/env python3
# hdbscan_plot.py
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# ------------------------------------------------------
# Funciones auxiliares
# ------------------------------------------------------
def load_arff(path):
    from scipy.io import arff
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = df[c].str.decode("utf-8")
            except Exception:
                pass
    return df


def eval_labels_no_noise(X, labels):
    mask = labels != -1
    if mask.sum() < 2:
        return dict(silhouette=np.nan, calinski=np.nan, davies=np.nan,
                    n_clusters=0, noise_ratio=1.0)
    labs = labels[mask]
    if len(np.unique(labs)) < 2:
        return dict(silhouette=np.nan, calinski=np.nan, davies=np.nan,
                    n_clusters=len(np.unique(labs)), noise_ratio=1.0 - mask.mean())
    Xn = X[mask]
    return dict(
        silhouette=silhouette_score(Xn, labs),
        calinski=calinski_harabasz_score(Xn, labs),
        davies=davies_bouldin_score(Xn, labs),
        n_clusters=len(np.unique(labs)),
        noise_ratio=float((labels == -1).mean()),
    )


def plot_clusters(X, labels, title="", save=None, show=True, noise_label=-1):
    plt.figure(figsize=(6, 6))
    uniq = np.unique(labels)
    for lab in uniq:
        m = labels == lab
        if lab == noise_label:
            plt.scatter(X[m, 0], X[m, 1], s=18, facecolors="none", edgecolors="grey", label="noise")
        else:
            plt.scatter(X[m, 0], X[m, 1], s=20, label=f"c{lab}")
    plt.legend(fontsize=8)
    plt.title(title)
    plt.tight_layout()
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        plt.savefig(save, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


# ------------------------------------------------------
# k-distance plot (como en DBSCAN)
# ------------------------------------------------------
def k_distance_plot(X, k, best_size=None, save=None, show=True):
    """Curva de distancias al k-ésimo vecino (como guía de densidad)."""
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    dists, _ = nn.kneighbors(X)
    kd = np.sort(dists[:, -1])
    plt.figure(figsize=(7, 4))
    plt.plot(kd, color="steelblue", label=f"Distancia al {k}º vecino")
    if best_size is not None:
        idx = len(kd) // 2
        plt.axvline(idx, color="red", linestyle="--", label=f"min_cluster_size óptimo ≈ {best_size}")
    plt.xlabel("Puntos ordenados")
    plt.ylabel(f"Distancia al {k}º vecino")
    plt.title(f"k-distance plot (k={k}) – guía de densidad")
    plt.legend()
    plt.tight_layout()
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        plt.savefig(save, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
    return kd


# ------------------------------------------------------
# Grid search HDBSCAN (igual estructura que DBSCAN)
# ------------------------------------------------------
def grid_search_hdbscan(X, min_cluster_sizes, min_samples=None):
    import hdbscan
    rows = []
    for mcs in min_cluster_sizes:
        model = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=min_samples)
        labels = model.fit_predict(X)
        m = eval_labels_no_noise(X, labels)
        m["min_cluster_size"] = mcs
        rows.append(m)
    df = pd.DataFrame(rows)
    df_sorted = df.sort_values(
        by=["silhouette", "calinski", "davies"],
        ascending=[False, True, False]
    )
    best_size = int(df_sorted.iloc[0]["min_cluster_size"])
    return best_size, df


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="HDBSCAN con análisis igual que DBSCAN")
    ap.add_argument("--arff", type=str, help="Ruta a .arff (x,y[,class])")
    ap.add_argument("--csv", type=str, help="Ruta a .csv")
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--save", type=str, help="PNG base de salida")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--min-cluster-size", type=int, dest="min_cluster_size", help="Tamaño mínimo de cluster (opcional)")
    ap.add_argument("--min-samples", type=int, default=None, dest="min_samples", help="Número mínimo de vecinos")
    args = ap.parse_args()

    try:
        import hdbscan
    except ImportError:
        raise SystemExit("Falta 'hdbscan'. Instala con: pip install hdbscan")

    if not args.arff and not args.csv:
        ap.error("Proporciona --arff o --csv")

    # -----------------------------
    # Cargar dataset
    # -----------------------------
    df = load_arff(args.arff) if args.arff else pd.read_csv(args.csv)
    X = df.iloc[:, :2].values
    if args.standardize:
        X = StandardScaler().fit_transform(X)

    # -----------------------------
    # k-distance plot de densidad
    # -----------------------------
    base = os.path.splitext(args.save)[0] if args.save else None
    kdist_png = f"{base}_kdist.png" if base else None
    clusters_png = f"{base}_hdbscan_clusters.png" if base else None
    comp_png = f"{base}_hdbscan_comparativa.png" if base else None

    print("\nMostrando curva de densidad (k-distance plot)...")
    k_distance_plot(X, k=args.min_samples or 5, best_size=args.min_cluster_size, save=kdist_png, show=not args.no_show)

    # -----------------------------
    # Búsqueda de parámetros
    # -----------------------------
    min_cluster_sizes = list(range(5, 51, 5))
    best_size, df = grid_search_hdbscan(X, min_cluster_sizes, min_samples=args.min_samples)

    print("\n=== HDBSCAN – Evaluación de parámetros ===")
    print(df.to_string(index=False, formatters={
        "min_cluster_size": lambda v: f"{v:>3}",
        "n_clusters": lambda v: f"{v:>3}",
        "silhouette": lambda v: f"{v:.3f}",
        "calinski": lambda v: f"{v:.1f}",
        "davies": lambda v: f"{v:.3f}",
        "noise_ratio": lambda v: f"{100*v:5.1f}%",
    }))

    size_to_use = args.min_cluster_size if args.min_cluster_size else best_size
    print(f"\nUsaremos min_cluster_size = {size_to_use} "
          f"({'fijado por el usuario' if args.min_cluster_size else 'óptimo por Silhouette'})")

    # -----------------------------
    # Entrenar modelo final
    # -----------------------------
    model = hdbscan.HDBSCAN(min_cluster_size=size_to_use, min_samples=args.min_samples)
    labels = model.fit_predict(X)
    m = eval_labels_no_noise(X, labels)
    title = (f"HDBSCAN (min_cluster_size={size_to_use}, min_samples={args.min_samples}) – "
             f"clusters={m['n_clusters']} – "
             f"Sil={m['silhouette']:.3f}  CH={m['calinski']:.1f}  DB={m['davies']:.3f}  "
             f"ruido={100*m['noise_ratio']:.1f}%")
    print("\nMétricas del modelo final (sin ruido):", title)

    plot_clusters(X, labels, title=title, save=clusters_png, show=not args.no_show)

    # -----------------------------
    # Gráfica comparativa
    # -----------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(df["min_cluster_size"], df["silhouette"], marker="o", label="Silhouette (↑)")
    plt.plot(df["min_cluster_size"], df["calinski"]/df["calinski"].replace(0,np.nan).max(), marker="s",
             label="Calinski (norm ↑)")
    plt.plot(df["min_cluster_size"], df["davies"]/df["davies"].replace(0,np.nan).max(), marker="^",
             label="Davies (norm ↓)")
    plt.axvline(size_to_use, color="red", linestyle="--", label=f"min_cluster_size elegido = {size_to_use}")
    plt.xlabel("min_cluster_size")
    plt.ylabel("Índices normalizados de calidad")
    plt.title(f"HDBSCAN – métricas vs min_cluster_size (min_samples={args.min_samples})")
    plt.legend(); plt.tight_layout()
    if comp_png:
        plt.savefig(comp_png, dpi=160, bbox_inches="tight")
    if not args.no_show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()



