import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

# -------------------- IO / Utils --------------------
def load_arff(path):
    from scipy.io import arff
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    # Decodificar posibles columnas bytes->str
    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = df[c].str.decode("utf-8")
            except Exception:
                pass
    return df

def eval_labels(X, labels):
    u = np.unique(labels)
    if len(u) < 2:
        return dict(silhouette=np.nan, calinski=np.nan, davies=np.nan)
    return dict(
        silhouette=silhouette_score(X, labels),
        calinski=calinski_harabasz_score(X, labels),
        davies=davies_bouldin_score(X, labels),
    )

def plot_clusters(X, labels, title="", save=None, show=True):
    plt.figure(figsize=(6, 6))
    uniq = np.unique(labels)
    for lab in uniq:
        mask = labels == lab
        plt.scatter(X[mask, 0], X[mask, 1], s=20, label=f"c{lab}")
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

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="Agglomerative clustering plot")
    ap.add_argument("--arff", type=str, help="Ruta a .arff (x,y[,class])")
    ap.add_argument("--csv", type=str, help="Ruta a .csv (alternativa a --arff)")
    # Para compatibilidad con tu script anterior, mantenemos xcol/ycol,
    # pero por defecto tomaremos las 2 primeras columnas si no existen:
    ap.add_argument("--xcol", type=str, default=None)
    ap.add_argument("--ycol", type=str, default=None)
    ap.add_argument("--k", type=int, default=3, help="Número de clusters")
    ap.add_argument(
        "--linkage",
        type=str,
        default="ward",
        choices=["ward", "complete", "average", "single"],
        help="Criterio de enlace (ward requiere distancia euclídea)",
    )
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--save", type=str, help="PNG de salida")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    if not args.arff and not args.csv:
        ap.error("Proporciona --arff o --csv")

    # Cargar datos
    df = load_arff(args.arff) if args.arff else pd.read_csv(args.csv)

    # Selección de columnas: prioriza xcol/ycol si existen; si no, 2 primeras
    if args.xcol and args.ycol and args.xcol in df.columns and args.ycol in df.columns:
        X = df[[args.xcol, args.ycol]].values
    else:
        X = df.iloc[:, :2].values  # dos primeras columnas

    # Estandarizar si procede
    if args.standardize:
        X = StandardScaler().fit_transform(X)

    # Ajuste del modelo aglomerativo
    model = AgglomerativeClustering(n_clusters=args.k, linkage=args.linkage)
    labels = model.fit_predict(X)

    # Métricas y título
    metrics = eval_labels(X, labels)
    n_clusters = len(np.unique(labels))
    title = (
        f"Agglomerative (linkage={args.linkage}, k={args.k}) – "
        f"clusters={n_clusters} – "
        f"Sil={metrics['silhouette']:.3f}  CH={metrics['calinski']:.1f}  DB={metrics['davies']:.3f}"
    )
    print(title)

    # Plot
    plot_clusters(
        X,
        labels,
        title=title,
        save=args.save,
        show=not args.no_show,
    )

if __name__ == "__main__":
    main()

