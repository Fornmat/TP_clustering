#!/usr/bin/env python3
import argparse, os, numpy as np, matplotlib.pyplot as plt, pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


# ------------------------------------------------------
# FUNCIONES AUXILIARES
# ------------------------------------------------------
def load_arff(path):
    from scipy.io import arff
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = df[c].str.decode("utf-8")
            except:
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


def plot_clusters(X, labels, centers=None, title="", save=None, show=True):
    plt.figure(figsize=(6, 6))
    for lab in np.unique(labels):
        mask = labels == lab
        plt.scatter(X[mask, 0], X[mask, 1], s=20, label=f"c{lab}")
    if centers is not None:
        plt.scatter(
            centers[:, 0], centers[:, 1],
            c="black", marker="x", s=120, label="centers"
        )
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
# BÚSQUEDA AUTOMÁTICA DE k Y MÉTRICAS
# ------------------------------------------------------
def kmeans_grid_search(X, ks=range(2, 11), n_init=20, random_state=42):
    metrics_rows = []
    for k in ks:
        km = KMeans(n_clusters=k, init="k-means++", n_init=n_init,
                    random_state=random_state)
        labels = km.fit_predict(X)
        m = eval_labels(X, labels)
        metrics_rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": m["silhouette"],
            "calinski": m["calinski"],
            "davies": m["davies"],
        })
    dfm = pd.DataFrame(metrics_rows)
    # Seleccionar mejor k
    dfm_sorted = dfm.sort_values(
        by=["silhouette", "calinski", "davies", "inertia"],
        ascending=[False, True, False, True]
    )
    best_k = int(dfm_sorted.iloc[0]["k"])
    return best_k, dfm


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="K-Means con análisis comparativo")
    ap.add_argument("--arff", type=str, help="Ruta a .arff (x,y[,class])")
    ap.add_argument("--csv", type=str, help="Ruta a .csv")
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--save", type=str, help="PNG base de salida")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--k", type=int, help="Número fijo de clusters (opcional)")
    args = ap.parse_args()

    if not args.arff and not args.csv:
        ap.error("Proporciona --arff o --csv")

    # Cargar dataset
    df = load_arff(args.arff) if args.arff else pd.read_csv(args.csv)
    X = df.iloc[:, :2].values
    if args.standardize:
        X = StandardScaler().fit_transform(X)

    # Buscar mejor k (si no se especifica)
    best_k, dfm = kmeans_grid_search(X, ks=range(2, 11))

    print("\n=== K-MEANS – SELECCIÓN AUTOMÁTICA DE HIPERPARÁMETROS ===")
    print(dfm.to_string(index=False, formatters={
        "inertia": lambda v: f"{v:.1f}",
        "silhouette": lambda v: f"{v:.3f}",
        "calinski": lambda v: f"{v:.1f}",
        "davies": lambda v: f"{v:.3f}",
    }))

    # ------------------------------------------------------------
    # Si el usuario pasa --k, se usa ese valor, si no, se usa el óptimo
    # ------------------------------------------------------------
    k_to_use = args.k if args.k else best_k
    print(f"\nSe utilizará k = {k_to_use} "
          f"({'fijado por el usuario' if args.k else 'óptimo por Silhouette'})")

    # Entrenar modelo final con ese k
    model = KMeans(n_clusters=k_to_use, init="k-means++", n_init=20, random_state=42)
    labels = model.fit_predict(X)
    metrics = eval_labels(X, labels)

    title = (f"K-Means (k={k_to_use}) – "
             f"Sil={metrics['silhouette']:.3f}  "
             f"CH={metrics['calinski']:.1f}  DB={metrics['davies']:.3f}")
    print("\nMétricas del modelo final:", title)

    # Mostrar clustering final
    best_png = None
    if args.save:
        base, _ = os.path.splitext(args.save)
        best_png = f"{base}_k{k_to_use}_clusters.png"
    plot_clusters(X, labels, centers=model.cluster_centers_, title=title,
                  save=best_png, show=not args.no_show)

    # ------------------------------------------------------------
    # Gráfica comparativa de calidad vs k
    # ------------------------------------------------------------
    plt.figure(figsize=(7,5))
    plt.plot(dfm["k"], dfm["silhouette"], marker="o", label="Silhouette (↑ mejor)")
    plt.plot(dfm["k"], dfm["calinski"]/dfm["calinski"].max(), marker="s",
             label="Calinski (normalizado ↑ mejor)")
    plt.plot(dfm["k"], dfm["davies"]/dfm["davies"].max(), marker="^",
             label="Davies (normalizado ↓ mejor)")
    plt.axvline(best_k, color="red", linestyle="--", label=f"k óptimo = {best_k}")
    if args.k:
        plt.axvline(args.k, color="blue", linestyle=":", label=f"k usuario = {args.k}")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Índices normalizados de calidad")
    plt.title("Comparativa de métricas de clustering vs k")
    plt.legend()
    plt.tight_layout()
    if args.save:
        base, _ = os.path.splitext(args.save)
        plt.savefig(f"{base}_comparativa_metricas.png", dpi=160, bbox_inches="tight")
    if not args.no_show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()


