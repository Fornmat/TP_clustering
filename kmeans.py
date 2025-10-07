import argparse, os, numpy as np, matplotlib.pyplot as plt, pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def load_arff(path):
    from scipy.io import arff
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    for c in df.columns:
        if df[c].dtype == object:
            try: df[c] = df[c].str.decode("utf-8")
            except: pass
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
    plt.figure(figsize=(6,6))
    for lab in np.unique(labels):
        mask = labels==lab
        plt.scatter(X[mask,0], X[mask,1], s=20, label=f"c{lab}")
    if centers is not None:
        plt.scatter(centers[:,0], centers[:,1], c="black", marker="x", s=120, label="centers")
    plt.legend(fontsize=8)
    plt.title(title); plt.tight_layout()
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        plt.savefig(save, dpi=160, bbox_inches="tight")
    if show: plt.show()
    else: plt.close()

def main():
    ap = argparse.ArgumentParser(description="K-Means plot")
    ap.add_argument("--arff", type=str, help="Ruta a .arff (x,y[,class])")
    ap.add_argument("--csv", type=str, help="Ruta a .csv (alternativa a --arff)")
    ap.add_argument("--xcol", type=str, default="x")
    ap.add_argument("--ycol", type=str, default="y")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--save", type=str, help="PNG de salida")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    if not args.arff and not args.csv:
        ap.error("Proporciona --arff o --csv")

    if args.arff:
        df = load_arff(args.arff)
    else:
        df = pd.read_csv(args.csv)

    X = df.iloc[:, :2].values
    if args.standardize:
        X = StandardScaler().fit_transform(X)

    model = KMeans(n_clusters=args.k, init="k-means++", n_init=20, random_state=42)
    labels = model.fit_predict(X)
    metrics = eval_labels(X, labels)
    title = f"K-Means (k={args.k}) – Sil={metrics['silhouette']:.3f}  CH={metrics['calinski']:.1f}  DB={metrics['davies']:.3f}"
    print(title)

    plot_clusters(X, labels, centers=model.cluster_centers_, title=title, save=args.save)

if __name__ == "__main__":
    main()