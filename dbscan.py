#!/usr/bin/env python3
import argparse, os, numpy as np, matplotlib.pyplot as plt, pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# -------------------- IO --------------------
def load_arff(path):
    from scipy.io import arff
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    for c in df.columns:
        if df[c].dtype == object:
            try: df[c] = df[c].str.decode("utf-8")
            except: pass
    return df

# -------------------- Métricas (excluyendo ruido) --------------------
def eval_labels_no_noise(X, labels):
    mask = labels != -1
    if mask.sum() < 2:
        return dict(silhouette=np.nan, calinski=np.nan, davies=np.nan, n_clusters=0, noise_ratio=1.0)
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

# -------------------- Plots --------------------
def plot_clusters(X, labels, title="", save=None, show=True, noise_label=-1):
    plt.figure(figsize=(6,6))
    uniq = np.unique(labels)
    for lab in uniq:
        m = labels==lab
        if lab == noise_label:
            plt.scatter(X[m,0], X[m,1], s=18, facecolors="none", edgecolors="grey", label="noise")
        else:
            plt.scatter(X[m,0], X[m,1], s=20, label=f"c{lab}")
    plt.legend(fontsize=8)
    plt.title(title); plt.tight_layout()
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        plt.savefig(save, dpi=160, bbox_inches="tight")
    if show: plt.show()
    else: plt.close()

def k_distance_plot(X, k, best_eps=None, save=None, show=True):
    """ Curva de distancias al k-ésimo vecino (k-distance plot). 
        Si se pasa best_eps, marca visualmente el valor óptimo.
    """
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    dists, _ = nn.kneighbors(X)
    kd = np.sort(dists[:, -1])

    plt.figure(figsize=(7, 4))
    plt.plot(kd, color='steelblue', label=f"Distancia al {k}º vecino")
    if best_eps is not None:
        # buscar índice más cercano al valor óptimo
        idx = np.argmin(np.abs(kd - best_eps))
        plt.axhline(best_eps, color="red", linestyle="--", label=f"ε óptimo ≈ {best_eps:.4f}")
        plt.axvline(idx, color="orange", linestyle=":", label="punto de corte")
    plt.xlabel("Puntos ordenados")
    plt.ylabel(f"Distancia al {k}º vecino")
    plt.title(f"k-distance plot (k={k})")
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




# -------------------- Selección automática de eps --------------------
def candidate_eps_from_kdist(kd, percentiles=(85, 88, 90, 92, 94, 95, 96, 97, 98, 99)):
    kd = np.asarray(kd)
    kd = kd[np.isfinite(kd)]
    cand = [np.percentile(kd, p) for p in percentiles]
    # quitar duplicados y no-positivos
    return sorted({float(x) for x in cand if x > 0})

def grid_search_dbscan(X, min_samples, eps_candidates, noise_penalty=0.5):
    rows = []
    for eps in eps_candidates:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        m = eval_labels_no_noise(X, labels)
        # Penalizamos soluciones con mucho ruido
        score = (m["silhouette"] if np.isfinite(m["silhouette"]) else -np.inf) - noise_penalty*m["noise_ratio"]
        rows.append(dict(eps=eps, score=score, **m))
    df = pd.DataFrame(rows).sort_values(by=["score","silhouette","calinski","davies"], ascending=[False,False,False,True])
    best = df.iloc[0].to_dict()
    return best, df

# -------------------- MAIN --------------------
def main():
    ap = argparse.ArgumentParser(description="DBSCAN robusto (selección eps/min_samples con teoría correcta)")
    ap.add_argument("--arff", type=str, help="Ruta a .arff (x,y[,class])")
    ap.add_argument("--csv", type=str, help="Ruta a .csv")
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--save", type=str, help="PNG base (se usarán sufijos)")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--eps", type=float, help="Radio eps (si lo das, se usa tal cual)")
    ap.add_argument("--min-samples", type=int, default=5, dest="min_samples")
    ap.add_argument("--no-kdist", action="store_true", help="No mostrar/guardar k-distance plot")
    args = ap.parse_args()

    if not args.arff and not args.csv:
        ap.error("Proporciona --arff o --csv")

    # Cargar
    df = load_arff(args.arff) if args.arff else pd.read_csv(args.csv)
    X = df.iloc[:, :2].values
    if args.standardize:
        X = StandardScaler().fit_transform(X)

    # Archivos de salida
    base = os.path.splitext(args.save)[0] if args.save else None
    kdist_png = f"{base}_kdist.png" if base else None
    clusters_png = f"{base}_dbscan_clusters.png" if base else None
    comp_png = f"{base}_dbscan_comparativa_eps.png" if base else None

    # 1) k-distance plot (teórico y útil para eps)
    kd = k_distance_plot(X, k=args.min_samples, save=None if args.no_kdist else kdist_png, show=False)

    # 2) Candidatos de eps (si no nos dan uno)
    if args.eps is None:
        eps_candidates = candidate_eps_from_kdist(kd)
        if len(eps_candidates) == 0:
            raise SystemExit("No se pudieron obtener candidatos de eps. Revisa escalado o min_samples.")
        best, df = grid_search_dbscan(X, args.min_samples, eps_candidates)
        eps_use = float(best["eps"])
        print("\n=== DBSCAN – selección automática de eps desde k-distance ===")
        print(df.to_string(index=False, formatters={
            "eps": lambda v: f"{v:.4f}",
            "silhouette": lambda v: f"{v:.3f}",
            "calinski": lambda v: f"{v:.1f}",
            "davies": lambda v: f"{v:.3f}",
            "noise_ratio": lambda v: f"{100*v:,.1f}%",
            "score": lambda v: f"{v:.3f}",
        }))
        print(f"\nUsaremos eps = {eps_use:.4f} (min_samples={args.min_samples})")
        if not args.no_kdist:
            print("Mostrando k-distance plot con el ε óptimo marcado...")
            k_distance_plot(X, k=args.min_samples, best_eps=eps_use,
                            save=kdist_png, show=not args.no_show)

    else:
        eps_use = args.eps
        print(f"\nUsaremos eps proporcionado por el usuario: {eps_use:.4f} (min_samples={args.min_samples})")
        # (Opcional) aún así mostramos tabla para que compares contra percentiles:
        eps_candidates = candidate_eps_from_kdist(kd)
        if eps_candidates:
            best, df = grid_search_dbscan(X, args.min_samples, eps_candidates)
            print("\nTabla informativa (candidatos por percentil de k-distance):")
            print(df.to_string(index=False, formatters={
                "eps": lambda v: f"{v:.4f}",
                "silhouette": lambda v: f"{v:.3f}",
                "calinski": lambda v: f"{v:.1f}",
                "davies": lambda v: f"{v:.3f}",
                "noise_ratio": lambda v: f"{100*v:,.1f}%",
                "score": lambda v: f"{v:.3f}",
            }))

    # 3) Entrenar DBSCAN final y evaluar (sin ruido)
    labels = DBSCAN(eps=eps_use, min_samples=args.min_samples).fit_predict(X)
    m = eval_labels_no_noise(X, labels)
    title = (f"DBSCAN (eps={eps_use:.4f}, min_samples={args.min_samples}) – "
             f"clusters={m['n_clusters']} – "
             f"Sil={m['silhouette']:.3f}  CH={m['calinski']:.1f}  DB={m['davies']:.3f}  "
             f"ruido={100*m['noise_ratio']:.1f}%")
    print("\nMétricas del modelo final (sin ruido):", title)

    plot_clusters(X, labels, title=title, save=clusters_png, show=not args.no_show)

    # 4) Comparativa vs eps (solo para informar)
    if args.eps is None and len(eps_candidates) >= 2:
        plt.figure(figsize=(7,5))
        plt.plot(df["eps"], df["silhouette"], marker="o", label="Silhouette (↑)")
        plt.plot(df["eps"], df["calinski"]/df["calinski"].replace(0,np.nan).max(), marker="s", label="Calinski (norm ↑)")
        plt.plot(df["eps"], df["davies"]/df["davies"].replace(0,np.nan).max(), marker="^", label="Davies (norm ↓)")
        plt.axvline(eps_use, color="red", linestyle="--", label=f"eps elegido = {eps_use:.4f}")
        plt.xlabel("eps (derivado de k-distance)"); plt.ylabel("Índices normalizados")
        plt.title(f"Métricas vs eps (min_samples={args.min_samples})")
        plt.legend(); plt.tight_layout()
        if comp_png: plt.savefig(comp_png, dpi=160, bbox_inches="tight")
        if not args.no_show: plt.show()
        else: plt.close()

if __name__ == "__main__":
    main()




# -------------------- Selección automática de eps --------------------
def candidate_eps_from_kdist(kd, percentiles=(85, 88, 90, 92, 94, 95, 96, 97, 98, 99)):
    kd = np.asarray(kd)
    kd = kd[np.isfinite(kd)]
    cand = [np.percentile(kd, p) for p in percentiles]
    # quitar duplicados y no-positivos
    return sorted({float(x) for x in cand if x > 0})

def grid_search_dbscan(X, min_samples, eps_candidates, noise_penalty=0.5):
    rows = []
    for eps in eps_candidates:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        m = eval_labels_no_noise(X, labels)
        # Pen