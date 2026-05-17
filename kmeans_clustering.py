import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

# ── Palette ──────────────────────────────────────────────────────────────────
BG      = "#0a0f1e"
PANEL   = "#111827"
BORDER  = "#1f2d45"
TEXT    = "#e2e8f0"
MUTED   = "#64748b"
CLUSTER_COLORS = ["#38bdf8", "#f472b6", "#4ade80", "#fb923c", "#a78bfa",
                  "#facc15", "#34d399", "#f87171", "#818cf8", "#e879f9"]

# ── 1. Load & feature engineering ────────────────────────────────────────────
df = pd.read_csv("Mall_Customers.csv")
df["Gender_enc"] = (df["Gender"] == "Male").astype(int)

FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X_raw = df[FEATURES].values

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

print("=" * 56)
print("  K-Means Customer Segmentation")
print("=" * 56)
print(f"\n  Customers : {len(df):,}")
print(f"  Features  : {FEATURES}")

# ── 2. Elbow + Silhouette sweep ───────────────────────────────────────────────
K_RANGE = range(2, 11)
inertias, sil_scores = [], []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, km.labels_))

best_k = int(K_RANGE.start + np.argmax(sil_scores))
print(f"\n  Best k (silhouette) : {best_k}  (score = {max(sil_scores):.4f})")

# ── 3. Final model ────────────────────────────────────────────────────────────
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
km_final.fit(X)
labels = km_final.labels_
df["Cluster"] = labels

# Cluster stats
cluster_stats = df.groupby("Cluster")[FEATURES + ["Gender_enc"]].agg(
    ["mean", "std", "count"]
).round(1)

print("\n── Cluster Profiles ─────────────────────────────────")
for c in sorted(df["Cluster"].unique()):
    sub = df[df["Cluster"] == c]
    print(f"\n  Cluster {c}  (n={len(sub)})")
    print(f"    Age             : {sub['Age'].mean():.1f} ± {sub['Age'].std():.1f}")
    print(f"    Annual Income   : ${sub['Annual Income (k$)'].mean():.0f}k ± {sub['Annual Income (k$)'].std():.0f}k")
    print(f"    Spending Score  : {sub['Spending Score (1-100)'].mean():.1f} ± {sub['Spending Score (1-100)'].std():.1f}")
    print(f"    % Female        : {(sub['Gender']=='Female').mean()*100:.0f}%")

sil_vals = silhouette_samples(X, labels)
overall_sil = silhouette_score(X, labels)
print(f"\n  Overall Silhouette : {overall_sil:.4f}")

# ── 4. PCA for 2-D projections ────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
centroids_pca = pca.transform(km_final.cluster_centers_)

# ── 5. Cluster name heuristics ────────────────────────────────────────────────
def name_cluster(sub):
    income = sub["Annual Income (k$)"].mean()
    score  = sub["Spending Score (1-100)"].mean()
    age    = sub["Age"].mean()
    if income > 70 and score > 60:  return "High Earners\nBig Spenders"
    if income > 70 and score <= 60: return "Wealthy\nSavers"
    if income <= 50 and score > 60: return "Low Income\nHigh Spenders"
    if income <= 50 and score <= 50: return "Budget\nConscious"
    if age > 45:                    return "Mature\nModerate"
    return "Average\nCustomers"

cluster_names = {c: name_cluster(df[df["Cluster"] == c]) for c in range(best_k)}

# ── 6. Figure layout ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.44, wspace=0.35,
                       left=0.06, right=0.97, top=0.93, bottom=0.06)

def style(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    ax.tick_params(colors=MUTED, labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

# ── 6a. Elbow curve ───────────────────────────────────────────────────────────
ax_elbow = fig.add_subplot(gs[0, 0])
ks = list(K_RANGE)
ax_elbow.plot(ks, inertias, color=CLUSTER_COLORS[0], lw=2, marker="o",
              markersize=5, markerfacecolor=BG)
ax_elbow.axvline(best_k, color=CLUSTER_COLORS[2], lw=1.2, linestyle="--", alpha=0.8)
ax_elbow.fill_between(ks, inertias, alpha=0.08, color=CLUSTER_COLORS[0])
style(ax_elbow, "Elbow Curve — Inertia vs k", "Number of Clusters k", "Inertia")

# ── 6b. Silhouette scores ─────────────────────────────────────────────────────
ax_sil = fig.add_subplot(gs[0, 1])
bar_cols = [CLUSTER_COLORS[2] if k == best_k else CLUSTER_COLORS[0] for k in ks]
bars = ax_sil.bar(ks, sil_scores, color=bar_cols, edgecolor=BG, linewidth=0.5, width=0.6)
for b, v in zip(bars, sil_scores):
    ax_sil.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", color=TEXT, fontsize=6.5)
ax_sil.axhline(max(sil_scores), color=CLUSTER_COLORS[2], lw=0.8, linestyle="--", alpha=0.5)
style(ax_sil, "Silhouette Score vs k", "Number of Clusters k", "Silhouette Score")

# ── 6c. PCA scatter ───────────────────────────────────────────────────────────
ax_pca = fig.add_subplot(gs[0, 2])
for c in range(best_k):
    mask = labels == c
    ax_pca.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   color=CLUSTER_COLORS[c], s=30, alpha=0.7, edgecolors="none",
                   label=f"C{c}")
ax_pca.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
               color="white", s=120, marker="*", zorder=5, edgecolors=BG, linewidths=0.8)
style(ax_pca, "PCA Projection (2-D)", f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
      f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax_pca.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT,
              markerscale=0.8, ncol=best_k)

# ── 6d. Income vs Spending scatter ────────────────────────────────────────────
ax_main = fig.add_subplot(gs[1, :2])
for c in range(best_k):
    mask = labels == c
    ax_main.scatter(df.loc[mask, "Annual Income (k$)"],
                    df.loc[mask, "Spending Score (1-100)"],
                    color=CLUSTER_COLORS[c], s=55, alpha=0.75, edgecolors="none",
                    label=cluster_names[c].replace("\n", " "))
# Centroids (inverse transform)
cents_orig = scaler.inverse_transform(km_final.cluster_centers_)
feat_idx = {f: i for i, f in enumerate(FEATURES)}
ax_main.scatter(cents_orig[:, feat_idx["Annual Income (k$)"]],
                cents_orig[:, feat_idx["Spending Score (1-100)"]],
                color="white", s=200, marker="*", zorder=6, edgecolors=BG, linewidths=1)
for c in range(best_k):
    ax_main.annotate(cluster_names[c],
                     (cents_orig[c, feat_idx["Annual Income (k$)"]],
                      cents_orig[c, feat_idx["Spending Score (1-100)"]]),
                     textcoords="offset points", xytext=(8, 5),
                     fontsize=6.5, color=CLUSTER_COLORS[c], fontweight="bold")
style(ax_main, "Income vs Spending Score — Customer Segments",
      "Annual Income (k$)", "Spending Score (1–100)")
legend = ax_main.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT,
                        loc="upper left", markerscale=1.1)

# ── 6e. Silhouette plot per sample ────────────────────────────────────────────
ax_silplot = fig.add_subplot(gs[1, 2])
y_lower = 10
for c in range(best_k):
    c_sil = np.sort(sil_vals[labels == c])
    size_c = len(c_sil)
    y_upper = y_lower + size_c
    ax_silplot.fill_betweenx(np.arange(y_lower, y_upper), 0, c_sil,
                              facecolor=CLUSTER_COLORS[c], alpha=0.85)
    ax_silplot.text(-0.05, y_lower + size_c / 2, str(c),
                    ha="right", va="center", color=CLUSTER_COLORS[c], fontsize=8, fontweight="bold")
    y_lower = y_upper + 10
ax_silplot.axvline(overall_sil, color="white", lw=1, linestyle="--")
ax_silplot.set_xlim(-0.25, 1.0)
ax_silplot.set_yticks([])
ax_silplot.text(overall_sil + 0.01, y_lower * 0.95, f"avg={overall_sil:.3f}",
                color="white", fontsize=7)
style(ax_silplot, "Silhouette — Per Sample", "Coefficient", "Cluster")

# ── 6f. Age distribution per cluster ─────────────────────────────────────────
ax_age = fig.add_subplot(gs[2, 0])
for c in range(best_k):
    ages = df.loc[labels == c, "Age"]
    ax_age.hist(ages, bins=12, color=CLUSTER_COLORS[c], alpha=0.6, edgecolor=BG, lw=0.4,
                label=f"C{c}")
style(ax_age, "Age Distribution by Cluster", "Age", "Count")
ax_age.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

# ── 6g. Radar / spider chart ─────────────────────────────────────────────────
ax_radar = fig.add_subplot(gs[2, 1], projection="polar")
ax_radar.set_facecolor(PANEL)
categories = ["Age", "Income", "Spending\nScore"]
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# Normalise centroids to 0-1 for radar
mins = X_raw.min(axis=0)
rngs = X_raw.max(axis=0) - mins
cents_norm = (cents_orig[:, [feat_idx[f] for f in FEATURES]] - mins) / rngs

for c in range(best_k):
    vals = cents_norm[c].tolist() + [cents_norm[c][0]]
    ax_radar.plot(angles, vals, color=CLUSTER_COLORS[c], lw=1.8, label=f"C{c}")
    ax_radar.fill(angles, vals, color=CLUSTER_COLORS[c], alpha=0.12)

ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(categories, color=TEXT, fontsize=8)
ax_radar.set_yticklabels([])
ax_radar.set_title("Cluster Radar Profiles", color=TEXT, fontsize=10,
                   fontweight="bold", pad=14)
ax_radar.tick_params(colors=MUTED)
ax_radar.spines["polar"].set_edgecolor(BORDER)
ax_radar.grid(color=BORDER, linestyle="--", linewidth=0.5)
ax_radar.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT,
                loc="upper right", bbox_to_anchor=(1.35, 1.1))

# ── 6h. Cluster size bar ─────────────────────────────────────────────────────
ax_size = fig.add_subplot(gs[2, 2])
sizes = [int((labels == c).sum()) for c in range(best_k)]
bar2 = ax_size.bar(range(best_k), sizes, color=CLUSTER_COLORS[:best_k],
                    edgecolor=BG, linewidth=0.5, width=0.55)
for b, v in zip(bar2, sizes):
    ax_size.text(b.get_x() + b.get_width() / 2, v + 1, str(v),
                 ha="center", va="bottom", color=TEXT, fontsize=8, fontweight="bold")
ax_size.set_xticks(range(best_k))
ax_size.set_xticklabels([f"C{c}" for c in range(best_k)])
style(ax_size, "Customers per Cluster", "Cluster", "Count")

# ── Title ─────────────────────────────────────────────────────────────────────
fig.suptitle(
    f"K-Means Customer Segmentation  ·  k = {best_k}  ·  Silhouette = {overall_sil:.3f}",
    color=TEXT, fontsize=14, fontweight="bold", y=0.97
)

out = "kmeans_clustering.png"

plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()

print(f"\nDashboard saved → {out}")
