import csv
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(BASE_DIR, "corpusAB.csv")

THRESHOLDS = [t / 100 for t in range(10, 101, 10)]

# Lecture du CSV
rows_data = []
with open(INPUT, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        texte = row["Texte"]
        letters = [c for c in texte if c.isalpha()] # on ne regarde que les lettres
        upper = [c for c in letters if c.isupper()]
        ratio = len(upper) / len(letters) if letters else 0.0
        rows_data.append({
            "ID": row["ID"],
            "Texte": texte,
            "Langue": row["Langue"],
            "Niveau": row["Niveau"],
            "ratio": ratio,
        })

total_rows = len(rows_data)
print(f"{total_rows} textes charges.")

# Comptage par seuil
counts = {}
pct_of_rows = {}

for thr in THRESHOLDS:
    n = sum(1 for r in rows_data if r["ratio"] >= thr)
    counts[thr] = n
    pct_of_rows[thr] = (n / total_rows * 100) if total_rows else 0

# Chutes entre seuils consecutifs
drops = {}
for i in range(1, len(THRESHOLDS)):
    drops[THRESHOLDS[i]] = counts[THRESHOLDS[i - 1]] - counts[THRESHOLDS[i]]

# Mediane des chutes comme reference de stabilite
drop_values = [drops[t] for t in THRESHOLDS[1:]]
mediane = sorted(drop_values)[len(drop_values) // 2]

# Zone de tri : descente initiale avant que la courbe se stabilise
SEUIL_MIN = THRESHOLDS[0] # premier seuil analyse à 10%
SEUIL_MAX = THRESHOLDS[0]
for i in range(1, len(THRESHOLDS)):
    if drops[THRESHOLDS[i]] > mediane:
        SEUIL_MAX = THRESHOLDS[i] # on avance tant que les chutes restent significatives
    else:
        break # première stabilisation = fin de la zone

# Zone nettoyage : reprise des chutes après le plateau de stabilisation, en scannant depuis la fin
SEUIL_NETTOYAGE = THRESHOLDS[-1]
for i in range(len(THRESHOLDS) - 1, 0, -1):
    if drops[THRESHOLDS[i]] > mediane:
        SEUIL_NETTOYAGE = THRESHOLDS[i - 1] # on remonte tant que les chutes restent significatives
    else:
        break # fin du plateau = début de la zone nettoyage

print(f"\nZone détectée : {int(SEUIL_MIN * 100)}% a {int(SEUIL_MAX * 100)}%")
print(f"Zone nettoyage détectée : supérieur ou égal a {int(SEUIL_NETTOYAGE * 100)}%")

print("\n Résultats par seuil ")
print(f"{'Seuil':>7} {'Détectés':>10} {'Pourcentage du corpus':>9} {'Chute':>8}")
for i, thr in enumerate(THRESHOLDS):
    chute = f"{drops[thr]:>6}" if i > 0 else "     -"
    print(f" {int(thr*100):3d}% {counts[thr]:>10} {pct_of_rows[thr]:>8.2f}% {chute}")

# Variables pour le graphique
x_labels = [f"{int(t*100)}%" for t in THRESHOLDS]
x_pos = np.arange(len(THRESHOLDS))
y_counts = [counts[t] for t in THRESHOLDS]

# Index des zones pour le graphique
idx_min = THRESHOLDS.index(SEUIL_MIN)
idx_max = THRESHOLDS.index(SEUIL_MAX)
idx_nettoyage = THRESHOLDS.index(SEUIL_NETTOYAGE)

# Création de la figure
fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
ax.set_facecolor("#f8f9fa")

# Zones colorées sur la courbe
ax.axvspan(idx_min - 0.5, idx_max + 0.5, color="#f4a261", alpha=0.18, zorder=1, label="_nolegend_") # tri manuel
ax.axvspan(idx_nettoyage + 0.5, len(THRESHOLDS) - 0.5, color="#e63946", alpha=0.12, zorder=1, label="_nolegend_") # nettoyage

# Paramètres de la courbe
ax.plot(x_pos, y_counts, color="#007bff", linewidth=2.5, zorder=3)
ax.fill_between(x_pos, y_counts, alpha=0.1, color="#007bff")
ax.scatter(x_pos, y_counts, color="#007bff", s=60, zorder=4)

# Étiquettes de zone
ymax = max(y_counts)
ax.text((idx_min + idx_max) / 2, ymax * 0.97, "Tri\nmanuel", ha="center", va="top",
        fontsize=8.5, color="#c75000", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f4a261", alpha=0.35, edgecolor="none"))
ax.text((idx_nettoyage + len(THRESHOLDS) - 1) / 2 + 0.5, ymax * 0.97, "Nettoyage\nautomatique", ha="center", va="top",
        fontsize=8.5, color="#9b1a23", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e63946", alpha=0.25, edgecolor="none"))

# Titres et labels
ax.set_title("Analyse de présence des majuscules dans les textes du corpus",
             fontsize=14, fontweight="bold", pad=20)
ax.set_xlabel("Pourcentage de majuscules dans un texte", fontsize=11)
ax.set_ylabel("Nombre de textes détectés", fontsize=11)

# Mise en forme des axes
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.grid(axis="y", color="#dee2e6", linestyle="--", linewidth=0.7)
ax.set_xlim(-0.5, 9.5)

# Ajout des valeurs au-dessus des points
for xi, yi in zip(x_pos, y_counts):
    ax.annotate(f"{yi:,}", (xi, yi), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=9)

plt.tight_layout()
plt.show()

# Rapport textuel
out_txt = os.path.join(BASE_DIR, "rapport_seuils.txt")
with open(out_txt, "w", encoding="utf-8") as f:
    f.write("Rapport d'analyse des seuils de majuscules\n\n")
    f.write(f"Zone détectée : {int(SEUIL_MIN * 100)}% a {int(SEUIL_MAX * 100)}%\n")
    f.write(f"Zone nettoyage detectee : superieur ou egal a {int(SEUIL_NETTOYAGE * 100)}%\n\n")
    f.write(f"{'Seuil':<8}{'Detectes':>12}{'% corpus':>12}{'Chute':>10}\n")
    f.write("-" * 44 + "\n")
    for i, thr in enumerate(THRESHOLDS):
        chute = f"{drops[thr]}" if i > 0 else "-"
        f.write(f"{int(thr*100):>4}%    {counts[thr]:>10}    {pct_of_rows[thr]:>9.2f}%  {chute:>6}\n")
print(f"Rapport texte géneré dans {out_txt}")

# Export CSV nettoyage automatique (ratio supérieur ou égal au seuil de stagnation détecté)
out_csv_clean = os.path.join(BASE_DIR, f"nettoyage_{int(SEUIL_NETTOYAGE * 100)}pct_et_plus.csv")
clean_rows = [r for r in rows_data if r["ratio"] >= SEUIL_NETTOYAGE]

with open(out_csv_clean, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["ID", "Texte", "Langue", "Niveau", "ratio_majuscules"],
        extrasaction="ignore"
    )
    writer.writeheader()
    for r in clean_rows:
        writer.writerow({
            "ID": r["ID"], "Texte": r["Texte"],
            "Langue": r["Langue"], "Niveau": r["Niveau"],
            "ratio_majuscules": f"{r['ratio']:.4f}",
        })
print(f"CSV nettoyage exporté : {len(clean_rows)} textes dans {out_csv_clean}")

# Export CSV tri manuel (zone avant stagnation détectée automatiquement)
out_csv_tri = os.path.join(BASE_DIR, f"tri_manuel_{int(SEUIL_MIN * 100)}_{int(SEUIL_MAX * 100)}pct.csv")
tri_rows = [r for r in rows_data if SEUIL_MIN <= r["ratio"] < SEUIL_MAX]

with open(out_csv_grey, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["ID", "Texte", "Langue", "Niveau", "ratio_majuscules"],
        extrasaction="ignore"
    )
    writer.writeheader()
    for r in grey_rows:
        writer.writerow({
            "ID": r["ID"], "Texte": r["Texte"],
            "Langue": r["Langue"], "Niveau": r["Niveau"],
            "ratio_majuscules": f"{r['ratio']:.4f}",
        })
print(f"CSV tri manuel exporté : {len(tri_rows)} textes dans {out_csv_tri}")
