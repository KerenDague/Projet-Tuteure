import matplotlib
matplotlib.use("Agg")
import re
import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from collections import Counter
from scipy.sparse import hstack, issparse, csr_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix, classification_report

warnings.filterwarnings("ignore")

nlp = spacy.load("fr_core_news_md")

TEXT_COLUMN = 'Texte'
LABEL_COLUMN = 'Langue'
OUTPUT_DIR = "resultats_f_indéppendant"

N_PERMUTATIONS = 100
N_FOLDS = 5

# connecteurs logiques courants en français
CONNECTEURS = [
    "et", "mais", "ou", "donc", "or", "ni", "car",
    "cependant", "pourtant", "néanmoins", "toutefois", "en revanche", "au contraire",
    "parce que", "puisque", "étant donné", "vu que",
    "ainsi", "par conséquent", "de ce fait", "c'est pourquoi",
    "de plus", "en outre", "par ailleurs", "également",
    "bien que", "quoique", "même si", "malgré", "certes",
]

PRONOMS_SUJETS = {"je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles"}

# les 16 premières sont numériques, les 3 dernières sont les tfidf
FEATURE_NAMES = [
    "punct_ratio", "det_ratio", "freq_prepositions", "freq_majuscules_hors_debut",
    "nb_mots_moyen_par_phrase", "freq_pronoms_sujets", "freq_present",
    "freq_passe_compose", "freq_imparfait", "freq_futur", "freq_conditionnel",
    "freq_connecteurs", "ratio_sov", "lisibilite", "complexite_syntaxique",
    "repetition_noms", "tfidf_char", "tfidf_word", "tfidf_pos",
]


# Extraction POS pour le vectoriseur tfidf_pos
class POSExtractor(BaseEstimator, TransformerMixin):
    # convertit chaque texte en séquence de POS tags (ex: "NOUN VERB DET ...")
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = []
        for doc in nlp.pipe(X, disable=["ner"]):
            out.append(" ".join(t.pos_ for t in doc if not t.is_space))
        return out


# Features numériques / stylistiques
def count_syllables(word):
    return max(1, len(re.findall(r'[aeiouyàâéèêëîïôùûü]+', word.lower())))


def tree_depth(token):
    d, t = 0, token
    while t.head != t:
        d += 1
        t = t.head
    return d


def extract_all_features(texts):
    rows = []
    for doc in nlp.pipe(texts, disable=["ner"]):
        tokens = [t for t in doc if not t.is_space]
        nb_tok = max(1, len(tokens))
        text_low = doc.text.lower()
        phrases = list(doc.sents)
        nb_phrases = max(1, len(phrases))

        punct_ratio = sum(1 for t in tokens if t.text in ".,;:!?") / nb_tok

        det_count = sum(1 for t in tokens if t.pos_ == "DET")
        noun_count = max(1, sum(1 for t in tokens if t.pos_ == "NOUN"))
        det_ratio = det_count / noun_count

        freq_prep = sum(1 for t in tokens if t.pos_ == "ADP") / nb_tok

        nb_maj = 0
        for sent in doc.sents:
            st = [t for t in sent if not t.is_space]
            for t in st[1:]:
                if t.text and t.text[0].isupper() and t.pos_ != "PROPN":
                    nb_maj += 1
        freq_maj = nb_maj / nb_tok

        nb_mots_moy = float(np.mean([len([t for t in s if not t.is_space]) for s in phrases]))

        freq_pron = sum(
            1 for t in tokens if t.text.lower() in PRONOMS_SUJETS and t.dep_ == "nsubj"
        ) / nb_tok

        nb_verbes = max(1, sum(1 for t in tokens if t.pos_ in ("VERB", "AUX")))
        freq_pres= sum(1 for t in tokens if t.pos_ == "VERB" and "Tense=Pres" in t.tag_) / nb_verbes
        freq_pc= sum(1 for t in tokens if t.pos_ == "AUX"  and "Tense=Pres" in t.tag_) / nb_verbes
        freq_imp= sum(1 for t in tokens if t.pos_ == "VERB" and "Tense=Imp" in t.tag_) / nb_verbes
        freq_fut= sum(1 for t in tokens if t.pos_ == "VERB" and "Tense=Fut" in t.tag_) / nb_verbes
        freq_cond  = sum(1 for t in tokens if t.pos_ == "VERB" and "Mood=Cnd" in t.tag_) / nb_verbes

        nb_conn = sum(
            len(re.findall(r'\b' + re.escape(c) + r'\b', text_low)) for c in CONNECTEURS
        )
        freq_conn = nb_conn / nb_tok

        # ordre sujet-verbe-objet : +1 si OV, -1 si VS, 0 sinon
        scores_sov = []
        for sent in doc.sents:
            st = list(sent)
            pos_map = {t.i: idx for idx, t in enumerate(st)}
            for t in st:
                if t.pos_ == "VERB":
                    children = list(t.children)
                    subj = next((c for c in children if c.dep_ == "nsubj"), None)
                    obj  = next((c for c in children if c.dep_ == "obj"),   None)
                    if subj and obj:
                        v = pos_map.get(t.i, 0)
                        s = pos_map.get(subj.i, 0)
                        o = pos_map.get(obj.i, 0)
                        if o < v: scores_sov.append(1)
                        elif s > v: scores_sov.append(-1)
                        else: scores_sov.append(0)
        ratio_sov = float(np.mean(scores_sov)) if scores_sov else 0.0

        mots_alpha = [t.text for t in tokens if t.is_alpha]
        nb_mots = max(1, len(mots_alpha))
        nb_syllabes = sum(count_syllables(m) for m in mots_alpha)
        lisibilite = 206.835 - 1.015 * (nb_mots / nb_phrases) - 84.6 * (nb_syllabes / nb_mots)

        depths = [tree_depth(t) for t in tokens if t.pos_ in ("NOUN", "VERB", "ADJ")]
        prof_moy = float(np.mean(depths)) if depths else 0.0
        ratio_sconj = sum(1 for t in tokens if t.pos_ == "SCONJ") / nb_phrases
        complexite_syntaxique = prof_moy + ratio_sconj

        noun_lemmas = [t.lemma_.lower() for t in tokens if t.pos_ == "NOUN"]
        if noun_lemmas:
            nc = Counter(noun_lemmas)
            repetition_noms = sum(1 for v in nc.values() if v >= 2) / len(nc)
        else:
            repetition_noms = 0.0

        rows.append([
            punct_ratio, det_ratio, freq_prep, freq_maj, nb_mots_moy, freq_pron,
            freq_pres, freq_pc, freq_imp, freq_fut, freq_cond,
            freq_conn, ratio_sov, lisibilite, complexite_syntaxique, repetition_noms,
        ])

    return np.array(rows, dtype=float)


# Vectoriseurs TF-IDF
def build_tfidf_pipelines():
    char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 6), max_features=50000, sublinear_tf=True)
    word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=30000, sublinear_tf=True)
    pos_pipeline = Pipeline([("pos_extract", POSExtractor()), ("pos_tfidf", TfidfVectorizer())])
    return char_vectorizer, word_vectorizer, pos_pipeline


def fit_transform_tfidf(texts_train, texts_all, char_vec, word_vec, pos_pipe):
    # on fit uniquement sur le train pour éviter la fuite de données
    char_vec.fit(texts_train)
    word_vec.fit(texts_train)
    pos_pipe.fit(texts_train)
    return (char_vec.transform(texts_all), word_vec.transform(texts_all), pos_pipe.transform(texts_all))


# Chargement
def load_data(path):
    try:
        try:
            df = pd.read_csv(path, sep=';')
            if df.shape[1] < 2:
                df = pd.read_csv(path, sep=',')
        except Exception:
            df = pd.read_csv(path, sep=None, engine='python')
    except FileNotFoundError:
        print(f"Le fichier '{path}' n'a pas été trouvé.")
        return None, None
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    print(f"{len(df)} échantillons chargés — {df[LABEL_COLUMN].nunique()} classes")
    return df[TEXT_COLUMN], df[LABEL_COLUMN]


# Test de permutation
def permutation_test_feature(X_feat, y, cv, f1_observe, random_state=42):
    rng = np.random.default_rng(random_state)

    if issparse(X_feat):
        pipe = Pipeline([("svm", SVC(kernel="rbf", C=1.0, class_weight="balanced"))])
    else:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, class_weight="balanced")),
        ])

    null_dist = []
    for _ in range(N_PERMUTATIONS):
        y_perm = rng.permutation(y)
        s = cross_val_score(pipe, X_feat, y_perm, cv=cv, scoring="f1_macro", n_jobs=-1)
        null_dist.append(s.mean())

    null_dist = np.array(null_dist)
    p_val = float(np.mean(null_dist >= f1_observe))

    return {
        "f1_obs": f1_observe, "null_scores": null_dist,
        "null_mean": float(null_dist.mean()), "null_std": float(null_dist.std()),
        "p_value": p_val, "significatif": p_val < 0.05,
    }


# Plots
def plot_confusion_matrix(y_true, y_pred, labels, name, out_dir):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(cm, index=labels, columns=labels),
                annot=True, fmt='d', cmap='Blues', linewidths=.5)
    plt.title(f'Matrice de confusion — {name}', fontsize=13)
    plt.ylabel('Vraie langue', fontsize=11)
    plt.xlabel('Langue prédite', fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_permutation_histograms(perm_results, out_dir):
    n = len(perm_results)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for ax, (feat, res) in zip(axes_flat, perm_results.items()):
        color = "#27ae60" if res["significatif"] else "#e74c3c"
        ax.hist(res["null_scores"], bins=30, color="#aab4c8", edgecolor="white", alpha=0.85)
        ax.axvline(res["f1_obs"],    color=color,  linewidth=2.5, label=f"F1 obs = {res['f1_obs']:.3f}")
        ax.axvline(res["null_mean"], color="#555", linewidth=1.2, linestyle="--",
                   label=f"Moy. nulle = {res['null_mean']:.3f}")
        note = f"p = {res['p_value']:.3f}\n{'✓ SIGNIFICATIF' if res['significatif'] else '✗ non significatif'}"
        ax.text(0.97, 0.96, note, transform=ax.transAxes, ha="right", va="top",
                fontsize=8, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgrey"))
        ax.set_title(feat.replace("_", " "), fontsize=9, fontweight="bold")
        ax.set_xlabel("F1-macro")
        ax.legend(fontsize=7)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.suptitle(f"Test de permutation par feature\n(SVM RBF, {N_PERMUTATIONS} permutations, CV {N_FOLDS}-fold)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(out_dir, "permutation_test_features.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" {out}")


def plot_summary(df_res, baseline_f1_cv, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    colors = []
    for _, row in df_res.iterrows():
        sig = row["significatif"] in (True, "OK")
        if sig:
            colors.append("#2ecc71" if row["f1_cv_mean"] >= baseline_f1_cv else "#e67e22")
        else:
            colors.append("#e74c3c")

    bars = axes[0].barh(df_res["feature"], df_res["f1_cv_mean"], xerr=df_res["f1_cv_std"],
                        color=colors, edgecolor="white", linewidth=0.8,
                        error_kw={"elinewidth": 1.2, "capsize": 3})
    axes[0].axvline(baseline_f1_cv, color="navy", linestyle="--", linewidth=1.5,
                    label=f"Baseline CV (toutes) = {baseline_f1_cv:.3f}")

    for bar, (_, row) in zip(bars, df_res.iterrows()):
        sig = row["significatif"] in (True, "OK")
        label = f"p={row['p_value']:.3f} {'✓' if sig else '✗'}"
        axes[0].text(bar.get_width() + df_res["f1_cv_mean"].max() * 0.01,
                     bar.get_y() + bar.get_height() / 2,
                     label, va="center", fontsize=7.5,
                     color="#27ae60" if sig else "#e74c3c", fontweight="bold")

    axes[0].set_xlabel("F1-macro")
    axes[0].set_title("F1-macro CV par feature\nVert = sig. ≥ baseline  |  Orange = sig. < baseline  |  Rouge = non sig.",
                      fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, df_res["f1_cv_mean"].max() * 1.25)

    cols = ["accuracy", "precision", "recall", "f1_macro", "f1_cv_mean"]
    sns.heatmap(df_res.set_index("feature")[cols],
                annot=True, fmt=".3f", cmap="YlOrRd", ax=axes[1],
                linewidths=0.5, linecolor="lightgrey", vmin=0, vmax=1)
    axes[1].set_title("Métriques par feature (split test + CV)", fontsize=10)
    axes[1].tick_params(axis='y', rotation=0)
    axes[1].set_xticklabels(["Accuracy", "Précision", "Rappel", "F1 (split)", "F1 CV moy"],
                             rotation=30, ha="right")

    plt.suptitle("Évaluation indépendante des features — SVM RBF", fontsize=14, y=1.01)
    plt.tight_layout()
    out = os.path.join(out_dir, "features_independantes_v2.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" {out}")


# Main
def main():
    parser = argparse.ArgumentParser(description="Test indépendant des features — SVM RBF")
    parser.add_argument("-f", "--fichierCSV", required=True, help="chemin vers le fichier CSV")
    args = parser.parse_args()

    matrices_dir = os.path.join(OUTPUT_DIR, "matrices_confusion")
    os.makedirs(matrices_dir, exist_ok=True)

    X, y = load_data(args.fichierCSV)
    if X is None:
        return

    labels = sorted(y.unique())
    texts = X.astype(str).tolist()
    y_arr = y.values

    print("\nExtraction des features numériques...")
    X_num = extract_all_features(texts)
    X_num = np.nan_to_num(X_num, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  {X_num.shape[1]} features extraites")

    # split indices pour pouvoir les réutiliser sur les matrices sparse
    train_idx, test_idx = train_test_split(
        np.arange(len(texts)), test_size=0.2, random_state=42, stratify=y_arr
    )
    texts_train = [texts[i] for i in train_idx]
    y_train = y_arr[train_idx]
    y_test = y_arr[test_idx]
    X_num_train = X_num[train_idx]
    X_num_test = X_num[test_idx]
    print(f"Split 80/20 — train: {len(train_idx)}, test: {len(test_idx)}")

    print("\nConstruction des features TF-IDF...")
    char_vec, word_vec, pos_pipe = build_tfidf_pipelines()
    X_char, X_word, X_pos = fit_transform_tfidf(texts_train, texts, char_vec, word_vec, pos_pipe)
    print(f"  char={X_char.shape[1]}  word={X_word.shape[1]}  pos={X_pos.shape[1]}")

    X_char_train, X_char_test = X_char[train_idx], X_char[test_idx]
    X_word_train, X_word_test = X_word[train_idx], X_word[test_idx]
    X_pos_train, X_pos_test = X_pos[train_idx], X_pos[test_idx]

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # dictionnaire feature: (all, train, test, is_sparse)
    feat_map = {
        name: (X_num[:, i].reshape(-1,1), X_num_train[:, i].reshape(-1,1),
               X_num_test[:, i].reshape(-1,1), False)
        for i, name in enumerate(FEATURE_NAMES[:16])
    }
    feat_map.update({
        "tfidf_char": (X_char, X_char_train, X_char_test, True),
        "tfidf_word": (X_word, X_word_train, X_word_test, True),
        "tfidf_pos":  (X_pos,  X_pos_train,  X_pos_test,  True),
    })

    resultats    = []
    perm_results = {}
    reports      = {}

    print(f"\nEvaluation par feature (CV {N_FOLDS}-fold + {N_PERMUTATIONS} permutations)...")
    for name in FEATURE_NAMES:
        Xa, Xtr, Xte, sparse = feat_map[name]

        if sparse:
            pipe = Pipeline([("svm", SVC(kernel="rbf", C=1.0, class_weight="balanced"))])
            pipe_cv = Pipeline([("svm", SVC(kernel="rbf", C=1.0, class_weight="balanced"))])
        else:
            pipe = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", C=1.0, class_weight="balanced"))])
            pipe_cv = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", C=1.0, class_weight="balanced"))])

        pipe.fit(Xtr, y_train)
        y_pred = pipe.predict(Xte)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1  = f1_score(y_test,  y_pred, average="macro", zero_division=0)
        reports[name] = classification_report(y_test, y_pred, labels=labels, zero_division=0)

        cv_scores = cross_val_score(pipe_cv, Xa, y_arr, cv=cv, scoring="f1_macro", n_jobs=-1)
        f1_cv_mean = cv_scores.mean()
        f1_cv_std = cv_scores.std()

        print(f"  [{name:<30}] split={f1:.3f} | cv={f1_cv_mean:.3f}±{f1_cv_std:.3f}",
              end="  perm... ", flush=True)
        perm = permutation_test_feature(Xa, y_arr, cv, f1_cv_mean)
        perm_results[name] = perm
        print(f"p={perm['p_value']:.3f}  {'OK' if perm['significatif'] else 'NOPE'}")

        resultats.append({
            "feature": name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1_macro": f1,
            "f1_cv_mean": f1_cv_mean, "f1_cv_std": f1_cv_std,
            "p_value": perm["p_value"], "significatif": perm["significatif"],
        })
        plot_confusion_matrix(y_test, y_pred, labels, name, matrices_dir)

    df_res = pd.DataFrame(resultats).sort_values("f1_cv_mean", ascending=False)

    # baseline : toutes les features combinées
    print("\nBaseline toutes features...")
    X_all_combined = hstack([csr_matrix(X_num), X_char, X_word,   X_pos])
    X_train_combined = hstack([csr_matrix(X_num_train), X_char_train, X_word_train, X_pos_train])
    X_test_combined = hstack([csr_matrix(X_num_test),  X_char_test,  X_word_test,  X_pos_test])

    pipe_all = Pipeline([("svm", SVC(kernel="rbf", C=1.0, class_weight="balanced"))])
    pipe_all.fit(X_train_combined, y_train)
    y_pred_all = pipe_all.predict(X_test_combined)
    baseline_acc = accuracy_score(y_test, y_pred_all)
    baseline_f1= f1_score(y_test, y_pred_all, average="macro", zero_division=0)
    cv_all  = cross_val_score(pipe_all, X_all_combined, y_arr, cv=cv, scoring="f1_macro", n_jobs=-1)
    baseline_f1_cv = cv_all.mean()
    print(f"split: acc={baseline_acc:.3f}  f1={baseline_f1:.3f}")
    print(f"CV: f1={baseline_f1_cv:.3f} ± {cv_all.std():.3f}")

    reports["baseline_toutes_features"] = classification_report(y_test, y_pred_all, labels=labels, zero_division=0)
    plot_confusion_matrix(y_test, y_pred_all, labels, "baseline_toutes_features", matrices_dir)

    # sauvegarde des rapports de classification
    print("\nSauvegarde des rapports...")
    for fname, rpt in reports.items():
        path = os.path.join(OUTPUT_DIR, f"rapport_{fname.replace(' ', '_')}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Rapport de classification — {fname}\n{'='*60}\n\n{rpt}")
        print(f" {path}")

    recap_path = os.path.join(OUTPUT_DIR, "rapport_recapitulatif.txt")
    with open(recap_path, "w", encoding="utf-8") as f:
        df_print = df_res.copy()
        df_print["significatif"] = df_print["significatif"].map({True: "OK", False: "NOPE"})
        f.write("Récapitulatif — features triées par F1-macro CV\n" + "="*75 + "\n\n")
        f.write(df_print[["feature","f1_cv_mean","f1_cv_std","f1_macro","p_value","significatif"]].to_string(index=False))
        f.write(f"\n\nBaseline CV : {baseline_f1_cv:.3f} ± {cv_all.std():.3f}\n")
        f.write(f"Baseline split : acc={baseline_acc:.3f}  f1={baseline_f1:.3f}\n")
        f.write("\n\n" + "="*75 + "\nRapport par feature\n" + "="*75 + "\n")
        for fname, rpt in reports.items():
            f.write(f"\n{'─'*60}\n{fname}\n{'─'*60}\n{rpt}")
    print(f" {recap_path}")

    print("\nGraphiques...")
    plot_summary(df_res, baseline_f1_cv, OUTPUT_DIR)
    plot_permutation_histograms(perm_results, OUTPUT_DIR)

    print("Récapitulatif")
    df_print = df_res.copy()
    df_print["significatif"] = df_print["significatif"].map({True: "OK", False: "NOPE"})
    print(df_print[["feature","f1_cv_mean","f1_cv_std","f1_macro","p_value","significatif"]].to_string(index=False))
    print(f"\nBaseline CV : {baseline_f1_cv:.3f} ± {cv_all.std():.3f}")
    print(f"\nFichiers dans : {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()