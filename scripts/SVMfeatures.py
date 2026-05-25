import matplotlib
matplotlib.use("Agg")
import re
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import time
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import (train_test_split, StratifiedKFold,cross_val_score, GridSearchCV)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score, confusion_matrix, classification_report)
from sklearn.utils.validation import check_is_fitted



TEXT_COLUMN  = 'Texte'
LABEL_COLUMN = 'Langue'

# Connecteurs et pronoms
CONNECTEURS = [
    "et", "mais", "ou", "donc", "or", "ni", "car",
    "cependant", "pourtant", "néanmoins", "toutefois", "en revanche", "au contraire",
    "parce que", "puisque", "étant donné", "vu que",
    "ainsi", "par conséquent", "de ce fait", "c'est pourquoi",
    "de plus", "en outre", "par ailleurs", "également",
    "bien que", "quoique", "même si", "malgré", "certes",
]
PRONOMS_SUJETS = {"je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles"}

SPACY_MODEL = "fr_core_news_md"

  
# Utilitaires
def count_syllables(word: str) -> int:
    return max(1, len(re.findall(r'[aeiouyàâéèêëîïôùûü]+', word.lower())))


def tree_depth(token) -> int:
    depth = 0
    t = token
    while t.head != t:
        depth += 1
        t = t.head
    return depth

# Extracteurs sklearn
class POSExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, spacy_model=SPACY_MODEL):
        self.spacy_model = spacy_model

    def fit(self, X, y=None):
        self.nlp_ = spacy.load(self.spacy_model)
        self.fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self)
        pos_strings = []
        for doc in self.nlp_.pipe(X, disable=["ner"]):
            seq = " ".join([token.pos_ for token in doc])
            pos_strings.append(seq)
        return np.array(pos_strings)


class StylometricExtractor(BaseEstimator, TransformerMixin):
    """
    Features stylo basiques (longueur, ponctuation, diversité lex.)
    """

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self)
        features = []
        for text in X:
            length = len(text)
            punct_ratio = sum(1 for c in text if c in ".,;:!?") / max(1, length)
            tokens = text.split()
            lexdiv = len(set(tokens)) / max(1, len(tokens))
            features.append([length, punct_ratio, lexdiv])
        return np.array(features)


class L1NumericExtractor(BaseEstimator, TransformerMixin):
    """
    Retourne uniquement les features numériques de L1 transfer.
    """

    def __init__(self, spacy_model=SPACY_MODEL):
        self.spacy_model = spacy_model

    def fit(self, X, y=None):
        self.nlp_ = spacy.load(self.spacy_model)
        self.fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self)
        numeric = []
        for doc in self.nlp_.pipe(X, disable=["ner"]):
            det_count = noun_count = 0
            verb_positions = []
            subj_positions = []
            obj_positions= []
            plural_count = 0

            for i, token in enumerate(doc):
                if token.pos_ == "DET":  det_count += 1
                if token.pos_ == "NOUN": noun_count += 1
                if token.pos_ == "VERB": verb_positions.append(i)
                if token.dep_ == "nsubj": subj_positions.append(i)
                if token.dep_ == "obj":   obj_positions.append(i)
                if token.tag_ == "NOUN__Number=Plur": plural_count += 1

            det_ratio = det_count / max(1, noun_count)
            subj_mean= np.mean(subj_positions) if subj_positions else 0.0
            verb_mean = np.mean(verb_positions) if verb_positions else 0.0
            obj_mean = np.mean(obj_positions)  if obj_positions  else 0.0
            numeric.append([det_ratio, subj_mean, verb_mean, obj_mean, plural_count])

        return np.array(numeric, dtype=float)


class L1DepExtractor(BaseEstimator, TransformerMixin):
    """
    Retourne uniquement la séquence de dépendances (string) de L1 transfer.
    """

    def __init__(self, spacy_model=SPACY_MODEL):
        self.spacy_model = spacy_model

    def fit(self, X, y=None):
        self.nlp_ = spacy.load(self.spacy_model)
        self.fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self)
        deps = []
        for doc in self.nlp_.pipe(X, disable=["ner"]):
            deps.append(" ".join(token.dep_ for token in doc))
        return np.array(deps)


# Extracteur features linguistiques 
class AdvancedLinguisticExtractor(BaseEstimator, TransformerMixin):
    """
    Extrait 14 features linguistiques :
    lisibilité Flesch, longueur moyenne des phrases, complexité syntaxique,
    fréquence prépositions, répétition noms, connecteurs, ratio déterminants,
    majuscules hors début, ratio SOV, ratio verbes/noms, ratio passif,
    distance verbe-sujet, ratio pro-drop.
    """
    def __init__(self, spacy_model=SPACY_MODEL):
        self.spacy_model = spacy_model

    def fit(self, X, y=None):
        self.nlp_ = spacy.load(self.spacy_model)
        self.fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self)
        rows = []
        for doc in self.nlp_.pipe(X, disable=["ner"]):
            tokens = [t for t in doc if not t.is_space]
            nb_tokens  = max(1, len(tokens))
            text_low = doc.text.lower()
            phrases = list(doc.sents)
            nb_phrases = max(1, len(phrases))

            # 1. Ponctuation
            punct_ratio = sum(1 for t in tokens if t.text in ".,;:!?") / nb_tokens

            # 2. Lisibilité Flesch (adapté français)
            mots_alpha  = [t.text for t in tokens if t.is_alpha]
            nb_mots = max(1, len(mots_alpha))
            nb_syllabes = sum(count_syllables(m) for m in mots_alpha)
            lisibilite = (206.835 - 1.015 * (nb_mots / nb_phrases) - 84.6  * (nb_syllabes / nb_mots))

            # 3. Longueur moyenne des phrases
            nb_mots_moy = float(np.mean([len([t for t in s if not t.is_space]) for s in phrases]))

            # 4. Complexité syntaxique
            depths = [tree_depth(t) for t in tokens if t.pos_ in ("NOUN", "VERB", "ADJ")]
            profondeur_moy = float(np.mean(depths)) if depths else 0.0
            ratio_sconj  = sum(1 for t in tokens if t.pos_ == "SCONJ") / nb_phrases
            complexite_syntaxique = profondeur_moy + ratio_sconj

            # 5. Fréquence des prépositions
            freq_prep = sum(1 for t in tokens if t.pos_ == "ADP") / nb_tokens

            # 6. répétition des noms
            noun_lemmas = [t.lemma_.lower() for t in tokens if t.pos_ == "NOUN"]
            if noun_lemmas:
                noun_counter = Counter(noun_lemmas)
                nb_uniq_nouns = len(noun_counter)
                nb_repetes = sum(1 for c in noun_counter.values() if c >= 2)
                repetition_noms = nb_repetes / nb_uniq_nouns
            else:
                repetition_noms = 0.0

            # 7. Fréquence des connecteurs
            nb_conn = sum(len(re.findall(r'\b' + re.escape(c) + r'\b', text_low))for c in CONNECTEURS)
            freq_conn = nb_conn / nb_tokens

            # 8. Ratio déterminant / nom
            det_count = sum(1 for t in tokens if t.pos_ == "DET")
            noun_count = max(1, sum(1 for t in tokens if t.pos_ == "NOUN"))
            det_ratio = det_count / noun_count

            # 9. Majuscules hors début de phrase
            nb_maj = 0
            for sent in doc.sents:
                sent_toks = [t for t in sent if not t.is_space]
                for t in sent_toks[1:]:
                    if t.text and t.text[0].isupper() and t.pos_ != "PROPN":
                        nb_maj += 1
            freq_maj = nb_maj / nb_tokens

            # 10 Ratio SOV
            scores_ordre = []
            for sent in doc.sents:
                sent_toks = list(sent)
                pos_map = {t.i: idx for idx, t in enumerate(sent_toks)}
                for t in sent_toks:
                    if t.pos_ == "VERB":
                        children = list(t.children)
                        subj = next((c for c in children if c.dep_ == "nsubj"), None)
                        obj = next((c for c in children if c.dep_ == "obj"),   None)
                        if subj and obj:
                            v = pos_map.get(t.i, 0)
                            s = pos_map.get(subj.i, 0)
                            o = pos_map.get(obj.i, 0)
                            if o < v:   scores_ordre.append(1)
                            elif s > v: scores_ordre.append(-1)
                            else:       scores_ordre.append(0)
            ratio_sov = float(np.mean(scores_ordre)) if scores_ordre else 0.0

            # 11. Ratio verbes / noms
            nb_verbes_seuls = sum(1 for t in tokens if t.pos_ == "VERB")
            ratio_verbes_noms = nb_verbes_seuls / noun_count

            # 12. Ratio passif
            nb_passif = 0
            for t in tokens:
                if t.dep_ == "auxpass":
                    nb_passif += 1
                elif (t.pos_ == "AUX" and t.lemma_ == "être"
                      and any(c.dep_ == "nsubj:pass" for c in t.head.children)):
                    nb_passif += 1
            ratio_passif = nb_passif / nb_phrases

            # 13. Distance verbe-sujet
            distances = []
            for t in tokens:
                if t.pos_ == "VERB":
                    subj = next( (c for c in t.children if c.dep_ in ("nsubj", "nsubj:pass")), None)
                    if subj:
                        distances.append(abs(t.i - subj.i))
            distance_verbe_sujet = float(np.mean(distances)) if distances else 0.0

            # 14. Ratio pro-drop
            verbes_conjugues = [
                t for t in tokens
                if t.pos_ in ("VERB", "AUX")
                and t.morph.get("VerbForm") not in (["Inf"], ["Part"])
            ]
            nb_verbes_conj = max(1, len(verbes_conjugues))
            nb_pron_sujet  = sum(
                1 for t in tokens
                if t.text.lower() in PRONOMS_SUJETS and t.dep_ == "nsubj"
            )
            ratio_prodrop = nb_pron_sujet / nb_verbes_conj

            rows.append([
                punct_ratio, lisibilite, nb_mots_moy, complexite_syntaxique,
                freq_prep, repetition_noms, freq_conn, det_ratio, freq_maj,
                ratio_sov, ratio_verbes_noms, ratio_passif,
                distance_verbe_sujet, ratio_prodrop,
            ])

        result = np.array(rows, dtype=float)
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


# Chargement des données
def load_data(file_path: str, text_col: str, label_col: str):
    try:
        try:
            df = pd.read_csv(file_path, sep=';')
            if df.shape[1] < 2:
                df = pd.read_csv(file_path, sep=',')
        except Exception:
            df = pd.read_csv(file_path, sep=None, engine='python')
    except FileNotFoundError:
        print(f"Le fichier '{file_path}' n'a pas été trouvé.")
        return None, None

    df = df.dropna(subset=[text_col, label_col])
    print(f"Données chargées : {len(df)} échantillons.")
    print(f"Nombre de classes : {df[label_col].nunique()}")
    print(df[label_col].value_counts().to_string())
    return df[text_col], df[label_col]



# Construction du pipeline
def build_svm_pipeline():
    char_vectorizer = TfidfVectorizer(
        analyzer='char', ngram_range=(3, 6), max_features=50000, sublinear_tf=True)
    word_vectorizer = TfidfVectorizer(
        analyzer='word', ngram_range=(1, 2), max_features=30000, sublinear_tf=True)

    pos_pipeline = Pipeline([
        ("pos_extract", POSExtractor()),
        ("pos_tfidf",   TfidfVectorizer()),
    ])

    l1_pipeline_numeric = Pipeline([
        ("l1_numeric", L1NumericExtractor()),
    ])

    l1_pipeline_dep = Pipeline([
        ("l1_dep",      L1DepExtractor()),
        ("dep_tfidf",   TfidfVectorizer()),
    ])

    advanced_pipeline = Pipeline([
        ("adv_extract", AdvancedLinguisticExtractor()),
    ])

    vectorizer = ColumnTransformer([
        ('char_tfidf', char_vectorizer,TEXT_COLUMN),
        ('word_tfidf', word_vectorizer, TEXT_COLUMN),
        ('pos', pos_pipeline, TEXT_COLUMN),
        ('stylometric', StylometricExtractor(), TEXT_COLUMN),
        ('l1_numeric', l1_pipeline_numeric,TEXT_COLUMN),
        ('l1_dep', l1_pipeline_dep,TEXT_COLUMN),
        ('advanced',advanced_pipeline,TEXT_COLUMN),
    ], transformer_weights={
        'char_tfidf': 1.0,
        'word_tfidf': 1.0,
        'pos': 0.6,
        'stylometric': 0.3,
        'l1_numeric': 0.8,
        'l1_dep':1.0,
        'advanced':0.9,
    })

    classifier = SVC(kernel='linear', C=1.0, class_weight='balanced')

    pipeline = Pipeline([
        ('features', vectorizer),
        ('svm', classifier),
    ])

    return pipeline


# Visualisations
def plot_confusion_matrix(y_true, y_pred, labels, output_dir):
    cm  = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(max(8, len(labels) + 2), max(7, len(labels) + 1)))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues',linewidths=.5, annot_kws={"size": 11})
    plt.title('Matrice de confusion — SVM', fontsize=13)
    plt.ylabel('Vraie langue', fontsize=11)
    plt.xlabel('Langue prédite', fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrice.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"{path}")


def plot_metrics_par_langue(df_metrics: pd.DataFrame, output_dir: str):
    df_plot = df_metrics[df_metrics["langue"] != "GLOBAL"].copy()
    x  = np.arange(len(df_plot))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(df_plot) * 1.4), 6))
    ax.bar(x - width, df_plot["precision"], width, label="Précision", color="#3498db", edgecolor="white")
    ax.bar(x,  df_plot["recall"], width, label="Rappel", color="#2ecc71", edgecolor="white")
    ax.bar(x + width, df_plot["f1"], width, label="F1-score", color="#e67e22", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["langue"], rotation=30, ha="right", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Précision, Rappel et F1-score par langue maternelle", fontsize=13)
    ax.legend(fontsize=10)
    global_f1 = df_metrics.loc[df_metrics["langue"] == "GLOBAL", "f1"].values[0]
    ax.axhline(global_f1, color="navy", linestyle="--", linewidth=1.2, label="F1-macro global")
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f", fontsize=7.5, padding=2)
    plt.tight_layout()
    path = os.path.join(output_dir, "metrics_par_langue.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" {path}")


def plot_metrics_heatmap(df_metrics: pd.DataFrame, output_dir: str):
    df_plot = df_metrics[df_metrics["langue"] != "GLOBAL"].set_index("langue")
    fig, ax = plt.subplots(figsize=(7, max(4, len(df_plot) + 1)))
    sns.heatmap(df_plot[["precision", "recall", "f1"]],
                annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0, vmax=1, linewidths=0.5, linecolor="lightgrey",
                ax=ax, annot_kws={"size": 11})
    ax.set_title("Métriques par langue — SVM", fontsize=13)
    ax.set_xticklabels(["Précision", "Rappel", "F1-score"], rotation=0)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    path = os.path.join(output_dir, "heatmap_metrics_langues.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" {path}")


# Main
def main():
    parser = argparse.ArgumentParser(description="SVM — identification de langue maternelle")
    parser.add_argument("-f", "--fichierCSV", required=True,help="Chemin vers le fichier CSV")
    parser.add_argument("--n_folds", type=int, default=5,help="Nombre de folds CV (défaut : 5)")
    parser.add_argument("--tune", action="store_true",help="Active GridSearchCV sur C (plus lent)")
    args = parser.parse_args()

    # Chargement 
    X, y = load_data(args.fichierCSV, TEXT_COLUMN, LABEL_COLUMN)
    if X is None:
        return

    labels   = sorted(y.unique())
    df_train = pd.DataFrame({TEXT_COLUMN: X, LABEL_COLUMN: y})

    X_train, X_test, y_train, y_test = train_test_split(
        df_train[[TEXT_COLUMN]], df_train[LABEL_COLUMN],
        test_size=0.2, random_state=42, stratify=y
    )
    print(f"Split 80/20 : Train : {len(X_train)}  |  Test : {len(X_test)}")

    os.makedirs("resultats", exist_ok=True)

    #Construction et entrainement 
    svm_model = build_svm_pipeline()

    if args.tune:
        print("\nGridSearchCV en cours (kernel=linear, C)...")
        param_grid = {"svm__C": [0.01, 0.1, 1, 10, 100]}
        cv_strat = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
        grid = GridSearchCV(svm_model, param_grid, cv=cv_strat,
                            scoring="f1_macro", n_jobs=1, verbose=1)
        start = time.time()
        grid.fit(X_train, y_train)
        print(f"Meilleur C : {grid.best_params_['svm__C']}  "
              f"(F1-macro CV : {grid.best_score_:.3f})")
        print(f"Durée : {time.time() - start:.1f}s")
        svm_model = grid.best_estimator_
    else:
        print("\nDébut de l'entrainement...")
        start = time.time()
        svm_model.fit(X_train, y_train)
        print(f"Entrainement terminé en {time.time() - start:.1f}s")

    #  Prédictions
    y_pred = svm_model.predict(X_test)

    # Metriques globales 
    acc  = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)

    # Cross-validation
    cv_strat  = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(svm_model, df_train[[TEXT_COLUMN]], df_train[LABEL_COLUMN], cv=cv_strat, scoring="f1_macro", n_jobs=1)

    #  Metriques par langue 
    precs = precision_score(y_test, y_pred, average=None, labels=labels, zero_division=0)
    recs = recall_score(y_test, y_pred, average=None, labels=labels, zero_division=0)
    f1s = f1_score(y_test, y_pred, average=None,  labels=labels, zero_division=0)
    supports = [np.sum(np.array(y_test) == l) for l in labels]

    rows = [{"langue": l, "precision": round(precs[i], 3),
             "recall": round(recs[i], 3), "f1": round(f1s[i], 3),
             "support": supports[i]}
            for i, l in enumerate(labels)]
    rows.append({"langue": "GLOBAL", "precision": round(prec_macro, 3),
                 "recall": round(rec_macro, 3), "f1": round(f1_macro, 3),
                 "support": len(y_test)})
    df_metrics = pd.DataFrame(rows)

    # Affichage console
    print("\n" + "═" * 60)
    print("PERFORMANCES DU MODÈLE SVM")
    print("═" * 60)
    print(f"Accuracy : {acc:.3f}")
    print(f"F1-macro (split): {f1_macro:.3f}")
    print(f"F1-macro (CV {args.n_folds}-fold) : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"F1-weighted : {f1_weighted:.3f}")
    print(f"Précision macro  : {prec_macro:.3f}")
    print(f"Rappel macro  : {rec_macro:.3f}")
    print()
    print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))
    print("\nDétail par langue :")
    print(df_metrics.to_string(index=False))

    #  Graphiques & CSV 
    print("\n── Génération des graphiques ──")
    plot_confusion_matrix(y_test, y_pred, labels, "resultats")
    plot_metrics_par_langue(df_metrics, "resultats")
    plot_metrics_heatmap(df_metrics, "resultats")
    df_metrics.to_csv("resultats/performances_par_langue.csv", index=False)

    print("\nFichiers générés dans 'resultats/' :")
    print("  · confusion_matrice.png")
    print("  · metrics_par_langue.png")
    print("  · heatmap_metrics_langues.png")
    print("  · performances_par_langue.csv")


if __name__ == "__main__":
    main()