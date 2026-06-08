import os
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import CamembertTokenizer, CamembertModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

from bert_features import TextFeatureExtractor, FEATURE_GROUP_COLUMNS

TEXT_COLUMN = "Texte"
LABEL_COLUMN = "Langue"
BERT_MODEL = "camembert-base"
BATCH_SIZE = 32
MAX_LENGTH = 256
N_PERMUTATIONS = 300

FEATURE_NAMES = list(FEATURE_GROUP_COLUMNS.keys())


def extract_bert_embeddings(texts, tokenizer, model, device):
    model.eval()
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            output = model(**encoded)
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        emb = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        all_embeddings.append(emb.cpu().numpy())
        print(f"  Embeddings : {min(i + BATCH_SIZE, len(texts))}/{len(texts)}", end="\r")
    print()
    return np.vstack(all_embeddings)


def load_data(file_path):
    try:
        df = pd.read_csv(file_path, sep=';')
    except FileNotFoundError:
        print(f"ERREUR : Le fichier '{file_path}' n'a pas ete trouve.")
        return None, None
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    print(f"Donnees chargees : {len(df)} echantillons.")
    print(f"Nombre de classes : {df[LABEL_COLUMN].nunique()}")
    return df[TEXT_COLUMN], df[LABEL_COLUMN]


def permutation_test(X_feat, y, cv, f1_observe, n_permutations=N_PERMUTATIONS, random_state=42):
    rng = np.random.default_rng(random_state)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)),
    ])
    null_scores = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        scores = cross_val_score(pipe, X_feat, y_perm, cv=cv, scoring="f1_macro", n_jobs=-1)
        null_scores.append(scores.mean())
    null_scores = np.array(null_scores)
    p_value = float(np.mean(null_scores >= f1_observe))
    return p_value


def main():
    parser = argparse.ArgumentParser(description="Apport des features sur CamemBERT - test de significativite")
    parser.add_argument("-f", "--fichierCSV", required=True, help="Chemin vers le fichier CSV")
    parser.add_argument("--n_perm", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    X, y = load_data(args.fichierCSV)
    if X is None:
        return

    texts = X.astype(str).tolist()
    y_arr = y.values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print("Chargement de CamemBERT...")
    tokenizer = CamembertTokenizer.from_pretrained(BERT_MODEL)
    model = CamembertModel.from_pretrained(BERT_MODEL).to(device)

    print("Extraction des embeddings CamemBERT (geles)...")
    X_bert = extract_bert_embeddings(texts, tokenizer, model, device)

    print("Extraction des features linguistiques...")
    extractor = TextFeatureExtractor(feature_names=FEATURE_NAMES, scale=False)
    X_feats_df = extractor.fit_transform(texts)
    col_names = extractor.column_names
    X_feats_df = pd.DataFrame(X_feats_df, columns=col_names)

    indices = np.arange(len(texts))
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices, y_arr, test_size=0.2, random_state=42, stratify=y_arr
    )

    cv = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    pipe_template = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)),
    ])

    # Baseline : CamemBERT seul
    print("Calcul baseline CamemBERT seul...")
    cv_scores_bert = cross_val_score(pipe_template, X_bert, y_arr, cv=cv, scoring="f1_macro", n_jobs=-1)
    baseline_f1_cv = cv_scores_bert.mean()
    pipe_bert = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0))])
    pipe_bert.fit(X_bert[idx_train], y_train)
    baseline_f1 = f1_score(y_test, pipe_bert.predict(X_bert[idx_test]), average="macro", zero_division=0)
    print(f"  Baseline CamemBERT seul : f1_cv={baseline_f1_cv:.3f}  f1_split={baseline_f1:.3f}")

    print(f"\nApport de chaque feature sur CamemBERT (CV {args.n_folds}-fold, {args.n_perm} permutations)")

    resultats = []
    for name in FEATURE_NAMES:
        cols = FEATURE_GROUP_COLUMNS[name]
        feat_vals = X_feats_df[cols].values

        scaler = StandardScaler()
        feat_all_scaled = scaler.fit_transform(feat_vals)
        X_combined_all = np.hstack([X_bert, feat_all_scaled])

        cv_scores = cross_val_score(pipe_template, X_combined_all, y_arr, cv=cv, scoring="f1_macro", n_jobs=-1)
        f1_cv_mean = cv_scores.mean()
        f1_cv_std = cv_scores.std()

        feat_train = scaler.fit_transform(feat_vals[idx_train])
        feat_test = scaler.transform(feat_vals[idx_test])
        X_train = np.hstack([X_bert[idx_train], feat_train])
        X_test = np.hstack([X_bert[idx_test], feat_test])

        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0))])
        pipe.fit(X_train, y_train)
        f1_macro = f1_score(y_test, pipe.predict(X_test), average="macro", zero_division=0)

        apport = round(f1_cv_mean - baseline_f1_cv, 4)

        print(f"  {name:<30} f1_cv={f1_cv_mean:.3f} apport={apport:+.4f} | perm...", end="", flush=True)
        p_value = permutation_test(X_combined_all, y_arr, cv, f1_cv_mean, n_permutations=args.n_perm)
        sig = "OUI" if p_value < 0.05 else "NON"
        print(f" p={p_value:.3f} {sig}")

        resultats.append({
            "feature": name,
            "f1_cv_mean": round(f1_cv_mean, 6),
            "f1_cv_std": round(f1_cv_std, 6),
            "f1_macro": round(f1_macro, 6),
            "apport_vs_bert": apport,
            "baseline_bert_cv": round(baseline_f1_cv, 6),
            "p_value": round(p_value, 2),
            "significatif": sig,
        })

    df_res = pd.DataFrame(resultats).sort_values("apport_vs_bert", ascending=False)
    print("\nRECAPITULATIF - trie par apport decroissant")
    print(df_res.to_string(index=False))
    df_res.to_csv("significativite_bert.csv", index=False)
    print(f"\nBaseline CamemBERT seul : f1_cv={baseline_f1_cv:.3f}")
    print("Sauvegarde : significativite_bert.csv")


if __name__ == "__main__":
    main()

