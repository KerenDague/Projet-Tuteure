"""
Classification L1 avec CamemBERT + MLP pour features linguistiques

Exemples d'execution :
    python run_camembert_mlp.py -f data.csv --features feature_a
    python run_camembert_mlp.py -f data.csv --features feature_b
    python run_camembert_mlp.py -f data.csv --features feature_a feature_b
    python run_camembert_mlp.py -f data.csv --features none
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer

from bert_features import TextFeatureExtractor, available_features


# Configuration :
TEXT_COLUMN = 'Texte'
LABEL_COLUMN = 'Langue'

MODEL_MAP = {
    "flaubert": "flaubert/flaubert_base_cased",
    "camembert": "camembert-base",
    "roberta": "xlm-roberta-base"  # multilingue
}

MODEL_PARAMS = {
    "flaubert": {"lr": 2e-5, "epochs": 5, "max_length": 256, "batch_size": 8},
    "camembert": {"lr": 3e-5, "epochs": 6, "max_length": 384, "batch_size": 8},
    "roberta": {"lr": 1e-5, "epochs": 6, "max_length": 384, "batch_size": 8},
}

# fix random seeds
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 1. Chargement des données
def load_data(file_path, text_col, label_col):
    try:
        df = pd.read_csv(file_path, sep=None, engine="python")
    except FileNotFoundError:
        print(f"ERREUR : Le fichier '{file_path}' n'a pas été trouvé.")
        return None, None
    except Exception as e:
        print(f"ERREUR lecture CSV : {e}")
        return None, None

    df = df.dropna(subset=[text_col, label_col])
    X = df[text_col].astype(str)
    y = df[label_col].astype(str)
    
    print(f"Données chargées : {len(df)} échantillons.")

    return X, y


def build_tokenizer(model_choice: str):
    model_name = MODEL_MAP[model_choice]
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as exc:
        msg = str(exc)
        if "sacremoses" in msg or "MosesTokenizer" in msg or "FlaubertTokenizer" in msg:
            print("ImportError lie a sacremoses ou tokenizer Moses detecte.")
            print("Installe sacremoses: pip install sacremoses")
        raise
    return tokenizer


# This class defines a model that combines a transformer with an optional MLP for features. The transformer processes the text input, while the MLP processes the numerical features. The outputs are concatenated and fed into a final classifier layer.
class TransformerWithFeatureMLP(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        feature_dim: int,
        feature_hidden_dim: int = 64,
        dropout: float = 0.2) -> None:
        super().__init__()

        self.transformer = AutoModel.from_pretrained(model_name) # name of the transformer model to use
        self.feature_dim = feature_dim # dimension of the input features (0 if no features)

        hidden_size = self.transformer.config.hidden_size
        transformer_dropout = getattr(self.transformer.config, "hidden_dropout_prob", dropout)

        self.text_head = nn.Sequential(
            nn.Dropout(transformer_dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(transformer_dropout),
        )

        if feature_dim > 0:
            self.feature_mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            classifier_input_dim = hidden_size + feature_hidden_dim
        else:
            self.feature_mlp = None
            classifier_input_dim = hidden_size

        self.classifier = nn.Linear(classifier_input_dim, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, features=None):
        transformer_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }
        if token_type_ids is not None:
            transformer_kwargs["token_type_ids"] = token_type_ids

        outputs = self.transformer(**transformer_kwargs)
        text_repr = outputs.last_hidden_state[:, 0, :]
        text_repr = self.text_head(text_repr)

        if self.feature_mlp is not None:
            if features is None:
                raise ValueError("features must be provided when feature_dim > 0")
            feature_repr = self.feature_mlp(features)
            text_repr = torch.cat([text_repr, feature_repr], dim=1)

        return self.classifier(text_repr)



def build_model_and_tokenizer(model_choice: str, num_labels: int, feature_dim: int, feature_hidden_dim: int):
    model_name = MODEL_MAP[model_choice]
    tokenizer = build_tokenizer(model_choice)
    model = TransformerWithFeatureMLP(
        model_name=model_name,
        num_labels=num_labels,
        feature_dim=feature_dim,
        feature_hidden_dim=feature_hidden_dim,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Modele selectionne : {model_name} -> device: {device}")
    print(f"Nombre de features numeriques : {feature_dim}")

    params = MODEL_PARAMS.get(model_choice, {"lr": 2e-5})
    optimizer = AdamW(model.parameters(), lr=params["lr"])
    return tokenizer, model, optimizer, device


def encode_batch(tokenizer, texts, max_length: int = 256, device: str = "cpu") -> dict[str, torch.Tensor]:
    enc = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in enc.items()}

def train_model(
    tokenizer,
    model,
    optimizer,
    device: str,
    x_train: pd.Series,
    y_train: pd.Series,
    train_features: np.ndarray,
    class_weights=None,
    epochs: int = 3,
    batch_size: int = 8,
    max_length: int = 256,
):
    model.train()
    n = len(x_train)
    print(f"Train set size: {n}, epochs: {epochs}, batch_size: {batch_size}, max_length: {max_length}")

    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
        print("Utilisation de CrossEntropyLoss avec class weights.")
    else:
        loss_fn = nn.CrossEntropyLoss()
        

    x_train = x_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    for epoch in range(epochs):
        print(f"\n--- Epoque {epoch + 1}/{epochs} ---")
        indices = np.random.permutation(n)
        running_loss = 0.0

        for i in range(0, n, batch_size):
            batch_idx = indices[i : i + batch_size]
            texts = x_train.iloc[batch_idx].tolist()
            labels = y_train.iloc[batch_idx].tolist()

            enc = encode_batch(tokenizer, texts, max_length=max_length, device=device)
            labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
            features_tensor = None
            if train_features.shape[1] > 0:
                features_tensor = torch.tensor(train_features[batch_idx], dtype=torch.float32, device=device)

            logits = model(
                        input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                        features=features_tensor
                    )
            loss = loss_fn(logits, labels_tensor)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            batch_num = i // batch_size
            print(f"Batch {batch_num} - Loss : {loss.item():.4f}")

        epoch_loss = running_loss / max(1, int(np.ceil(n / batch_size)))
        print(f"Loss moyenne epoch {epoch + 1}: {epoch_loss:.4f}")

    return model


def predict_model(
    tokenizer,
    model,
    device: str,
    texts: list[str],
    features: np.ndarray,
    batch_size: int = 16,
    max_length: int = 256) -> list[int]:

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = encode_batch(tokenizer, batch_texts, max_length=max_length, device=device)
            features_tensor = None
            if features.shape[1] > 0:
                features_tensor = torch.tensor(features[i : i + batch_size], dtype=torch.float32, device=device)

            logits = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    features=features_tensor
                )
            batch_preds = torch.argmax(logits, dim=1).cpu().tolist()
            preds.extend(batch_preds)
    return preds

# 6. Matrice de confusion et évaluation
def plot_confusion_matrix(y_true, y_pred, labels, filename, model_name, experiment_name):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Oranges", linewidths=0.5)
    plt.title(f"Matrice de Confusion - {model_name} - {experiment_name}")
    plt.ylabel("Vraie Langue")
    plt.xlabel("Langue Predite")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Matrice enregistree sous : {filename}")


def make_experiment_name(model_choice: str, feature_names: list[str]) -> str:
    if not feature_names:
        return f"{model_choice}_bert_only"
    return f"{model_choice}_{'+'.join(feature_names)}"


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Classification BERT + MLP features")
    parser.add_argument("-f", "--fichierCSV", required=True, help="CSV avec colonnes Texte, Langue")
    parser.add_argument(
        "-m",
        "--modele",
        choices=["flaubert", "camembert", "roberta"],
        default="camembert",
        help="Modele a utiliser : flaubert, camembert, roberta",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=["none"],
        help=f"Features a utiliser. Choix : {', '.join(available_features())}, none",
    )
    parser.add_argument("--results_dir", default="results", help="Dossier racine pour sauvegarder les resultats")
    parser.add_argument("--max_length", type=int, default=None, help="Max tokens")
    parser.add_argument("--epochs", type=int, default=None, help="Nombre d'epoques")
    parser.add_argument("--batch_size", type=int, default=None, help="Taille de batch")
    parser.add_argument("--feature_hidden_dim", type=int, default=64, help="Dimension cachee MLP features")
    parser.add_argument("--test_size", type=float, default=0.20, help="Proportion test")
    parser.add_argument("--seed", type=int, default=42, help="Seed reproductible")
    args = parser.parse_args()

    if args.features == ["none"]:
        feature_names = []
    else:
        feature_names = args.features

    set_seed(args.seed)

    file_path = args.fichierCSV
    model_choice = args.modele

    x, y = load_data(file_path, TEXT_COLUMN, LABEL_COLUMN)
    if x is None or y is None:
        return

    labels = sorted(y.unique().tolist())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    y_num = y.map(label2id)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y_num,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_num,
    )

    print(f"Taille de l'entrainement : {len(x_train)}, Taille du test : {len(x_test)}")
    print(f"Features selectionnees : {feature_names if feature_names else ['none']}")
    print("-" * 30)

    feature_extractor = TextFeatureExtractor(feature_names=feature_names, scale=True)
    train_features = feature_extractor.fit_transform(x_train.tolist())
    test_features = feature_extractor.transform(x_test.tolist())

    params = MODEL_PARAMS.get(model_choice, {})
    max_length = args.max_length if args.max_length is not None else params.get("max_length", 256)
    epochs = args.epochs if args.epochs is not None else params.get("epochs", 3)
    batch_size = args.batch_size if args.batch_size is not None else params.get("batch_size", 8)
    lr = params.get("lr", 2e-5)

    tokenizer, model, optimizer, device = build_model_and_tokenizer(
        model_choice,
        num_labels=len(labels),
        feature_dim=feature_extractor.n_features,
        feature_hidden_dim=args.feature_hidden_dim,
    )

    for group in optimizer.param_groups:
        group["lr"] = lr

    class_weights = None
    try:
        y_train_list = y_train.tolist()
        classes = np.unique(y_train_list)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_list)

        class_weights = np.zeros(len(labels), dtype=float)
        for cls_idx, weight in zip(classes, weights):
            class_weights[int(cls_idx)] = weight

        print("Class weights calcules :", class_weights)
    except Exception as exc:
        print("Impossible de calculer les class weights :", exc)

    experiment_name = make_experiment_name(model_choice, feature_names)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.results_dir) / experiment_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "csv": os.path.abspath(file_path),
        "model_choice": model_choice,
        "model_name": MODEL_MAP[model_choice],
        "features": feature_names,
        "feature_columns": feature_extractor.column_names,
        "feature_hidden_dim": args.feature_hidden_dim,
        "max_length": max_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "test_size": args.test_size,
        "seed": args.seed,
        "labels": labels,
        "label2id": label2id,
    }
    save_json(output_dir / "config.json", config)

    print("Debut de l'entrainement...")
    start_time = time.time()

    model = train_model(
        tokenizer,
        model,
        optimizer,
        device,
        x_train,
        y_train,
        train_features,
        class_weights=class_weights,
        epochs=epochs,
        batch_size=batch_size,
        max_length=max_length,
    )

    elapsed = time.time() - start_time
    print(f"Entrainement termine en {elapsed:.2f} secondes.")

    print("\nPrediction...")
    y_pred_num = predict_model(
        tokenizer,
        model,
        device,
        x_test.tolist(),
        test_features,
        batch_size=max(8, batch_size),
        max_length=max_length,
    )

    y_pred = [id2label[idx] for idx in y_pred_num]
    y_test_labels = [id2label[idx] for idx in y_test.tolist()]

    accuracy = accuracy_score(y_test_labels, y_pred)
    report_dict = classification_report(y_test_labels, y_pred, digits=3, output_dict=True)
    report_text = classification_report(y_test_labels, y_pred, digits=3)

    print(f"\nAccuracy : {accuracy * 100:.2f}%\n")
    print(report_text)

    metrics = {
        "accuracy": accuracy,
        "classification_report": report_dict,
        "elapsed_seconds": elapsed,
    }
    save_json(output_dir / "metrics.json", metrics)

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    predictions = pd.DataFrame(
        {
            "Texte": x_test.tolist(),
            "true_label": y_test_labels,
            "predicted_label": y_pred,
        }
    )
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    try:
        plot_confusion_matrix(
            y_test_labels,
            y_pred,
            labels,
            output_dir / "confusion_matrix.png",
            model_choice,
            experiment_name,
        )
    except Exception as exc:
        print("Erreur plot confusion matrix :", exc)

    print(f"Resultats sauvegardes dans : {output_dir}")

if __name__ == "__main__":
    main()
