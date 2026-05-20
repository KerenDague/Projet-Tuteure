"""
Classification de texte avec différents modèles BERT.

Ce script permet de prédire la langue maternelle (L1) d'un locuteur à partir d'un texte écrit en français (L2) à l'aide d'un Transformer BERT.

Modèles disponibles à choisir en argument -m : flaubert, camembert ou roberta

Il inclut les étapes suivantes :
1. Chargement et préparation des données textuelles depuis un fichier CSV
2. Construction et préparation du modèle BERT séléctionné par l'utilisateur'
3. Encodage des textes
4. Entraînement du modèle sur l’ensemble d’entraînement
5. Prédiction sur l’ensemble de test
6. Préparation de la matrice de confusion et évaluation des performances (accuracy + classification report)
"""

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os

# Configuration :
TEXT_COLUMN = 'Texte'
LABEL_COLUMN = 'Langue'

MODEL_MAP = {
    "flaubert": "flaubert/flaubert_base_cased",
    "camembert": "camembert-base",
    "roberta": "xlm-roberta-base"  # multilingue
}

# Paramètres par modèle (valeurs recommandées pour fine-tuning sur petites données)
MODEL_PARAMS = {
    "flaubert": {"lr": 2e-5, "epochs": 5, "max_length": 256, "batch_size": 8},
    "camembert": {"lr": 3e-5, "epochs": 6, "max_length": 384, "batch_size": 8},
    "roberta": {"lr": 1e-5, "epochs": 6, "max_length": 384, "batch_size": 8},
}

# 1. Chargement des données
def load_data(file_path, text_col, label_col):
    try:
        df = pd.read_csv(file_path, sep = ";")
    except FileNotFoundError:
        print(f"ERREUR : Le fichier '{file_path}' n'a pas été trouvé.")
        return None, None
    except Exception as e:
        print(f"ERREUR lecture CSV : {e}")
        return None, None

    df = df.dropna(subset=[text_col, label_col])
    X = df[text_col].astype(str)
    y = df[label_col].astype(str)
    return X, y

    print(f"Données chargées : {len(df)} échantillons.")

# 2. Construction du modèle
def build_model_and_tokenizer(model_choice, num_labels):
    model_name = MODEL_MAP[model_choice]
    # pour Flaubert, si sacremoses absent, tokenizer demandera l'install
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        # message explicite pour sacremoses
        msg = str(e)
        if "sacremoses" in msg or "MosesTokenizer" in msg or "FlaubertTokenizer" in msg:
            print("ImportError lié à sacremoses ou tokeniser Moses détecté.")
            print("Installe sacremoses: pip install sacremoses (ou conda install -c conda-forge sacremoses)")
            raise
        else:
            raise

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Modèle sélectionné : {model_name} -> device: {device}")

    # Paramètres optimiseur réglés suivant le modèle
    params = MODEL_PARAMS.get(model_choice, {"lr": 2e-5})
    optimizer = AdamW(model.parameters(), lr=params["lr"])

    # Pour XLM-R (ou autres) : réinitialiser la tête de classification (utile sur petits jeux)
    try:
        # tentative générique : reset si attribut classifier présent
        if hasattr(model, "classifier"):
            model.classifier.reset_parameters()
            print("Réinitialisation des paramètres de la tête 'classifier'.")
        elif hasattr(model, "score"):
            # improbable, mais safe-guard
            pass
    except Exception:
        # si reset non supportée, on ignore proprement
        pass

    return tokenizer, model, optimizer, device

# 3. Encodage des textes
def encode_batch(tokenizer, texts, labels=None, max_length=256, device="cpu"):
    # Important : NE PAS lower() ni supprimer ponctuation pour Flaubert/XLM-R
    enc = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    if labels is not None:
        # labels is a list/Series of integer ids
        enc['labels'] = torch.tensor(labels, dtype=torch.long, device=device)
    return enc

# 4. Entraînement
def train_model(tokenizer, model, optimizer, device, X_train, y_train, class_weights=None,
                epochs=3, batch_size=8, max_length=256):
    model.train()
    n = len(X_train)
    print(f"Train set size: {n}, epochs: {epochs}, batch_size: {batch_size}, max_length: {max_length}")

    # loss function (with class weights si fournis)
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
        use_custom_loss = True
        print("Utilisation de CrossEntropyLoss avec class weights.")
    else:
        loss_fn = None
        use_custom_loss = False

    for epoch in range(epochs):
        print(f"\n--- Époque {epoch+1}/{epochs} ---")
        indices = np.random.permutation(n)

        for i in range(0, n, batch_size):
            batch_idx = indices[i:i+batch_size]
            texts = X_train.iloc[batch_idx].tolist()
            labels = y_train.iloc[batch_idx].tolist()

            enc = encode_batch(tokenizer, texts, labels=None, max_length=max_length, device=device)
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

            # forward
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            logits = outputs.logits  # shape (batch_size, num_labels)

            if use_custom_loss:
                loss = loss_fn(logits, labels_tensor)
            else:
                # utiliser la loss interne du modèle en fournissant labels dans l'appel
                outputs_with_labels = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_tensor, return_dict=True)
                loss = outputs_with_labels.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_num = i // batch_size
            print(f"Batch {batch_num} — Loss : {loss.item():.4f}")

    return model

# 5. Prédiction
def predict_model(tokenizer, model, device, texts, batch_size=16, max_length=256):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            enc = encode_batch(tokenizer, batch_texts, labels=None, max_length=max_length, device=device)
            outputs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], return_dict=True)
            batch_preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
            preds.extend(batch_preds)
    return preds

# 6. Matrice de confusion et évaluation
def plot_confusion_matrix(y_true, y_pred, labels, filename, model_name):
    print("Génération de la matrice de confusion visuelle...")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Oranges", linewidths=.5)
    plt.title(f"Matrice de Confusion - Bert {model_name}", fontsize=16)
    plt.ylabel("Vraie Langue")
    plt.xlabel("Langue Prédite")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Matrice enregistrée sous : {filename}")

# Main
def main():
    parser = argparse.ArgumentParser(description="Classification BERT")
    parser.add_argument("-f", "--fichierCSV", required=True, help="Nom du fichier CSV (doit contenir colonnes Texte, Langue)")
    parser.add_argument("-m", "--modele", choices=["flaubert", "camembert", "roberta"], default="camembert",
                        help="Modèle à utiliser : flaubert, camembert, roberta (xlm-roberta pour 'roberta')")
    parser.add_argument("--max_length", type=int, default=None, help="Max tokens (optionnel, remplace le défaut du modèle)")
    parser.add_argument("--epochs", type=int, default=None, help="Nombre d'époques (optionnel, remplace le défaut)")
    parser.add_argument("--batch_size", type=int, default=None, help="Taille de batch (optionnel, remplace le défaut)")
    args = parser.parse_args()

    FILE_PATH = args.fichierCSV
    MODEL_CHOICE = args.modele

    X, y = load_data(FILE_PATH, TEXT_COLUMN, LABEL_COLUMN)
    if X is None:
        return

    labels = sorted(y.unique().tolist())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    y_num = y.map(label2id)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_num, test_size=0.15, random_state=42, stratify=y_num
    )

    print(f"Taille de l'entraînement : {len(X_train)}, Taille du test : {len(X_test)}")
    print("-" * 30)

    params = MODEL_PARAMS.get(MODEL_CHOICE, {})
    max_length = args.max_length if args.max_length is not None else params.get("max_length", 256)
    epochs = args.epochs if args.epochs is not None else params.get("epochs", 3)
    batch_size = args.batch_size if args.batch_size is not None else params.get("batch_size", 8)
    lr = params.get("lr", 2e-5)

    try:
        tokenizer, model, optimizer, device = build_model_and_tokenizer(MODEL_CHOICE, num_labels=len(labels))
    except Exception as e:
        print("Erreur lors de la création du tokenizer/modèle :", e)
        return


    for g in optimizer.param_groups:
        g["lr"] = lr


    class_weights = None # activable et utilise si déséquilibre dans les classes
    try:
        y_train_list = y_train.tolist()
        classes = np.unique(y_train_list)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_list)
        # compute_class_weight returns weights aligning with classes array; remap to full range
        # classes are label ids, but compute_class_weight returns in order of classes array
        class_weights = np.zeros(len(labels), dtype=float)
        for cls_idx, w in zip(classes, weights):
            class_weights[int(cls_idx)] = w
        print("Class weights calculés :", class_weights)
    except Exception as e:
        print("Impossible de calculer les class weights :", e)
        class_weights = None

    print("Début de l'entraînement...")
    start_time = time.time()
    model = train_model(
        tokenizer, model, optimizer, device,
        X_train, y_train,      # <-- garder Series
        class_weights=class_weights,
        epochs=epochs, batch_size=batch_size, max_length=max_length
    )
    elapsed = time.time() - start_time
    print(f"Entraînement terminé en {elapsed:.2f} secondes.")

    print("\nPrédiction...")
    y_pred_num = predict_model(tokenizer, model, device, X_test.tolist(), batch_size= max(8, batch_size), max_length=max_length)
    y_pred = [id2label[idx] for idx in y_pred_num]
    y_test_labels = [id2label[idx] for idx in y_test.tolist()]

    accuracy = accuracy_score(y_test_labels, y_pred)
    print(f"\nAccuracy : {accuracy * 100:.2f}%\n")
    print(classification_report(y_test_labels, y_pred, digits=3))

    # 7. Matrice de confusion + sauvegarde (nom contient modèle et timestamp)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_image_name = f"matrice_confusion_{MODEL_CHOICE}.png"
    plot_confusion_matrix(y_test_labels, y_pred, labels, output_image_name, MODEL_CHOICE)

if __name__ == "__main__":
    main()
