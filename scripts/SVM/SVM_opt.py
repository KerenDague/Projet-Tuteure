import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import time


TEXT_COLUMN = 'Texte'
LABEL_COLUMN = 'Langue'
OUTPUT_IMAGE_NAME = 'matrice_confusion_svm_opt.png'


def load_data(file_path, text_col, label_col):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"ERREUR : Le fichier '{file_path}' n'a pas été trouvé.")
        return None, None

    df = df.dropna(subset=[text_col, label_col])

    X = df[text_col]
    y = df[label_col]

    print(f"Données chargées : {len(df)} échantillons.")
    print(f"Nombre de classes (L1) : {y.nunique()}")
    return X, y


def build_svm_pipeline():
    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(1, 9),     # OPTIMISATION
        lowercase=True,
        sublinear_tf=True,      # OPTIMISATION 
        min_df=2                # OPTIMISATION 
    )

    classifier = LinearSVC(
        C=0.5,                   # OPTIMISATION 
        class_weight='balanced'
    )

    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('svm', classifier)
    ])
    return pipeline


def plot_confusion_matrix(y_true, y_pred, labels, filename):
    print(f"Génération de la matrice de confusion visuelle...")
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df,annot=True,fmt='d',cmap='Blues',linewidths=.5 )
    plt.title('Matrice de Confusion-SVM', fontsize=16)
    plt.ylabel('Vraie Langue (True Label)', fontsize=12)
    plt.xlabel('Langue Prédite (Predicted Label)', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    try:
        plt.savefig(filename)
        print(f"Matrice de confusion enregistrée sous : '{filename}'")
    except Exception as e:
        print(f"Erreur lors de l'enregistrement de l'image : {e}")

def main():

    parser = argparse.ArgumentParser(description="Choix d'une table CSV")
    parser.add_argument("-f", "--fichierCSV", help="Entrez le nom du fichier CSV")
    args = parser.parse_args()

    FILE_PATH = args.fichierCSV

    X, y = load_data(FILE_PATH, TEXT_COLUMN, LABEL_COLUMN)
    if X is None:
        return

    labels = sorted(y.unique())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2, # meilleurs résultats avec 0.20
        random_state=42,
        stratify=y
    )
    print(f"Taille de l'entraînement : {len(X_train)}, Taille du test : {len(X_test)}")
    print("-" * 30)

    svm_model = build_svm_pipeline()
    print("Début de l'entraînement...")
    start_time = time.time()
    svm_model.fit(X_train, y_train)
    print(f"Entraînement terminé en {time.time() - start_time:.2f} secondes.")

    print("Évaluation du modèle...")
    y_pred = svm_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy : {accuracy * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, digits=3))
    plot_confusion_matrix(y_test, y_pred, labels, OUTPUT_IMAGE_NAME)

if __name__ == "__main__":
    main()
