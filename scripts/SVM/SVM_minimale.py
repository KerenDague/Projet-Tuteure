import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import sys

csv_path = sys.argv[1]

try:
    data = pd.read_csv(csv_path, sep=';')
except:
    data = pd.read_csv(csv_path, sep=',')

X = data['Texte']
y = data['Langue']

vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_vectorized,y,test_size=0.2, random_state=42)
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy:.4f}")

report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)
with open("classification_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Accuracy : {accuracy:.4f}\n\n")
    f.write(report)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=svm_model.classes_,yticklabels=svm_model.classes_)
plt.xlabel("Prédictions")
plt.ylabel("Vraies classes")
plt.title("Matrice de confusion - SVM Baseline")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()
print("\nFichiers générés :")
print("- classification_report.txt")
print("- confusion_matrix.png")