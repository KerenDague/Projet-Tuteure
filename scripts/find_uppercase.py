import csv
from collections import Counter
import os

### CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT = os.path.join(BASE_DIR, "corpusAB_clean.csv")
OUTPUT = os.path.join(BASE_DIR, "output.txt")
OUTPUT_IDS = os.path.join(BASE_DIR, "output_ids.txt")
#####


total_rows     = 0
uppercase_rows = 0
counts = Counter()
ids    = []

with open(INPUT, "r", encoding="utf-8") as f:
    r = csv.DictReader(f) #first row keys
    for row in r:                          
        total_rows += 1
        texte   = row["Texte"]
        langue  = row["Langue"]
        niveau  = row["Niveau"]
        row_id  = row["ID"]

        letters = [c for c in texte if c.isalpha()] # we will look only letters 
        upper   = [c for c in letters if c.isupper()] 

        if letters and len(upper) / len(letters) >= 0.6:  #if uppercase letter ratio is more than 60 its a match
            uppercase_rows += 1
            counts[(langue, niveau)] += 1
            ids.append(row_id)

percentage = (uppercase_rows / total_rows) * 100 if total_rows else 0

with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write(f"Lignes totales : {total_rows}\n")
    f.write(f"Textes à au moins 60% en majuscules : {uppercase_rows}\n")
    f.write(f"Pourcentage : {percentage:.2f}%\n")
    f.write("Langue et Niveau :\n")
    for (langue, niveau), count in counts.items():   
        f.write(f"{count} - {langue} - {niveau}\n")

with open(OUTPUT_IDS, "w", encoding="utf-8") as f:
    for row_id in ids:                              
        f.write(row_id + "\n")