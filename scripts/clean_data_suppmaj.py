"""
Script de nettoyage et de préparation du corpus pour le classifieur qui doit :
- Supprimer toutes les balises HTML présentes dans un fichier CSV
- Supprimer les lignes dont l'id apparaît dans un fichier txt
- Produire un fichier CSV nettoyé
"""

import re
import argparse
from pathlib import Path
import polars as pl

# Configuration
TEXT_COLUMN      = "Texte"
LANGUE_COLUMN    = "Langue"
ID_COLUMN        = "ID"
HTML_TAG_PATTERN = re.compile(r"<[^>]*>")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nettoyage et préparation du corpus.")
    parser.add_argument("-i", "--input-csv",  required=True, type=Path, help="Fichier CSV d'entrée.")
    parser.add_argument("-o", "--output-csv", required=True, type=Path, help="Fichier CSV de sortie.")
    parser.add_argument("-r", "--remove-ids", default=None,  type=Path, help="Fichier txt contenant les ids à supprimer (un par ligne).")
    return parser.parse_args()

# Nettoyage des balises html
def strip_html(text: str) -> str:
    """Supprime les balises HTML."""
    text = HTML_TAG_PATTERN.sub("", str(text))
    return text.strip()


def load_ids_to_remove(ids_file: Path) -> set[str]:
    with ids_file.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}

# Nettoyage des lignes dont les ids ont été identifiés par find_uppercase.py
def remove_rows_by_id(dataframe: pl.DataFrame, ids_to_remove: set[str]) -> pl.DataFrame:
    if ID_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Colonne id introuvable: '{ID_COLUMN}'. "
            f"Colonnes disponibles: {', '.join(dataframe.columns)}"
        )
    before = len(dataframe)
    dataframe = dataframe.filter(
        ~pl.col(ID_COLUMN).cast(pl.Utf8).is_in(ids_to_remove)
    )
    print(f"  {before - len(dataframe)} ligne(s) supprimée(s) par id sur {before}.")
    return dataframe

# Fonction d'exécution'
def run(input_csv: Path, output_csv: Path, remove_ids: Path | None) -> None:
    dataframe = pl.read_csv(input_csv, infer_schema_length=False)

    for col in [TEXT_COLUMN, LANGUE_COLUMN]:
        if col not in dataframe.columns:
            raise ValueError(
                f"Colonne introuvable: '{col}'. "
                f"Colonnes disponibles: {', '.join(dataframe.columns)}"
            )

    # suppression des lignes par id
    if remove_ids is not None:
        ids_to_remove = load_ids_to_remove(remove_ids)
        print(f"{len(ids_to_remove)} id(s) à supprimer chargés depuis '{remove_ids}'.")
        dataframe = remove_rows_by_id(dataframe, ids_to_remove)

    # nettoyage HTML
    texts = (
        dataframe.get_column(TEXT_COLUMN)
        .cast(pl.Utf8, strict=False)
        .fill_null("")
        .to_list()
    )
    print(f"{len(texts)} textes à nettoyer.")
    cleaned_texts = [strip_html(text) for text in texts]

    dataframe = dataframe.with_columns(
        pl.Series(name=TEXT_COLUMN, values=cleaned_texts)
    )
    dataframe.write_csv(output_csv)
    print(f"Fichier sauvegardé : {output_csv}")

# Main
def main() -> None:
    args = parse_args()
    run(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        remove_ids=args.remove_ids,
    )


if __name__ == "__main__":
    main()


