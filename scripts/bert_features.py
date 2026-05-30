from __future__ import annotations
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable
import numpy as np
import pandas as pd
import spacy
from sklearn.preprocessing import StandardScaler

#
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


FEATURE_GROUP_COLUMNS = {
    "punctuation": ["punctuation_punct_ratio"],
    "articles": ["articles_det_ratio"],
    "majuscules": ["majuscules_freq_maj"],
    "prepositions": ["prepositions_freq_prep"],
    "pronouns": ["pronouns_ratio_prodrop"],
    "sentence_structure": [
        "sentence_structure_subj_mean",
        "sentence_structure_verb_mean",
        "sentence_structure_obj_mean",
        "sentence_structure_ratio_sov",
        "sentence_structure_distance_verb_subject",
    ],
    "words_per_sentence": ["words_per_sentence_mean"],
    "discourse_connectors": ["discourse_connectors_freq"],
    "verb_tenses": [
        "verb_tenses_present_ratio",
        "verb_tenses_future_ratio",
        "verb_tenses_past_ratio",
        "verb_tenses_composed_ratio",
    ],
    "readability": ["readability_flesch_fr"],
    "syntactic_complexity": ["syntactic_complexity_dependency_depth"],
    "noun_repetition": ["noun_repetition_ratio"],
    "verb_noun_ratio": ["verb_noun_ratio"],
    "passive": ["passive_ratio"],
}


def available_features() -> list[str]:
    return sorted(FEATURE_GROUP_COLUMNS)


@lru_cache(maxsize=1)
def get_nlp():
    try:
        return spacy.load(SPACY_MODEL)
    except OSError as exc:
        raise OSError(
            f"spaCy model '{SPACY_MODEL}' is not installed. "
            f"Install it with: python -m spacy download {SPACY_MODEL}"
        ) from exc


def count_syllables(word: str) -> int:
    return max(1, len(re.findall(r'[aeiouyàâéèêëîïôùûü]+', word.lower())))


def tree_depth(token) -> int:
    depth = 0
    current = token
    while current.head != current:
        depth += 1
        current = current.head
    return depth


# calculate mean and return 0.0 if the list is empty
def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


# with spacy, determine the ratio of verbs in present, future, past and composed tenses
def _verb_tense_features(tokens) -> tuple[float, float, float, float]:
    finite_verbs = [
        token
        for token in tokens
        if token.pos_ in ("VERB", "AUX")
        and token.morph.get("VerbForm") not in (["Inf"], ["Part"])
    ]
    total = max(1, len(finite_verbs))

    present = 0
    future = 0
    past = 0
    composed = 0

    for token in finite_verbs:
        tense_values = token.morph.get("Tense")
        verb_form = token.morph.get("VerbForm")

        if "Pres" in tense_values:
            present += 1
        if "Fut" in tense_values:
            future += 1
        if "Past" in tense_values or "Imp" in tense_values:
            past += 1

        has_participle_child = any(
            child.pos_ == "VERB" and child.morph.get("VerbForm") == ["Part"]
            for child in token.children
        )
        if token.pos_ == "AUX" and has_participle_child:
            composed += 1
        elif verb_form == ["Part"]:
            composed += 1

    return present / total, future / total, past / total, composed / total


def extract_all_linguistic_features(texts: Iterable[str]) -> pd.DataFrame:
    nlp = get_nlp()
    rows = []

    for doc in nlp.pipe([str(text) for text in texts], disable=["ner"]):
        tokens = [token for token in doc if not token.is_space]
        nb_tokens = max(1, len(tokens))
        text_low = doc.text.lower()
        phrases = list(doc.sents)
        nb_phrases = max(1, len(phrases))

        punct_ratio = sum(1 for token in tokens if token.text in ".,;:!?") / nb_tokens

        mots_alpha = [token.text for token in tokens if token.is_alpha]
        nb_mots = max(1, len(mots_alpha))
        nb_syllabes = sum(count_syllables(word) for word in mots_alpha)
        lisibilite = 206.835 - 1.015 * (nb_mots / nb_phrases) - 84.6 * (nb_syllabes / nb_mots)

        phrase_lengths = [len([token for token in sent if not token.is_space]) for sent in phrases]
        nb_mots_moy = _safe_mean(phrase_lengths)

        depths = [tree_depth(token) for token in tokens if token.pos_ in ("NOUN", "VERB", "ADJ")]
        profondeur_moy = _safe_mean(depths)
        ratio_sconj = sum(1 for token in tokens if token.pos_ == "SCONJ") / nb_phrases
        complexite_syntaxique = profondeur_moy + ratio_sconj

        freq_prep = sum(1 for token in tokens if token.pos_ == "ADP") / nb_tokens

        noun_lemmas = [token.lemma_.lower() for token in tokens if token.pos_ == "NOUN"]
        if noun_lemmas:
            noun_counts = pd.Series(noun_lemmas).value_counts()
            repetition_noms = float((noun_counts >= 2).sum() / len(noun_counts))
        else:
            repetition_noms = 0.0

        nb_conn = sum(len(re.findall(r"\b" + re.escape(conn) + r"\b", text_low)) for conn in CONNECTEURS)
        freq_conn = nb_conn / nb_tokens

        det_count = sum(1 for token in tokens if token.pos_ == "DET")
        noun_count = max(1, sum(1 for token in tokens if token.pos_ == "NOUN"))
        det_ratio = det_count / noun_count
        nb_verbes_seuls = sum(1 for token in tokens if token.pos_ == "VERB")
        ratio_verbes_noms = nb_verbes_seuls / noun_count

        nb_maj = 0
        for sent in doc.sents:
            sent_tokens = [token for token in sent if not token.is_space]
            for token in sent_tokens[1:]:
                if token.text and token.text[0].isupper() and token.pos_ != "PROPN":
                    nb_maj += 1
        freq_maj = nb_maj / nb_tokens

        scores_ordre = []
        for sent in doc.sents:
            sent_tokens = list(sent)
            pos_map = {token.i: idx for idx, token in enumerate(sent_tokens)}
            for token in sent_tokens:
                if token.pos_ == "VERB":
                    children = list(token.children)
                    subj = next((child for child in children if child.dep_ == "nsubj"), None)
                    obj = next((child for child in children if child.dep_ == "obj"), None)
                    if subj and obj:
                        verb_pos = pos_map.get(token.i, 0)
                        subj_pos = pos_map.get(subj.i, 0)
                        obj_pos = pos_map.get(obj.i, 0)
                        if obj_pos < verb_pos:
                            scores_ordre.append(1)
                        elif subj_pos > verb_pos:
                            scores_ordre.append(-1)
                        else:
                            scores_ordre.append(0)
        ratio_sov = _safe_mean(scores_ordre)

        nb_passif = 0
        for token in tokens:
            if token.dep_ == "auxpass":
                nb_passif += 1
            elif (
                token.pos_ == "AUX"
                and token.lemma_ == "être"
                and any(child.dep_ == "nsubj:pass" for child in token.head.children)
            ):
                nb_passif += 1
        ratio_passif = nb_passif / nb_phrases

        distances = []
        subj_positions = []
        verb_positions = []
        obj_positions = []
        for idx, token in enumerate(doc):
            if token.dep_ == "nsubj":
                subj_positions.append(idx)
            if token.dep_ == "obj":
                obj_positions.append(idx)
            if token.pos_ == "VERB":
                verb_positions.append(idx)
                subj = next((child for child in token.children if child.dep_ in ("nsubj", "nsubj:pass")), None)
                if subj:
                    distances.append(abs(token.i - subj.i))

        distance_verbe_sujet = _safe_mean(distances)
        subj_mean = _safe_mean(subj_positions)
        verb_mean = _safe_mean(verb_positions)
        obj_mean = _safe_mean(obj_positions)

        verbes_conjugues = [
            token
            for token in tokens
            if token.pos_ in ("VERB", "AUX")
            and token.morph.get("VerbForm") not in (["Inf"], ["Part"])
        ]
        nb_verbes_conj = max(1, len(verbes_conjugues))
        nb_pron_sujet = sum(
            1
            for token in tokens
            if token.text.lower() in PRONOMS_SUJETS and token.dep_ == "nsubj"
        )
        ratio_prodrop = nb_pron_sujet / nb_verbes_conj

        present_ratio, future_ratio, past_ratio, composed_ratio = _verb_tense_features(tokens)

        rows.append(
            {
                "punctuation_punct_ratio": punct_ratio,
                "articles_det_ratio": det_ratio,
                "majuscules_freq_maj": freq_maj,
                "prepositions_freq_prep": freq_prep,
                "pronouns_ratio_prodrop": ratio_prodrop,
                "sentence_structure_subj_mean": subj_mean,
                "sentence_structure_verb_mean": verb_mean,
                "sentence_structure_obj_mean": obj_mean,
                "sentence_structure_ratio_sov": ratio_sov,
                "sentence_structure_distance_verb_subject": distance_verbe_sujet,
                "words_per_sentence_mean": nb_mots_moy,
                "discourse_connectors_freq": freq_conn,
                "verb_tenses_present_ratio": present_ratio,
                "verb_tenses_future_ratio": future_ratio,
                "verb_tenses_past_ratio": past_ratio,
                "verb_tenses_composed_ratio": composed_ratio,
                "readability_flesch_fr": lisibilite,
                "syntactic_complexity_dependency_depth": complexite_syntaxique,
                "noun_repetition_ratio": repetition_noms,
                "verb_noun_ratio": ratio_verbes_noms,
                "passive_ratio": ratio_passif,
            }
        )

    result = pd.DataFrame(rows)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


def build_feature_dataframe(texts: Iterable[str], feature_names: list[str]) -> pd.DataFrame:
    unknown = sorted(set(feature_names) - set(FEATURE_GROUP_COLUMNS))
    if unknown:
        valid = ", ".join(available_features())
        raise ValueError(f"Unknown feature(s): {', '.join(unknown)}. Available: {valid}")

    text_list = list(texts)
    if not feature_names:
        return pd.DataFrame(index=range(len(text_list)))

    all_features = extract_all_linguistic_features(text_list)
    selected_columns = []
    for feature_name in feature_names:
        selected_columns.extend(FEATURE_GROUP_COLUMNS[feature_name])

    return all_features[selected_columns].reset_index(drop=True)


@dataclass
class TextFeatureExtractor:
    feature_names: list[str]
    scale: bool = True

    def __post_init__(self) -> None:
        self.scaler: StandardScaler | None = StandardScaler() if self.scale else None
        self.column_names: list[str] = []

    def fit(self, texts: Iterable[str]) -> "TextFeatureExtractor":
        features = build_feature_dataframe(texts, self.feature_names)
        self.column_names = features.columns.tolist()
        if self.scaler is not None and self.column_names:
            self.scaler.fit(features.values)
        return self

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        features = build_feature_dataframe(texts, self.feature_names)
        if self.column_names:
            features = features.reindex(columns=self.column_names, fill_value=0.0)
        values = features.values.astype(np.float32)
        if self.scaler is not None and self.column_names:
            values = self.scaler.transform(values).astype(np.float32)
        return values

    def fit_transform(self, texts: Iterable[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)

    @property
    def n_features(self) -> int:
        return len(self.column_names)