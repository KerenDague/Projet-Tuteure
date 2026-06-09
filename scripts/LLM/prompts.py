import json
from typing import Any

from feature_schema import NUMERIC_FEATURE_COLUMNS

TARGET_NATIVE_LANGUAGES: tuple[str, ...] = (
    "ANGLAIS",
    "ARABE",
    "CHINOIS",
    "PORTUGAIS",
    "RUSSE",
)
TARGET_NATIVE_LANGUAGE_CODES: tuple[str, ...] = ("A", "B", "C", "D", "E")
TARGET_NATIVE_LANGUAGE_LABEL_TO_CODE: dict[str, str] = {
    "ANGLAIS": "A",
    "ARABE": "B",
    "CHINOIS": "C",
    "PORTUGAIS": "D",
    "RUSSE": "E",
}
TARGET_NATIVE_LANGUAGE_CODEBOOK_TEXT = "A=ANGLAIS, B=ARABE, C=CHINOIS, D=PORTUGAIS, E=RUSSE"
TARGET_NATIVE_LANGUAGE_CODES_AS_JSON = json.dumps(
    list(TARGET_NATIVE_LANGUAGE_CODES),
    ensure_ascii=True,
)
JSON_ONLY_SYSTEM_PROMPT = (
    "Return valid JSON only. Do not use Markdown. Do not add any prose outside the JSON payload."
)


def render_feature_block(numeric_features: dict[str, float]) -> str:
    """Render the numeric feature block for one prompt.

    Args:
        numeric_features: Ordered numeric features for one text.

    Returns:
        A human-readable French feature block.
    """

    lines = [
        "Features statistiques:",
        "Chaque ligne suit le format nom_feature = valeur.",
    ]
    for column_name in NUMERIC_FEATURE_COLUMNS:
        if column_name in numeric_features:
            lines.append(f"- {column_name} = {numeric_features[column_name]:.2f}")
    return "\n".join(lines)


def render_few_shot_block(
    few_shot_examples: list[dict[str, Any]] | None,
    *,
    include_features: bool,
) -> str:
    """Render the deterministic few-shot block.

    Args:
        few_shot_examples: Ordered support examples or None.
        include_features: Whether each example should show numeric features.

    Returns:
        The formatted few-shot block, or an empty string.
    """

    if not few_shot_examples:
        return ""

    blocks = ["Exemples:"]
    for example_index, example in enumerate(few_shot_examples, start=1):
        lines = [
            f"Exemple {example_index}",
            f"Texte: {example['text']}",
        ]
        if include_features and example.get("numeric_features") is not None:
            lines.append(render_feature_block(example["numeric_features"]))
        lines.append(
            "Code correct de langue maternelle: "
            f"{example['language_code']}"
        )
        blocks.append("\n".join(lines))
    return "\n\n" + "\n\n".join(blocks) + "\n"


def build_direct_label_minimal_prompt(
    text: str,
    few_shot_examples: list[dict[str, Any]] | None = None,
) -> str:
    """Build the minimal direct-label prompt.

    Args:
        text: Raw learner text.
        few_shot_examples: Optional deterministic support examples.

    Returns:
        The full French prompt.
    """

    few_shot_block = render_few_shot_block(
        few_shot_examples,
        include_features=False,
    )
    return (
        "Vous etes un classifieur linguistique. Votre tache est d'identifier la "
        "langue maternelle la plus probable d'un auteur non natif a partir de son "
        "texte en francais. Vous recevez uniquement le texte original. Avant de "
        "choisir la langue finale, comparez mentalement les 5 langues candidates une "
        "par une. Utilisez uniquement les indices linguistiques observables dans "
        "l'ecriture. Ignorez toute information contextuelle telle que les noms, "
        "adresses, institutions, lieux ou references culturelles. Ne choisissez "
        "aucune langue par defaut simplement parce que le texte contient des fautes "
        "generales ou un niveau faible en francais. Si les indices sont faibles ou "
        "ambigus, choisissez quand meme la meilleure langue parmi les 5, mais baissez "
        "nettement `confidence`. Vous devez produire un code de langue et non pas le "
        f"nom complet. Codebook obligatoire: {TARGET_NATIVE_LANGUAGE_CODEBOOK_TEXT}. "
        "`predicted_native_language` doit etre exactement l'une des 5 valeurs "
        f"suivantes: {TARGET_NATIVE_LANGUAGE_CODES_AS_JSON}. Avant de produire le "
        "JSON final, verifiez que `predicted_native_language` appartient bien a la "
        f"liste fermee suivante: {TARGET_NATIVE_LANGUAGE_CODES_AS_JSON}. Renvoyez "
        "uniquement un objet JSON avec les cles `predicted_native_language` et "
        f"`confidence`.{few_shot_block}\nTexte:\n{text}"
    )


def build_direct_label_minimal_features_prompt(
    text: str,
    numeric_features: dict[str, float],
    few_shot_examples: list[dict[str, Any]] | None = None,
) -> str:
    """Build the features-aware direct-label prompt.

    Args:
        text: Raw learner text.
        numeric_features: Numeric features for the current text.
        few_shot_examples: Optional deterministic support examples.

    Returns:
        The full French prompt.
    """

    few_shot_block = render_few_shot_block(
        few_shot_examples,
        include_features=True,
    )
    feature_block = render_feature_block(numeric_features)
    return (
        "Vous etes un classifieur linguistique. Votre tache est d'identifier la "
        "langue maternelle la plus probable d'un auteur non natif a partir de son "
        "texte en francais. Vous recevez deux sources d'information: le texte "
        "original de l'apprenant et des features statistiques numeriques calculees "
        "automatiquement sur ce texte. Utilisez les features comme des indices "
        "auxiliaires et non comme une verite absolue. Appuyez-vous d'abord sur les "
        "indices linguistiques observables dans le texte, puis utilisez les features "
        "pour confirmer ou departager des hypotheses si besoin. Avant de choisir la "
        "langue finale, comparez mentalement les 5 langues candidates une par une. "
        "Ignorez toute information contextuelle telle que les noms, adresses, "
        "institutions, lieux ou references culturelles. Ne choisissez aucune langue "
        "par defaut simplement parce que le texte contient des fautes generales ou "
        "un niveau faible en francais. Si le texte et les features suggerent des "
        "hypotheses differentes, privilegiez les indices linguistiques explicites du "
        "texte et baissez nettement `confidence`. Vous devez produire un code de "
        f"langue et non pas le nom complet. Codebook obligatoire: {TARGET_NATIVE_LANGUAGE_CODEBOOK_TEXT}. "
        "`predicted_native_language` doit etre exactement l'une des 5 valeurs "
        f"suivantes: {TARGET_NATIVE_LANGUAGE_CODES_AS_JSON}. Avant de produire le "
        "JSON final, verifiez que `predicted_native_language` appartient bien a la "
        f"liste fermee suivante: {TARGET_NATIVE_LANGUAGE_CODES_AS_JSON}. Renvoyez "
        "uniquement un objet JSON avec les cles `predicted_native_language` et "
        f"`confidence`.{few_shot_block}\nTexte:\n{text}\n\n{feature_block}"
    )


def build_direct_label_repair_prompt(
    text: str,
    previous_output: str,
    validation_error: str,
    numeric_features: dict[str, float] | None = None,
) -> str:
    """Build the repair prompt for an invalid direct-label output.

    Args:
        text: Raw learner text.
        previous_output: Raw invalid model output.
        validation_error: Local validation error.
        numeric_features: Optional numeric feature block.

    Returns:
        The full French repair prompt.
    """

    feature_block = ""
    if numeric_features is not None:
        feature_block = f"\n\n{render_feature_block(numeric_features)}"
    return (
        "Tu corriges une sortie invalide d'un agent de prediction directe minimale. "
        "Tu ne dois pas refaire toute l'analyse et tu ne dois pas recopier la sortie "
        "invalide. Tu dois seulement produire un JSON final valide. Regle absolue: "
        "tu dois produire un code et non pas le nom complet de la langue. Codebook "
        f"obligatoire: {TARGET_NATIVE_LANGUAGE_CODEBOOK_TEXT}. "
        "`predicted_native_language` doit etre exactement l'une des 5 valeurs "
        f"suivantes: {TARGET_NATIVE_LANGUAGE_CODES_AS_JSON}. Toute autre valeur "
        "est interdite, y compris FRANCAIS, FRENCH, ESPAGNOL et les noms complets "
        "des langues. Ne choisis aucune langue par defaut. Si les indices sont "
        "ambigus, garde une confiance basse plutot que d'inventer une certitude. "
        "Tu renvoies uniquement un objet JSON. Format obligatoire: "
        "{{\"predicted_native_language\": \"A\", \"confidence\": 0.42}}.\n\n"
        f"Sortie invalide precedente:\n{previous_output}\n\n"
        f"Erreur de validation:\n{validation_error}\n\n"
        f"Texte:\n{text}{feature_block}"
    )
