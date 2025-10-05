from typing import List

import nltk
import spacy

try:
    NLP = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download

    download("en_core_web_sm")


def generate_spacy_tags(caption: str) -> List[str]:
    """Generate tags from caption using spaCy NLP."""
    # Use noun_chunks + nouns + adjectives, then dedupe and limit
    doc = NLP(caption)
    candidates_tags = []
    # nouns and proper nouns and adjectives
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN", "ADJ"} and not token.is_stop and token.is_alpha:
            txt = token.lemma_.lower().strip()
            if len(txt) > 1:
                candidates_tags.append(txt)
    # NLTK bigrams
    tokens = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
    for bg in nltk.bigrams(tokens):
        bigram = "_".join(bg)
        if len(bigram.replace("_", "")) > 2:
            candidates_tags.append(bigram)
            # cleanup and dedupe
    seen = set()
    tags = []
    for t in candidates_tags:
        if t not in seen:
            seen.add(t)
            tags.append(t.replace(" ", "_"))
        if len(tags) >= 50:
            break
    return tags
