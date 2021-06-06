from dataclasses import dataclass

import numpy as np
import pandas as pd
import spacy
from spacy.tokens import Doc, Token

nlp = spacy.load("en_core_web_lg")


@dataclass(frozen=True)
class WordsSummary:
    num_words: int
    num_stop_words: int
    num_words_oov: int
    num_distinct_words: int
    mean_word_length: float
    p90_word_length: float
    max_word_length: float
    perc_word_length_gt8: float
    perc_word_length_gt12: float
    perc_adjective: float
    perc_adverb: float
    perc_interjection: float
    perc_noun: float
    perc_verb: float


@dataclass(frozen=True)
class SentencesSummary:
    num_sentences: int
    mean_sentence_length: float
    p90_sentence_length: float
    max_sentence_length: float
    perc_sentence_length_gt_15: float
    perc_sentence_length_gt_30: float


def extract_tensor(doc: Doc) -> np.array:
    tensor_flat = np.nanmean(doc.tensor, axis=0)
    return tensor_flat


def extract_token_embeddings(doc: Doc) -> np.array:
    all_embeddings = np.vstack([tok.vector for tok in doc])
    embeddings_flat = np.nanmean(all_embeddings, axis=0)
    return embeddings_flat


def is_word_strict(tok: Token) -> bool:
    if any(
        [tok.is_digit, tok.is_punct, tok.is_space, tok.is_bracket, tok.is_quote, tok.is_currency]
    ):
        return False
    else:
        return True


def extract_words_summary(doc: Doc) -> WordsSummary:
    words = [tok for tok in doc if is_word_strict(tok)]
    stops = np.array([tok.is_stop for tok in words])
    words_excl_stops = [tok for tok in words if not tok.is_stop]
    words_oov = [tok for tok in words if tok.is_oov]

    distinct_words = np.unique([tok.lemma_ for tok in words_excl_stops])
    word_lengths = np.array([len(tok) for tok in words_excl_stops])
    pos = np.array([spacy.explain(tok.pos_) for tok in words_excl_stops])

    num_words = len(word_lengths)
    num_stop_words = np.sum(stops)
    num_words_oov = len(words_oov)
    num_distinct_words = len(distinct_words)
    mean_word_length = np.mean(word_lengths)
    p90_word_length = np.quantile(word_lengths, 0.9)
    max_word_length = np.max(word_lengths)
    perc_word_length_gt8 = np.sum(word_lengths > 8) / num_words
    perc_word_length_gt12 = np.sum(word_lengths > 12) / num_words

    perc_pos = {}
    for part in ["adjective", "adverb", "interjection", "noun", "verb"]:
        perc_pos[part] = np.sum(pos == part) / num_words

    return WordsSummary(
        num_words=num_words,
        num_stop_words=num_stop_words,
        num_words_oov=num_words_oov,
        num_distinct_words=num_distinct_words,
        mean_word_length=mean_word_length,
        p90_word_length=p90_word_length,
        max_word_length=max_word_length,
        perc_word_length_gt8=perc_word_length_gt8,
        perc_word_length_gt12=perc_word_length_gt12,
        perc_adjective=perc_pos["adjective"],
        perc_adverb=perc_pos["adverb"],
        perc_interjection=perc_pos["interjection"],
        perc_noun=perc_pos["noun"],
        perc_verb=perc_pos["verb"],
    )


def extract_sentences_summary(doc: Doc) -> SentencesSummary:
    sent_lengths = np.array([len(s) for s in doc.sents])
    num_sentences = sent_lengths.shape[0]

    return SentencesSummary(
        num_sentences=num_sentences,
        mean_sentence_length=np.mean(sent_lengths),
        p90_sentence_length=np.quantile(sent_lengths, 0.9),
        max_sentence_length=np.max(sent_lengths),
        perc_sentence_length_gt_15=np.sum(sent_lengths > 15) / num_sentences,
        perc_sentence_length_gt_30=np.sum(sent_lengths > 30) / num_sentences,
    )


def tensors_df(docs: list[Doc]) -> pd.DataFrame:
    values = np.vstack([extract_tensor(doc) for doc in docs])
    colnames = [f"tensor_{i}" for i in range(values.shape[1])]
    idx = [i for i in range(values.shape[0])]
    return pd.DataFrame(values, columns=colnames, index=idx)


def token_embeddings_df(docs: list[Doc]) -> pd.DataFrame:
    values = np.vstack([extract_token_embeddings(doc) for doc in docs])
    colnames = [f"token_embeddings_{i}" for i in range(values.shape[1])]
    idx = [i for i in range(values.shape[0])]
    return pd.DataFrame(values, columns=colnames, index=idx)


def words_summary_df(docs: list[Doc]) -> pd.DataFrame:
    values = [extract_words_summary(doc) for doc in docs]
    idx = [i for i in range(len(values))]
    return pd.DataFrame(values, index=idx)


def sentences_summary_df(docs: list[Doc]) -> pd.DataFrame:
    values = [extract_sentences_summary(doc) for doc in docs]
    idx = [i for i in range(len(values))]
    return pd.DataFrame(values, index=idx)


def process_texts(texts: list[str]) -> pd.DataFrame:
    docs = [nlp(text) for text in texts]
    tensors = tensors_df(docs)
    token_embeddings = token_embeddings_df(docs)
    words_summary = words_summary_df(docs)
    sentences_summary = sentences_summary_df(docs)
    features = pd.concat([tensors, token_embeddings, words_summary, sentences_summary], axis=1)
    return features
