import re
import logging
from rouge_score import rouge_scorer
from bert_score import score as bert_score

logger = logging.getLogger(__name__)

# Temporal connective words used to gauge sequence accuracy
_TEMPORAL_WORDS = {
    "first", "then", "next", "after", "before", "finally",
    "followed", "subsequently", "initially", "later", "begins",
    "starts", "ends", "continues", "meanwhile", "suddenly",
}


def compute_exact_match(predictions: list, references: list) -> float:
    """Exact match accuracy for MCQ tasks."""
    correct = sum(
        p.strip().lower() == r.strip().lower()
        for p, r in zip(predictions, references)
    )
    return round(correct / len(predictions) * 100, 2)


def compute_rouge_l(predictions: list, references: list) -> float:
    """ROUGE-L F-measure for open-ended generation quality."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(predictions, references)
    ]
    return round(sum(scores) / len(scores) * 100, 2)


def compute_rouge_1(predictions: list, references: list) -> float:
    """ROUGE-1 F-measure for unigram overlap."""
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    scores = [
        scorer.score(ref, pred)["rouge1"].fmeasure
        for pred, ref in zip(predictions, references)
    ]
    return round(sum(scores) / len(scores) * 100, 2)


def compute_bert_score(predictions: list, references: list) -> float:
    """BERTScore F1 for semantic similarity."""
    P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
    return round(F1.mean().item() * 100, 2)


def hallucination_rate(predictions: list, references: list) -> float:
    """% of predictions with zero word overlap with reference."""
    hallucinated = sum(
        len(set(p.lower().split()) & set(r.lower().split())) == 0
        for p, r in zip(predictions, references)
    )
    return round(hallucinated / len(predictions) * 100, 2)


def temporal_ordering_accuracy(predictions: list, references: list) -> float:
    """
    Measures how well predictions capture temporal language.

    Score per sample = F1 of temporal-word overlap between prediction
    and reference, then averaged across the dataset.

    A high score means the model uses the same temporal connectives
    as the ground truth (first/then/finally/followed by/etc.).
    """
    scores = []
    for pred, ref in zip(predictions, references):
        pred_words = set(re.findall(r"\b\w+\b", pred.lower()))
        ref_words  = set(re.findall(r"\b\w+\b", ref.lower()))

        pred_temporal = pred_words & _TEMPORAL_WORDS
        ref_temporal  = ref_words  & _TEMPORAL_WORDS

        if not ref_temporal:
            # No temporal words in reference — skip this sample
            continue

        precision = len(pred_temporal & ref_temporal) / len(pred_temporal) if pred_temporal else 0.0
        recall    = len(pred_temporal & ref_temporal) / len(ref_temporal)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        scores.append(f1)

    if not scores:
        return 0.0
    return round(sum(scores) / len(scores) * 100, 2)


def sound_event_recall(predictions: list, references: list) -> float:
    """
    Measures content word recall: what fraction of meaningful words
    in the reference appear in the prediction.

    Filters out stopwords to focus on sound-event nouns/verbs.
    """
    _STOPWORDS = {"a", "an", "the", "is", "are", "was", "were", "in", "on",
                  "at", "to", "of", "and", "or", "with", "this", "that",
                  "it", "by", "from", "be", "as", "for"}

    recalls = []
    for pred, ref in zip(predictions, references):
        pred_words = set(re.findall(r"\b\w+\b", pred.lower())) - _STOPWORDS
        ref_words  = set(re.findall(r"\b\w+\b", ref.lower()))  - _STOPWORDS

        if not ref_words:
            continue
        recall = len(pred_words & ref_words) / len(ref_words)
        recalls.append(recall)

    if not recalls:
        return 0.0
    return round(sum(recalls) / len(recalls) * 100, 2)


def full_evaluation(predictions: list, references: list, task: str = "temporal") -> dict:
    """
    Run all metrics and return results dict.

    task="temporal"  → ROUGE-1, ROUGE-L, BERTScore, hallucination rate,
                        temporal ordering accuracy, sound event recall
    task="mcq"       → exact match only
    task="open"      → ROUGE-L, BERTScore, hallucination rate
    """
    if not predictions or not references:
        return {}

    results = {}

    if task == "mcq":
        results["exact_match"] = compute_exact_match(predictions, references)

    elif task == "temporal":
        results["rouge_1"]                   = compute_rouge_1(predictions, references)
        results["rouge_l"]                   = compute_rouge_l(predictions, references)
        results["bert_score"]                = compute_bert_score(predictions, references)
        results["hallucination_rate"]        = hallucination_rate(predictions, references)
        results["temporal_ordering_accuracy"] = temporal_ordering_accuracy(predictions, references)
        results["sound_event_recall"]        = sound_event_recall(predictions, references)

    else:  # "open"
        results["rouge_l"]           = compute_rouge_l(predictions, references)
        results["bert_score"]        = compute_bert_score(predictions, references)
        results["hallucination_rate"] = hallucination_rate(predictions, references)

    logger.info(f"Evaluation results: {results}")
    return results
