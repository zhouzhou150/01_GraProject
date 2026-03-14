from __future__ import annotations

from collections import Counter


def _levenshtein(source: list[str], target: list[str]) -> int:
    if not source:
        return len(target)
    if not target:
        return len(source)
    rows = len(source) + 1
    cols = len(target) + 1
    matrix = [[0] * cols for _ in range(rows)]
    for row in range(rows):
        matrix[row][0] = row
    for col in range(cols):
        matrix[0][col] = col
    for row in range(1, rows):
        for col in range(1, cols):
            cost = 0 if source[row - 1] == target[col - 1] else 1
            matrix[row][col] = min(
                matrix[row - 1][col] + 1,
                matrix[row][col - 1] + 1,
                matrix[row - 1][col - 1] + cost,
            )
    return matrix[-1][-1]


def cer(reference: str, prediction: str) -> float:
    ref = list(reference.strip())
    pred = list(prediction.strip())
    if not ref:
        return 0.0 if not pred else 1.0
    return _levenshtein(ref, pred) / len(ref)


def wer(reference: str, prediction: str) -> float:
    ref = [token for token in reference.strip().split() if token]
    pred = [token for token in prediction.strip().split() if token]
    if not ref and reference.strip():
        ref = list(reference.strip())
    if not pred and prediction.strip():
        pred = list(prediction.strip())
    if not ref:
        return 0.0 if not pred else 1.0
    return _levenshtein(ref, pred) / len(ref)


def ser(reference: str, prediction: str) -> float:
    return 0.0 if reference.strip() == prediction.strip() else 1.0


def semdist_score(reference: str, prediction: str) -> float:
    ref = reference.strip()
    pred = prediction.strip()
    if not ref and not pred:
        return 100.0
    if not ref or not pred:
        return 0.0

    ref_bigrams = Counter(ref[index : index + 2] for index in range(max(1, len(ref) - 1)))
    pred_bigrams = Counter(pred[index : index + 2] for index in range(max(1, len(pred) - 1)))
    overlap = sum((ref_bigrams & pred_bigrams).values())
    total = max(1, sum(ref_bigrams.values()) + sum(pred_bigrams.values()))
    f1_like = 2 * overlap / total
    char_overlap = sum((Counter(ref) & Counter(pred)).values()) / max(1, len(ref))
    return round(min(100.0, max(0.0, (0.6 * f1_like + 0.4 * char_overlap) * 100)), 4)

