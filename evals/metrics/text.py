"""Word-overlap metrics, all scored from 0 to 1.

Normalization: Unicode NFKC, case folding, then Unicode word tokens.
Punctuation and Markdown are separators; numbers and negations are retained.
There is no stemming or stop-word removal. Two empty token sequences score 1;
if only one sequence is empty, the scores are 0.
"""

from collections import Counter
import re
import unicodedata


NORMALIZATION = "nfkc_casefold_unicode_words_v1"
METRIC_VERSION = 1


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        raise TypeError("response and expected_answer must be strings")
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())


def token_overlap(response: str, expected_answer: str) -> dict[str, float]:
    predicted = tokenize(response)
    expected = tokenize(expected_answer)
    if not predicted or not expected:
        score = float(predicted == expected)
        return {"token_precision": score, "token_recall": score, "token_f1": score}

    common = sum((Counter(predicted) & Counter(expected)).values())
    return {
        "token_precision": common / len(predicted),
        "token_recall": common / len(expected),
        "token_f1": 2 * common / (len(predicted) + len(expected)),
    }


def rouge_l(response: str, expected_answer: str) -> dict[str, float]:
    predicted = tokenize(response)
    expected = tokenize(expected_answer)
    if not predicted or not expected:
        return {"rouge_l_f1": float(predicted == expected)}

    # Longest common subsequence, keeping only one row of the DP table.
    if len(predicted) < len(expected):
        predicted, expected = expected, predicted
    previous = [0] * (len(expected) + 1)
    for word in predicted:
        current = [0]
        for index, other in enumerate(expected, start=1):
            if word == other:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current

    return {"rouge_l_f1": 2 * previous[-1] / (len(predicted) + len(expected))}
