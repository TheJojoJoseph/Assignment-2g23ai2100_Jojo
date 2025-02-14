import os
from itertools import combinations


def g23ai2100_read_docs(dir_path):
    files = {}
    for name in ['D1.txt', 'D2.txt', 'D3.txt', 'D4.txt']:
        try:
            with open(os.path.join(dir_path, name), 'r') as f:
                files[name] = ''.join(
                    c for c in f.read().lower() if c.islower() or c.isspace()).strip()
        except FileNotFoundError:
            print(f"Warning: {name} not found")
    return files


def g23ai2100_get_ngrams(text, n, by_word=False):
    tokens = text.split() if by_word else text
    return {''.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}


def g23ai2100_jaccard(set1, set2):
    return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 1.0


def g23ai2100_analyze(dir_path):
    docs = g23ai2100_read_docs(dir_path)
    ngram_types = {"Char-2": (2, False), "Char-3": (3,
                                                    False), "Word-2": (2, True)}
    ngrams = {t: {d: g23ai2100_get_ngrams(
        text, *p) for d, text in docs.items()} for t, p in ngram_types.items()}

    print("\nPart A: Distinct k-grams\n" + "=" * 40)
    for d in docs:
        print(f"\n{d}: " +
              " ".join(f"{t}: {len(ngrams[t][d])}" for t in ngrams))

    print("\nPart B: Jaccard Similarities\n" + "=" * 40)
    for t in ngrams:
        print(f"\n{t}:")
        for d1, d2 in combinations(docs, 2):
            print(
                f"{d1}-{d2}: {g23ai2100_jaccard(ngrams[t][d1], ngrams[t][d2]):.4f}")


if __name__ == "__main__":
    g23ai2100_analyze("minhash")
