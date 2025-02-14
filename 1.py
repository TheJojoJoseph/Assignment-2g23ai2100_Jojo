import os
from typing import List, Set, Dict
from itertools import combinations


def read_files(directory: str) -> Dict[str, str]:
    """Read all text files from the directory."""
    files = {}
    for filename in ['D1.txt', 'D2.txt', 'D3.txt', 'D4.txt']:
        path = os.path.join(directory, filename)
        try:
            with open(path, 'r') as f:
                content = f.read().strip().lower()
                # Ensure only lowercase letters and spaces
                content = ''.join(
                    c for c in content if c.islower() or c.isspace())
                files[filename] = content
        except FileNotFoundError:
            print(f"Warning: {filename} not found")
    return files


def get_char_ngrams(text: str, n: int) -> Set[str]:
    """Generate character n-grams from text."""
    return set(text[i:i+n] for i in range(len(text) - n + 1))


def get_word_ngrams(text: str, n: int) -> Set[str]:
    """Generate word n-grams from text."""
    words = text.split()
    return set(' '.join(words[i:i+n]) for i in range(len(words) - n + 1))


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union


def analyze_documents(directory: str):
    """Analyze documents and print results."""
    # Read files
    documents = read_files(directory)

    # Store k-grams for each document
    char_2grams = {name: get_char_ngrams(text, 2)
                   for name, text in documents.items()}
    char_3grams = {name: get_char_ngrams(text, 3)
                   for name, text in documents.items()}
    word_2grams = {name: get_word_ngrams(text, 2)
                   for name, text in documents.items()}

    # Part A: Count distinct k-grams
    print("\nPart A: Number of distinct k-grams")
    print("=" * 50)
    for doc_name in documents.keys():
        print(f"\n{doc_name}:")
        print(f"Character 2-grams: {len(char_2grams[doc_name])}")
        print(f"Character 3-grams: {len(char_3grams[doc_name])}")
        print(f"Word 2-grams: {len(word_2grams[doc_name])}")

    # Part B: Calculate Jaccard similarities
    print("\nPart B: Jaccard similarities")
    print("=" * 50)

    doc_pairs = list(combinations(documents.keys(), 2))

    print("\nCharacter 2-grams similarities:")
    for doc1, doc2 in doc_pairs:
        sim = jaccard_similarity(char_2grams[doc1], char_2grams[doc2])
        print(f"{doc1} - {doc2}: {sim:.4f}")

    print("\nCharacter 3-grams similarities:")
    for doc1, doc2 in doc_pairs:
        sim = jaccard_similarity(char_3grams[doc1], char_3grams[doc2])
        print(f"{doc1} - {doc2}: {sim:.4f}")

    print("\nWord 2-grams similarities:")
    for doc1, doc2 in doc_pairs:
        sim = jaccard_similarity(word_2grams[doc1], word_2grams[doc2])
        print(f"{doc1} - {doc2}: {sim:.4f}")


if __name__ == "__main__":
    directory_path = "minhash"
    analyze_documents(directory_path)
