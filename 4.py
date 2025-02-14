import numpy as np
from collections import defaultdict
import random
import os


class MinHash:
    def __init__(self, n_hash_functions):
        self.n_hash_functions = n_hash_functions
        self.prime = 4294967311  # A large prime number
        # Generate random coefficients for hash functions
        self.a = [random.randint(1, self.prime - 1)
                  for _ in range(n_hash_functions)]
        self.b = [random.randint(0, self.prime - 1)
                  for _ in range(n_hash_functions)]

    def hash_function(self, x, a, b):
        """Universal hash function: (ax + b) % prime"""
        return (a * x + b) % self.prime

    def compute_signature(self, document_words):
        """Compute minhash signature for a document's set of words"""
        signature = np.full(self.n_hash_functions, np.inf)

        for word in document_words:
            # Use hash of word as the input to our hash functions
            word_hash = hash(word) & 0xffffffff  # Ensure positive hash value
            for i in range(self.n_hash_functions):
                hash_value = self.hash_function(
                    word_hash, self.a[i], self.b[i])
                signature[i] = min(signature[i], hash_value)

        return signature


def load_documents(directory):
    """Load documents and create sets of words for each"""
    documents = {}

    # List all txt files in the directory
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                # Create set of words from the document
                words = set(f.read().strip().split())
                documents[filename] = words

    return documents


def jaccard_similarity(set1, set2):
    """Compute exact Jaccard similarity between two sets"""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0


def estimate_jaccard_similarity(sig1, sig2):
    """Estimate Jaccard similarity using minhash signatures"""
    return np.mean(sig1 == sig2)


def find_similar_pairs(documents, threshold=0.5, use_minhash=False, minhash_obj=None):
    """Find pairs of documents with similarity >= threshold"""
    similar_pairs = []
    doc_names = list(documents.keys())

    if use_minhash:
        # Compute signatures for all documents
        signatures = {doc: minhash_obj.compute_signature(words)
                      for doc, words in documents.items()}

        # Compare signatures
        for i in range(len(doc_names)):
            for j in range(i + 1, len(doc_names)):
                doc1, doc2 = doc_names[i], doc_names[j]
                estimated_sim = estimate_jaccard_similarity(
                    signatures[doc1], signatures[doc2])
                if estimated_sim >= threshold:
                    similar_pairs.append((doc1, doc2, estimated_sim))
    else:
        # Compute exact Jaccard similarities
        for i in range(len(doc_names)):
            for j in range(i + 1, len(doc_names)):
                doc1, doc2 = doc_names[i], doc_names[j]
                sim = jaccard_similarity(documents[doc1], documents[doc2])
                if sim >= threshold:
                    similar_pairs.append((doc1, doc2, sim))

    return similar_pairs


def evaluate_minhash(exact_pairs, estimated_pairs):
    """Compute false positives and false negatives"""
    exact_set = {(d1, d2) for d1, d2, _ in exact_pairs}
    estimated_set = {(d1, d2) for d1, d2, _ in estimated_pairs}

    false_positives = len(estimated_set - exact_set)
    false_negatives = len(exact_set - estimated_set)

    return false_positives, false_negatives


def main():
    # Load documents from the minhash directory
    documents = load_documents('minhash')

    # Compute exact similarities
    print("\nComputing exact similarities...")
    exact_pairs = find_similar_pairs(documents, threshold=0.5)
    print(f"Found {len(exact_pairs)} pairs with similarity >= 0.5")
    print("Exact similar pairs:")
    for doc1, doc2, sim in exact_pairs:
        print(f"{doc1} - {doc2}: {sim:.3f}")

    # Test different numbers of hash functions
    hash_function_counts = [50, 100, 200]

    for n_hash_functions in hash_function_counts:
        print(f"\nTesting with {n_hash_functions} hash functions")

        fp_total = fn_total = 0
        num_runs = 5

        for run in range(num_runs):
            minhash = MinHash(n_hash_functions)
            estimated_pairs = find_similar_pairs(documents, threshold=0.5,
                                                 use_minhash=True, minhash_obj=minhash)

            fp, fn = evaluate_minhash(exact_pairs, estimated_pairs)
            fp_total += fp
            fn_total += fn

            # Print estimated pairs for the first run
            if run == 0:
                print(f"\nRun {run + 1} estimated similar pairs:")
                for doc1, doc2, sim in estimated_pairs:
                    print(f"{doc1} - {doc2}: {sim:.3f}")

        # Report averages
        avg_fp = fp_total / num_runs
        avg_fn = fn_total / num_runs

        print(f"\nResults for {n_hash_functions} hash functions:")
        print(f"Average false positives: {avg_fp:.2f}")
        print(f"Average false negatives: {avg_fn:.2f}")


if __name__ == "__main__":
    main()
