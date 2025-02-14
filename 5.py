import numpy as np
from collections import defaultdict
import os
import re
from typing import List, Set, Tuple, Dict


class LSH:
    def __init__(self, signature_matrix: np.ndarray, b: int, r: int, threshold: float):
        """
        Initialize LSH with parameters

        Args:
            signature_matrix: MinHash signature matrix (n_hash_functions × n_documents)
            b: Number of bands
            r: Number of rows per band
            threshold: Similarity threshold
        """
        self.signature_matrix = signature_matrix
        self.b = b
        self.r = r
        self.threshold = threshold
        self.n_docs = signature_matrix.shape[1]

        # Verify that b * r equals number of hash functions
        assert signature_matrix.shape[0] == b * \
            r, "b * r must equal number of hash functions"

    def _hash_band(self, band: np.ndarray) -> int:
        """Hash a band to a bucket number"""
        return hash(tuple(band))

    def find_candidate_pairs(self) -> Set[Tuple[int, int]]:
        """Find candidate pairs using LSH banding technique"""
        candidate_pairs = set()

        # Process each band
        for i in range(self.b):
            # Get the band's rows from signature matrix
            start_row = i * self.r
            end_row = start_row + self.r
            band = self.signature_matrix[start_row:end_row]

            # Hash documents to buckets for this band
            buckets = defaultdict(list)
            for doc_id in range(self.n_docs):
                band_hash = self._hash_band(band[:, doc_id])
                buckets[band_hash].append(doc_id)

            # Add all pairs of documents that hash to the same bucket
            for bucket in buckets.values():
                if len(bucket) > 1:
                    for i in range(len(bucket)):
                        for j in range(i + 1, len(bucket)):
                            candidate_pairs.add(
                                tuple(sorted([bucket[i], bucket[j]])))

        return candidate_pairs


def compute_jaccard_similarity(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Compute Jaccard similarity from MinHash signatures"""
    return np.mean(sig1 == sig2)


def evaluate_lsh(signature_matrix: np.ndarray, b: int, r: int, threshold: float) -> Tuple[int, int]:
    """
    Evaluate LSH performance

    Returns:
        Tuple of (false_positives, false_negatives)
    """
    # Find candidate pairs using LSH
    lsh = LSH(signature_matrix, b, r, threshold)
    candidate_pairs = lsh.find_candidate_pairs()

    # Compute actual similar pairs
    true_similar_pairs = set()
    n_docs = signature_matrix.shape[1]

    for i in range(n_docs):
        for j in range(i + 1, n_docs):
            similarity = compute_jaccard_similarity(
                signature_matrix[:, i],
                signature_matrix[:, j]
            )
            if similarity >= threshold:
                true_similar_pairs.add((i, j))

    # Calculate false positives and negatives
    false_positives = len(candidate_pairs - true_similar_pairs)
    false_negatives = len(true_similar_pairs - candidate_pairs)

    return false_positives, false_negatives


def run_experiment(signature_matrix: np.ndarray, b: int, r: int, threshold: float, n_runs: int = 5) -> Dict:
    """Run LSH experiment multiple times and average results"""
    total_fp = 0
    total_fn = 0

    for _ in range(n_runs):
        fp, fn = evaluate_lsh(signature_matrix, b, r, threshold)
        total_fp += fp
        total_fn += fn

    return {
        'avg_false_positives': total_fp / n_runs,
        'avg_false_negatives': total_fn / n_runs
    }

# Example usage for different configurations


def main():
    # Load your MinHash signature matrices here
    # For example:
    # signatures_50 = np.load('minhash_signatures_50.npy')
    # signatures_100 = np.load('minhash_signatures_100.npy')
    # signatures_200 = np.load('minhash_signatures_200.npy')

    # Configuration parameters
    configs = [
        # For 50 hash functions
        {'signatures': 'signatures_50', 'b': 10, 'r': 5},

        # For 100 hash functions
        {'signatures': 'signatures_100', 'b': 20, 'r': 5},

        # For 200 hash functions
        {'signatures': 'signatures_200', 'b': 40, 'r': 5},
        {'signatures': 'signatures_200', 'b': 20, 'r': 10}
    ]

    thresholds = [0.6, 0.8]

    # Run experiments for each configuration and threshold
    for config in configs:
        for threshold in thresholds:
            print(
                f"\nConfiguration: {config['signatures']}, b={config['b']}, r={config['r']}")
            print(f"Threshold: {threshold}")

            # Replace this with your actual signature matrix
            # results = run_experiment(signatures, config['b'], config['r'], threshold)
            # print(f"Average False Positives: {results['avg_false_positives']:.2f}")
            # print(f"Average False Negatives: {results['avg_false_negatives']:.2f}")


if __name__ == "__main__":
    main()
