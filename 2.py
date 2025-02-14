import time
import random
import mmh3
import numpy as np
from typing import List, Set, Dict
import matplotlib.pyplot as plt


def generate_k_grams(text: str, k: int = 3) -> Set[str]:
    text = text.lower().replace('\n', ' ')
    return set(text[i:i+k] for i in range(len(text) - k + 1))


def create_hash_function(seed: int, m: int = 10000):
    def hash_func(value: str) -> int:
        return mmh3.hash(value, seed) % m
    return hash_func


def compute_minhash_signature(k_grams: Set[str], hash_functions: List) -> List[int]:
    signature = []
    for h in hash_functions:
        if not k_grams:
            signature.append(float('inf'))
            continue
        min_hash = min(h(gram) for gram in k_grams)
        signature.append(min_hash)
    return signature


def estimate_jaccard_similarity(sig1: List[int], sig2: List[int]) -> float:
    """Estimate Jaccard similarity from minhash signatures."""
    if not sig1 or not sig2 or len(sig1) != len(sig2):
        return 0.0
    matches = sum(1 for i in range(len(sig1)) if sig1[i] == sig2[i])
    return matches / len(sig1)


def compute_exact_jaccard(set1: Set[str], set2: Set[str]) -> float:
    """Compute exact Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def run_experiment(doc1: str, doc2: str, t_values: List[int], num_trials: int = 10):
    """Run experiments for different values of t."""
    k = 3  # k-gram size
    m = 10000  # hash range

    # Generate k-grams
    kgrams1 = generate_k_grams(doc1, k)
    kgrams2 = generate_k_grams(doc2, k)

    # Calculate exact Jaccard similarity
    exact_similarity = compute_exact_jaccard(kgrams1, kgrams2)

    results = {
        't_values': t_values,
        'mean_errors': [],
        'std_errors': [],
        'times': [],
        'estimated_similarities': []  # Added to store the actual similarity estimates
    }

    for t in t_values:
        errors = []
        similarities = []
        total_time = 0

        for trial in range(num_trials):
            start_time = time.time()

            # Create hash functions
            hash_functions = [create_hash_function(
                seed=i + trial * t, m=m) for i in range(t)]

            # Compute signatures
            sig1 = compute_minhash_signature(kgrams1, hash_functions)
            sig2 = compute_minhash_signature(kgrams2, hash_functions)

            # Calculate estimated similarity
            estimated_similarity = estimate_jaccard_similarity(sig1, sig2)

            end_time = time.time()
            total_time += (end_time - start_time)

            # Calculate error
            error = abs(exact_similarity - estimated_similarity)
            errors.append(error)
            similarities.append(estimated_similarity)

        results['mean_errors'].append(np.mean(errors))
        results['std_errors'].append(np.std(errors))
        results['times'].append(total_time / num_trials)
        results['estimated_similarities'].append(np.mean(similarities))

    return results, exact_similarity


def main():
    # Read documents
    try:
        with open('minhash/D1.txt', 'r', encoding='utf-8') as f:
            doc1 = f.read()
        with open('minhash/D2.txt', 'r', encoding='utf-8') as f:
            doc2 = f.read()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure D1.txt and D2.txt exist in the minhash directory")
        return
    except Exception as e:
        print(f"An error occurred while reading the files: {e}")
        return

    # Test t values as specified in the question
    t_values = [20, 60, 150, 300, 600]

    try:
        results, exact_similarity = run_experiment(doc1, doc2, t_values)

        # Print results
        print(f"\nExact Jaccard Similarity: {exact_similarity:.4f}")
        print("\nEstimated Similarities for different values of t:")
        print("t\tEstimated Similarity")
        print("-" * 30)
        for i, t in enumerate(results['t_values']):
            print(f"{t}\t{results['estimated_similarities'][i]:.4f}")

        # Print timing information
        print("\nComputation times:")
        print("t\tTime (seconds)")
        print("-" * 30)
        for i, t in enumerate(results['t_values']):
            print(f"{t}\t{results['times'][i]:.4f}")

    except Exception as e:
        print(f"An error occurred during the experiment: {e}")
        return


if __name__ == "__main__":
    main()
