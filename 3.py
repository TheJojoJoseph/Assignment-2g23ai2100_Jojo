import os


def generate_ngrams(text, n=3):
    return {text[i:i+n] for i in range(len(text)-n+1)}


def read_docs(directory):
    return {f: generate_ngrams(open(os.path.join(directory, f)).read()) for f in os.listdir(directory) if f.endswith('.txt')}


def jaccard_sim(set1, set2):
    return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0


def find_bands(t, threshold):
    return min(((r, t//r) for r in range(1, t+1) if t % r == 0), key=lambda rb: abs((1/rb[1])**(1/rb[0]) - threshold))


def lsh_prob(sim, r, b):
    return 1 - (1 - sim**r)**b


def main():
    t, threshold = 160, 0.7
    r, b = find_bands(t, threshold)
    print(f" Optimal r={r}, b={b}")
    docs = read_docs('minhash')

    for i, d1 in enumerate(docs):
        for d2 in list(docs)[i+1:]:
            sim = jaccard_sim(docs[d1], docs[d2])
            prob = lsh_prob(sim, r, b)
            print(
                f"{d1} - {d2}: Jaccard={sim:.4f}, LSH={prob:.4f}, {'Detected' if prob > 0.5 else 'Missed'}")


if __name__ == "__main__":
    main()
