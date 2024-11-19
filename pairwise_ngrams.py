import os
import json
import requests
import re
import random
from collections import Counter
from itertools import islice
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from urllib.parse import urlparse

# File paths
URL_LIST_PATH = 'sample_urls1.txt'
CACHE_PATH = 'url_cache.json'

# Parameters
DEFAULT_MIN_WORD_LENGTH = 6
DEFAULT_NGRAM_SIZE = 3
DEFAULT_SAMPLE_SIZE = 60

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
# Load set of valid English words
valid_words = set(words.words())

# Load URLs from file
def load_urls(file_path, sample_size=DEFAULT_SAMPLE_SIZE):
    with open(file_path, 'r') as file:
        urls = [line.strip() for line in file]
    return random.sample(urls, min(sample_size, len(urls)))

# Load or create URL cache
def load_url_cache(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return {}

# Save URL cache
def save_url_cache(file_path, cache):
    with open(file_path, 'w') as file:
        json.dump(cache, file)

# Download article content
def download_article(url, cache):
    if url not in cache:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                text = response.text
                cache[url] = text
        except requests.RequestException:
            print(f"Failed to download article from {url}")
    return cache.get(url, "")

# Extract text content from HTML
def extract_text_from_html(html):
    return re.sub(r'<[^>]+>', '', html)

# Generate n-grams from text
def generate_ngrams(text, n=DEFAULT_NGRAM_SIZE, min_word_length=DEFAULT_MIN_WORD_LENGTH):
    stop_words = set(stopwords.words('english'))
    words_list = [
        lemmatizer.lemmatize(word) for word in re.findall(r'\b\w+\b', text.lower())
        if word not in stop_words and len(word) >= min_word_length and word in valid_words
    ]
    return zip(*[islice(words_list, i, None) for i in range(n)])

# Get the top 10% n-grams
def get_top_ngrams(ngrams, percentage=30):
    ngram_counter = Counter(ngrams)
    most_common = ngram_counter.most_common()
    top_count = max(1, len(most_common) * percentage // 100)
    return [ngram for ngram, count in most_common[:top_count]]

# Generate Article DNA
def generate_article_dna(top_ngrams):
    unique_words = set(word for ngram in top_ngrams for word in ngram)
    return sorted(unique_words)

# Calculate Jaccard index between two sets, normalized to the shorter length set
def normalized_jaccard_index(set1, set2):
    intersection = len(set1.intersection(set2))
    min_length = min(len(set1), len(set2))
    return intersection / min_length if min_length != 0 else 0

# Main process
def main():
    urls = load_urls(URL_LIST_PATH)
    url_cache = load_url_cache(CACHE_PATH)
    article_dnas = {}

    for url in urls:
        # Download and cache the article
        article_html = download_article(url, url_cache)
        save_url_cache(CACHE_PATH, url_cache)

        # Extract text and calculate n-grams
        article_text = extract_text_from_html(article_html)
        ngrams = list(generate_ngrams(article_text))

        # Get top 10% n-grams and generate Article DNA
        top_ngrams = get_top_ngrams(ngrams, percentage=10)
        article_dna = generate_article_dna(top_ngrams)
        article_dnas[url] = set(article_dna)

        # Output Article DNA
        print(f"Article DNA for {url}: {' '.join(article_dna)}\n")

    # Calculate pairwise normalized Jaccard index
    urls_list = list(article_dnas.keys())
    num_articles = len(urls_list)
    jaccard_matrix = np.zeros((num_articles, num_articles))

    for i in range(num_articles):
        for j in range(num_articles):
            if i != j:
                jaccard_matrix[i][j] = 1 - normalized_jaccard_index(article_dnas[urls_list[i]], article_dnas[urls_list[j]])

    # Perform hierarchical clustering
    linked = linkage(jaccard_matrix, 'ward')

    # Plot hierarchical clustering dendrogram
    plt.figure(figsize=(10, 8))
    dendrogram(linked, labels=[urlparse(url).netloc for url in urls_list], orientation='right', distance_sort='descending', show_leaf_counts=True)
    plt.title('Hierarchical Clustering Dendrogram of Articles based on Normalized Jaccard Distance')
    plt.xlabel('Distance')
    plt.ylabel('Article URL Domain')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
