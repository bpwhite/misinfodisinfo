import requests
from bs4 import BeautifulSoup
import re
from collections import Counter
import nltk
from nltk import pos_tag, word_tokenize
from textblob import TextBlob
import json
import os
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from urllib.parse import urlparse
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Cache file path
CACHE_FILE_PATH = 'url_cache.json'
STOP_WORDS = set(stopwords.words('english'))

# Load the cache
def load_cache():
    if os.path.exists(CACHE_FILE_PATH):
        with open(CACHE_FILE_PATH, 'r') as f:
            return json.load(f)
    return {}

# Save to cache
def save_cache(cache):
    with open(CACHE_FILE_PATH, 'w') as f:
        json.dump(cache, f)

# Function to get cleaned text from a URL
def get_cleaned_text(url, cache):
    if url in cache:
        print(f"Using cached data for {url}")
        return cache[url]
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        cleaned_text = re.sub(r'[^A-Za-z0-9\.\s]', '', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        cache[url] = cleaned_text
        save_cache(cache)
        return cleaned_text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return ""

# Function to calculate linguistic metrics
def calculate_metrics(text, min_sentence_length=5):
    word_list = [word for word in text.split() if word.lower() not in STOP_WORDS]
    sentence_list = re.split(r'[.!?]', text)
    sentence_list = [sentence.strip() for sentence in sentence_list if len(sentence.split()) >= min_sentence_length]
    word_count = len(word_list)
    sentence_count = len(sentence_list)
    average_word_length = sum(len(word) for word in word_list) / word_count if word_count > 0 else 0
    unique_words = set(word_list)
    lexical_diversity = len(unique_words) / word_count if word_count > 0 else 0
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    pos_counts = Counter()
    for sentence in sentence_list:
        tokens = word_tokenize(sentence)
        tokens = [word for word in tokens if word.lower() not in STOP_WORDS]
        pos_tags = pos_tag(tokens)
        pos_counts.update(tag for word, tag in pos_tags)
    avg_nouns = pos_counts['NN'] / sentence_count if sentence_count > 0 else 0
    avg_verbs = pos_counts['VB'] / sentence_count if sentence_count > 0 else 0
    avg_adverbs = pos_counts['RB'] / sentence_count if sentence_count > 0 else 0
    avg_adjectives = pos_counts['JJ'] / sentence_count if sentence_count > 0 else 0
    num_adverbs_adjectives = pos_counts['RB'] + pos_counts['JJ']
    num_nouns_verbs = pos_counts['NN'] + pos_counts['VB']
    ratio_adverbs_adjectives = (num_adverbs_adjectives / num_nouns_verbs) if num_nouns_verbs > 0 else 0
    sentiment_scores = [TextBlob(sentence).sentiment.polarity for sentence in sentence_list]
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    total_characters = sum(len(word) for word in word_list)
    avg_characters_per_sentence = total_characters / sentence_count if sentence_count > 0 else 0
    unique_word_ratio = len(unique_words) / word_count if word_count > 0 else 0
    return {
        "Word Count": word_count,
        "Sentence Count": sentence_count,
        "Average Word Length": average_word_length,
        "Lexical Diversity": lexical_diversity,
        "Average Sentence Length": avg_sentence_length,
        "Average Nouns per Sentence": avg_nouns,
        "Average Verbs per Sentence": avg_verbs,
        "Average Adverbs per Sentence": avg_adverbs,
        "Average Adjectives per Sentence": avg_adjectives,
        "Average Ratio of Adverbs and Adjectives to Nouns and Verbs": ratio_adverbs_adjectives,
        "Average Sentiment": avg_sentiment,
        "Average Characters per Sentence": avg_characters_per_sentence,
        "Unique Word Ratio": unique_word_ratio
    }

# Main function to perform PCA
def main():
    # Load cache
    cache = load_cache()

    # Read URLs from file
    try:
        with open("sample_urls1.txt", 'r') as file:
            urls = [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        print("Error: sample_urls1.txt file not found.")
        return

    # Calculate metrics for each URL
    metrics_list = []
    labels = []
    for url in urls:
        text = get_cleaned_text(url, cache)
        metrics = calculate_metrics(text)
        metrics_list.append(list(metrics.values()))
        domain = urlparse(url).netloc
        labels.append(domain)

    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame(metrics_list, columns=list(metrics.keys()))
    
    # Standardize the metrics
    scaler = StandardScaler()
    standardized_metrics = scaler.fit_transform(metrics_df)
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(standardized_metrics)

    # Plot PCA result with color coding by domain
    unique_domains = list(set(labels))
    domain_colors = {domain: plt.cm.jet(float(i) / len(unique_domains)) for i, domain in enumerate(unique_domains)}
    
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        plt.scatter(pca_result[i, 0], pca_result[i, 1], color=domain_colors[label], alpha=0.6)
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Analysis of Linguistic Metrics by Domain (Color Coded)')
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=domain, markersize=10, markerfacecolor=color) for domain, color in domain_colors.items()], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    # Print PCA statistical results
    print("Explained variance ratio of each principal component:")
    for i, variance_ratio in enumerate(pca.explained_variance_ratio_):
        print(f"Principal Component {i+1}: {variance_ratio:.2f} ({variance_ratio * 100:.2f}% of the variance)")
    
    print("\nCumulative explained variance:")
    cumulative_variance = sum(pca.explained_variance_ratio_)
    print(f"Cumulative explained variance: {cumulative_variance:.2f} ({cumulative_variance * 100:.2f}% of the variance)")
    
    print("\nPrincipal Component Axes Interpretation:")
    for i in range(len(pca.components_)):
        print(f"Principal Component {i+1} is influenced by the following features:")
        feature_contributions = pca.components_[i]
        for feature, contribution in zip(metrics_df.columns, feature_contributions):
            print(f"  {feature}: {contribution:.2f}")
        print()

if __name__ == "__main__":
    main()
