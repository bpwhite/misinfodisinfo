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
import configparser
import time
import pyphen

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Cache file path
CACHE_FILE_PATH = 'url_cache.json'
STOP_WORDS = set(stopwords.words('english'))

# Config file paths
CONFIG_FILE_PATH = 'metrics_config.ini'
DOMAIN_COLOR_CONFIG_FILE = 'domain_color_config.ini'

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

# Load the configuration
def load_config():
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE_PATH)
    return config['METRICS'] if 'METRICS' in config else {}

# Load the domain color configuration
def load_domain_color_config():
    color_config = configparser.ConfigParser()
    color_config.read(DOMAIN_COLOR_CONFIG_FILE)
    return color_config['COLORS'] if 'COLORS' in color_config else {}

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
        time.sleep(1)
        return cleaned_text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return ""

# Function to calculate linguistic metrics
def calculate_metrics(text, config, min_sentence_length=5):
    word_list = [word for word in text.split() if word.lower() not in STOP_WORDS]
    sentence_list = re.split(r'[.!?]', text)
    sentence_list = [sentence.strip() for sentence in sentence_list if len(sentence.split()) >= min_sentence_length]
    word_count = len(word_list)
    sentence_count = len(sentence_list)
    metrics = {}

    if config.getboolean('word_count', fallback=True):
        metrics["Word Count"] = word_count
    if config.getboolean('sentence_count', fallback=True):
        metrics["Sentence Count"] = sentence_count
    if config.getboolean('average_word_length', fallback=True):
        metrics["Average Word Length"] = sum(len(word) for word in word_list) / word_count if word_count > 0 else 0
    if config.getboolean('lexical_diversity', fallback=True):
        unique_words = set(word_list)
        metrics["Lexical Diversity"] = len(unique_words) / word_count if word_count > 0 else 0
    if config.getboolean('average_sentence_length', fallback=True):
        metrics["Average Sentence Length"] = word_count / sentence_count if sentence_count > 0 else 0
    
    pos_counts = Counter()
    all_pos_tags = []
    for sentence in sentence_list:
        tokens = word_tokenize(sentence)
        tokens = [word for word in tokens if word.lower() not in STOP_WORDS]
        pos_tags = pos_tag(tokens)
        pos_counts.update(tag for word, tag in pos_tags)
        all_pos_tags.extend(pos_tags)
    
    if config.getboolean('average_nouns', fallback=True):
        metrics["Average Nouns per Sentence"] = pos_counts['NN'] / sentence_count if sentence_count > 0 else 0
    if config.getboolean('average_verbs', fallback=True):
        metrics["Average Verbs per Sentence"] = pos_counts['VB'] / sentence_count if sentence_count > 0 else 0
    if config.getboolean('average_adverbs', fallback=True):
        metrics["Average Adverbs per Sentence"] = pos_counts['RB'] / sentence_count if sentence_count > 0 else 0
    if config.getboolean('average_adjectives', fallback=True):
        metrics["Average Adjectives per Sentence"] = pos_counts['JJ'] / sentence_count if sentence_count > 0 else 0
    if config.getboolean('ratio_adverbs_adjectives', fallback=True):
        num_adverbs_adjectives = pos_counts['RB'] + pos_counts['JJ']
        num_nouns_verbs = pos_counts['NN'] + pos_counts['VB']
        metrics["Average Ratio of Adverbs and Adjectives to Nouns and Verbs"] = (num_adverbs_adjectives / num_nouns_verbs) if num_nouns_verbs > 0 else 0
    if config.getboolean('average_sentiment', fallback=True):
        sentiment_scores = [TextBlob(sentence).sentiment.polarity for sentence in sentence_list]
        metrics["Average Sentiment"] = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    if config.getboolean('average_characters_per_sentence', fallback=True):
        total_characters = sum(len(word) for word in word_list)
        metrics["Average Characters per Sentence"] = total_characters / sentence_count if sentence_count > 0 else 0
    if config.getboolean('unique_word_ratio', fallback=True):
        unique_words = set(word_list)
        metrics["Unique Word Ratio"] = len(unique_words) / word_count if word_count > 0 else 0
    if config.getboolean('type_token_ratio', fallback=True):
        metrics["Type-Token Ratio"] = len(set(word_list)) / len(word_list) if word_list else 0
    if config.getboolean('syllable_count_per_word', fallback=True):
        dic = pyphen.Pyphen(lang='en')
        metrics["Syllable Count per Word"] = sum(len(dic.inserted(word).split('-')) for word in word_list) / word_count if word_count > 0 else 0
    if config.getboolean('polysyllabic_word_count', fallback=True):
        dic = pyphen.Pyphen(lang='en')
        metrics["Polysyllabic Word Count"] = sum(1 for word in word_list if len(dic.inserted(word).split('-')) >= 3)
    if config.getboolean('passive_voice_frequency', fallback=True):
        metrics["Passive Voice Frequency"] = sum(1 for sentence in sentence_list if 'by' in sentence and ('was' in sentence or 'were' in sentence)) / sentence_count if sentence_count > 0 else 0
    if config.getboolean('pronoun_use', fallback=True):
        metrics["Pronoun Use"] = sum(1 for word, tag in all_pos_tags if tag.startswith('PRP')) / word_count if word_count > 0 else 0
    if config.getboolean('named_entity_count', fallback=True):
        named_entity_count = len(re.findall(r'\b[A-Z][a-z]*\b', text))  # Approximate method for named entity count
        metrics["Named Entity Count"] = named_entity_count
    
    return metrics

# Main function to perform PCA
def main():
    # Load cache
    cache = load_cache()

    # Load configuration
    config = load_config()
    domain_color_config = load_domain_color_config()

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
    for idx, url in enumerate(urls, start=1):
        text = get_cleaned_text(url, cache)
        metrics = calculate_metrics(text, config)
        metrics_list.append(list(metrics.values()))
        domain = urlparse(url).netloc
        labels.append(f"Article {idx}: {domain}")

    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame(metrics_list, columns=list(metrics.keys()))
    
    # Standardize the metrics
    scaler = StandardScaler()
    standardized_metrics = scaler.fit_transform(metrics_df)
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(standardized_metrics)

    # Plot PCA result with color coding by domain
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        color = domain_color_config.get(label.split(': ')[1], plt.cm.jet(float(i) / len(set(labels))))
        plt.scatter(pca_result[i, 0], pca_result[i, 1], color=color, alpha=0.6)
        plt.text(pca_result[i, 0], pca_result[i, 1], str(i + 1), fontsize=9, ha='right')  # Add article number to the plot
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Analysis of Linguistic Metrics by Domain (Color Coded)')
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor=domain_color_config.get(label.split(': ')[1], 'gray')) for label in labels], bbox_to_anchor=(1.05, 1), loc='upper left')
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
