import requests
from bs4 import BeautifulSoup
import re
import textstat
from collections import Counter
import nltk
from nltk import pos_tag, word_tokenize
from textblob import TextBlob
import json
import os
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Cache file name
CACHE_FILE = "pairwise_url_cache.json"

# Load stop words
STOP_WORDS = set(stopwords.words('english'))

# Load cache if it exists
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

# Save cache to file
def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

# Function to get cleaned text from a URL
def get_cleaned_text(url, cache):
    if url in cache:
        print(f"Using cached data for {url}")
        return cache[url]
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text from paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        
        # Clean text (remove extra whitespace, special characters, etc.)
        cleaned_text = re.sub(r'[^A-Za-z0-9\.\s]', '', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # Save to cache
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
    
    # Removing empty strings and short sentences from the list of sentences
    sentence_list = [sentence.strip() for sentence in sentence_list if len(sentence.split()) >= min_sentence_length]
    
    word_count = len(word_list)
    sentence_count = len(sentence_list)
    
    # Average word length
    average_word_length = sum(len(word) for word in word_list) / word_count if word_count > 0 else 0
    
    # Lexical diversity
    unique_words = set(word_list)
    lexical_diversity = len(unique_words) / word_count if word_count > 0 else 0
    
    # Readability score (Flesch Reading Ease)
    readability_score = textstat.flesch_reading_ease(text) if text else 0
    
    # Tokenize sentences and perform part-of-speech tagging
    pos_counts = Counter()
    for sentence in sentence_list:
        tokens = word_tokenize(sentence)
        tokens = [word for word in tokens if word.lower() not in STOP_WORDS]  # Remove stop words from tokens
        pos_tags = pos_tag(tokens)
        pos_counts.update(tag for word, tag in pos_tags)
    
    # Calculate average number of each grammar type per sentence
    avg_nouns = pos_counts['NN'] / sentence_count if sentence_count > 0 else 0
    avg_verbs = pos_counts['VB'] / sentence_count if sentence_count > 0 else 0
    avg_adverbs = pos_counts['RB'] / sentence_count if sentence_count > 0 else 0
    avg_adjectives = pos_counts['JJ'] / sentence_count if sentence_count > 0 else 0
    
    # Calculate average ratio of adverbs and adjectives to the sum of nouns and verbs
    num_adverbs_adjectives = pos_counts['RB'] + pos_counts['JJ']
    num_nouns_verbs = pos_counts['NN'] + pos_counts['VB']
    ratio_adverbs_adjectives = (num_adverbs_adjectives / num_nouns_verbs) if num_nouns_verbs > 0 else 0
    
    # Calculate average sentiment for all sentences
    sentiment_scores = [TextBlob(sentence).sentiment.polarity for sentence in sentence_list]
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    
    return {
        "Word Count": word_count,
        "Sentence Count": sentence_count,
        "Average Word Length": average_word_length,
        "Lexical Diversity": lexical_diversity,
        "Readability Score": readability_score,
        "Average Nouns per Sentence": avg_nouns,
        "Average Verbs per Sentence": avg_verbs,
        "Average Adverbs per Sentence": avg_adverbs,
        "Average Adjectives per Sentence": avg_adjectives,
        "Average Ratio of Adverbs and Adjectives to Nouns and Verbs": ratio_adverbs_adjectives,
        "Average Sentiment": avg_sentiment
    }

# Function to compare metrics between two articles
def compare_metrics(metrics1, metrics2):
    comparison = {}
    for key in metrics1:
        if isinstance(metrics1[key], (int, float)) and isinstance(metrics2[key], (int, float)):
            comparison[key] = metrics1[key] - metrics2[key]
    return comparison

# Main function to process two URLs
def main():
    # Load cache
    cache = load_cache()
    
    # Read URLs from file
    try:
        with open("compare2_url.txt", 'r') as file:
            urls = file.readlines()
            url1 = urls[0].strip()
            url2 = urls[1].strip()
    except FileNotFoundError:
        print("Error: compare2_url.txt file not found.")
        return
    except IndexError:
        print("Error: compare2_url.txt must contain at least two URLs.")
        return
    
    # Get cleaned text from URLs
    text1 = get_cleaned_text(url1, cache)
    text2 = get_cleaned_text(url2, cache)
    
    # Calculate metrics for each URL
    metrics1 = calculate_metrics(text1)
    metrics2 = calculate_metrics(text2)
    
    # Display metrics
    print("\nMetrics for URL 1:")
    for metric, value in metrics1.items():
        print(f"{metric}: {value}")
    
    print("\nMetrics for URL 2:")
    for metric, value in metrics2.items():
        print(f"{metric}: {value}")
    
    # Compare metrics between the two URLs
    comparison = compare_metrics(metrics1, metrics2)
    print("\nComparison between URL 1 and URL 2:")
    for metric, value in comparison.items():
        print(f"Difference in {metric}: {value}")

if __name__ == "__main__":
    main()
