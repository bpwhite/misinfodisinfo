import numpy as np
import difflib
import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk import pos_tag, word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.metrics import edit_distance
import json
import os
import scipy.stats as stats
from sklearn.metrics import jaccard_score
from tqdm import tqdm

# Download NLTK data files (only required once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Cache file to store downloaded content persistently
CACHE_FILE = "url_cache.json"

# Load cache from file if it exists
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r') as cache_file:
        url_cache = json.load(cache_file)
else:
    url_cache = {}

# Variable to limit the number of articles imported to analyze
MAX_ARTICLES = 10
# Variable to limit the minimum sentence length for analysis (after stop words are removed)
MIN_SENTENCE_LENGTH = 8
# Stop words set
STOP_WORDS = set(stopwords.words('english'))

def save_cache():
    with open(CACHE_FILE, 'w') as cache_file:
        json.dump(url_cache, cache_file)

def align_sentences(sentence1, sentence2):
    words1 = sentence1.split()
    words2 = sentence2.split()

    # Create an alignment matrix with difflib SequenceMatcher ratio
    matrix = np.zeros((len(words1), len(words2)))
    
    for i, word1 in enumerate(words1):
        for j, word2 in enumerate(words2):
            matcher = difflib.SequenceMatcher(None, word1, word2)
            similarity = matcher.ratio()
            matrix[i, j] = similarity
    
    # Find the best alignment using the matrix
    aligned_sentence1 = []
    aligned_sentence2 = []
    
    i, j = 0, 0
    while i < len(words1) and j < len(words2):
        if matrix[i, j] >= 0.5:  # If words are at least 50% similar
            aligned_sentence1.append(words1[i])
            aligned_sentence2.append(words2[j])
            i += 1
            j += 1
        elif np.max(matrix[i, :]) > np.max(matrix[:, j]):
            aligned_sentence1.append(words1[i])
            aligned_sentence2.append("_")  # No match found in sentence2
            i += 1
        else:
            aligned_sentence1.append("_")  # No match found in sentence1
            aligned_sentence2.append(words2[j])
            j += 1
    
    # Append remaining words
    while i < len(words1):
        aligned_sentence1.append(words1[i])
        aligned_sentence2.append("_")
        i += 1
    while j < len(words2):
        aligned_sentence1.append("_")
        aligned_sentence2.append(words2[j])
        j += 1

    return ' '.join(aligned_sentence1), ' '.join(aligned_sentence2)

def grab_and_clean_text_from_website(url):
    if url in url_cache:
        return url_cache[url]
    
    try:
        # Fetch the website content
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text from the HTML
        text = soup.get_text()
        
        # Clean the text by removing extra whitespace and non-alphanumeric characters
        cleaned_text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        cleaned_text = re.sub(r'[^\w\s\.,!?]', '', cleaned_text)  # Remove non-alphanumeric characters except punctuation
        
        # Store the cleaned text in the cache
        url_cache[url] = cleaned_text.strip()
        save_cache()
        
        return cleaned_text.strip()
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"

def remove_stop_words(sentence):
    # Tokenize the sentence
    tokens = word_tokenize(sentence)
    # Remove stop words
    filtered_tokens = [word for word in tokens if word.lower() not in STOP_WORDS]
    return ' '.join(filtered_tokens)

def convert_sentence_to_grammar_tokens(sentence):
    # Tokenize the sentence
    tokens = word_tokenize(sentence)
    
    # Get part of speech tags for each token
    tagged_tokens = pos_tag(tokens)
    
    return tagged_tokens

def process_text_and_align_tokens(url, known_tokens):
    # Step 1: Grab and clean text from the website
    cleaned_text = grab_and_clean_text_from_website(url)
    
    # Step 2: Split the text into sentences
    sentences = sent_tokenize(cleaned_text)
    
    # Step 3: Remove stop words and filter sentences by minimum length
    filtered_sentences = []
    for sentence in sentences:
        filtered_sentence = remove_stop_words(sentence)
        if len(filtered_sentence.split()) >= MIN_SENTENCE_LENGTH:
            filtered_sentences.append(filtered_sentence)
    
    # Step 4: Convert each sentence to grammar tokens
    tokenized_sentences = [convert_sentence_to_grammar_tokens(sentence) for sentence in filtered_sentences]
    
    # Step 5: Align grammar tokens to the known set of grammar tokens
    aligned_results = []
    for tagged_tokens in tokenized_sentences:
        sentence_tokens = [tag for word, tag in tagged_tokens]
        known_token_tags = [tag for word, tag in known_tokens]
        aligned_sentence, aligned_known = align_sentences(' '.join(sentence_tokens), ' '.join(known_token_tags))
        aligned_results.append((aligned_sentence, aligned_known))
    
    return aligned_results

def read_urls_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            urls = [line.strip() for line in file.readlines()]
        return urls[:MAX_ARTICLES]  # Limit the number of articles to analyze
    except FileNotFoundError as e:
        return f"An error occurred: {e}"

def compare_articles(urls):
    sentences_by_url = {}

    # Step 1: Extract and tokenize sentences for each URL
    for url in tqdm(urls, desc="Extracting and tokenizing sentences"):
        cleaned_text = grab_and_clean_text_from_website(url)
        sentences = sent_tokenize(cleaned_text)
        # Remove stop words and filter sentences by minimum length
        filtered_sentences = []
        for sentence in sentences:
            filtered_sentence = remove_stop_words(sentence)
            if len(filtered_sentence.split()) >= MIN_SENTENCE_LENGTH:
                filtered_sentences.append(filtered_sentence)
        tokenized_sentences = [convert_sentence_to_grammar_tokens(sentence) for sentence in filtered_sentences]
        sentences_by_url[url] = {'tokenized': tokenized_sentences}

    # Step 2: Pairwise comparison between articles
    results = []
    urls_list = list(sentences_by_url.keys())
    for i in tqdm(range(len(urls_list)), desc="Comparing articles", unit="pair"):
        for j in range(i + 1, len(urls_list)):
            url1, url2 = urls_list[i], urls_list[j]
            levenstein_distances = []
            jaccard_distances = []

            # Compare grammar tokenized sentences
            for sent1 in tqdm(sentences_by_url[url1]['tokenized'], desc=f"Comparing tokenized sentences between {url1} and {url2}", leave=False):
                for sent2 in sentences_by_url[url2]['tokenized']:
                    tokens1 = [tag for word, tag in sent1]
                    tokens2 = [tag for word, tag in sent2]
                    # Calculate Levenshtein distance and standardize it to 100
                    max_len = max(len(tokens1), len(tokens2))
                    levenstein_distance = edit_distance(tokens1, tokens2)
                    standardized_levenstein = (levenstein_distance / max_len) * 100 if max_len > 0 else 0
                    levenstein_distances.append(standardized_levenstein)
                    # Calculate Jaccard distance and standardize it to 100
                    set1 = set(tokens1)
                    set2 = set(tokens2)
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)
                    jaccard_similarity = (intersection / union) if union != 0 else 1
                    standardized_jaccard = (1 - jaccard_similarity) * 100
                    jaccard_distances.append(standardized_jaccard)

            # Calculate average standardized distances and 95% confidence intervals for this pair
            avg_levenstein_distance = np.mean(levenstein_distances)
            std_error_levenstein = stats.sem(levenstein_distances)
            confidence_interval_levenstein = stats.t.interval(0.95, len(levenstein_distances) - 1, loc=avg_levenstein_distance, scale=std_error_levenstein)

            avg_jaccard_distance = np.mean(jaccard_distances)
            std_error_jaccard = stats.sem(jaccard_distances)
            confidence_interval_jaccard = stats.t.interval(0.95, len(jaccard_distances) - 1, loc=avg_jaccard_distance, scale=std_error_jaccard)

            results.append((url1, url2, avg_levenstein_distance, confidence_interval_levenstein, avg_jaccard_distance, confidence_interval_jaccard))

    return results

# Example usage of reading URLs from a file
urls = read_urls_from_file("sample_urls1.txt")
if isinstance(urls, list):
    comparison_results = compare_articles(urls)
    for result in comparison_results:
        (url1, url2, avg_levenstein_distance, confidence_interval_levenstein, avg_jaccard_distance, 
        confidence_interval_jaccard) = result
        print(f"Comparison between {url1} and {url2}:")
        print(f"Average Standardized Levenshtein Distance: {avg_levenstein_distance}")
        print(f"95% Confidence Interval for Levenshtein Distance: {confidence_interval_levenstein}")
        print(f"Average Standardized Jaccard Distance: {avg_jaccard_distance}")
        print(f"95% Confidence Interval for Jaccard Distance: {confidence_interval_jaccard}")

    # Determine top 3 best matches and bottom 3 worst matches for each metric
    metrics = ['avg_levenstein_distance', 'avg_jaccard_distance']
    for metric in metrics:
        if metric == 'avg_levenstein_distance':
            sorted_results = sorted(comparison_results, key=lambda x: x[2])
        elif metric == 'avg_jaccard_distance':
            sorted_results = sorted(comparison_results, key=lambda x: x[4])
        print(f"\nTop 3 best matches for {metric}:")
        for best_match in sorted_results[:3]:
            print(f"{best_match[0]} vs {best_match[1]}: {best_match[2 if metric == 'avg_levenstein_distance' else 4]}")
        print(f"\nBottom 3 worst matches for {metric}:")
        for worst_match in sorted_results[-3:]:
            print(f"{worst_match[0]} vs {worst_match[1]}: {worst_match[2 if metric == 'avg_levenstein_distance' else 4]}")
else:
    print(urls)
