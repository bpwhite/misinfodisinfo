import numpy as np
import difflib
import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk import pos_tag, word_tokenize, sent_tokenize
from nltk.metrics import edit_distance
import json
import os
import scipy.stats as stats

# Download NLTK data files (only required once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Cache file to store downloaded content persistently
CACHE_FILE = "url_cache.json"

# Load cache from file if it exists
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r') as cache_file:
        url_cache = json.load(cache_file)
else:
    url_cache = {}

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
    
    # Step 3: Convert each sentence to grammar tokens
    tokenized_sentences = [convert_sentence_to_grammar_tokens(sentence) for sentence in sentences]
    
    # Step 4: Align grammar tokens to the known set of grammar tokens
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
        return urls
    except FileNotFoundError as e:
        return f"An error occurred: {e}"

def compare_articles_using_levenshtein(urls):
    sentences_by_url = {}

    # Step 1: Extract and tokenize sentences for each URL
    for url in urls:
        cleaned_text = grab_and_clean_text_from_website(url)
        sentences = sent_tokenize(cleaned_text)
        tokenized_sentences = [convert_sentence_to_grammar_tokens(sentence) for sentence in sentences]
        sentences_by_url[url] = tokenized_sentences

    # Step 2: Pairwise comparison between articles
    results = []
    urls_list = list(sentences_by_url.keys())
    for i in range(len(urls_list)):
        for j in range(i + 1, len(urls_list)):
            url1, url2 = urls_list[i], urls_list[j]
            distances = []
            for sent1 in sentences_by_url[url1]:
                for sent2 in sentences_by_url[url2]:
                    tokens1 = [tag for word, tag in sent1]
                    tokens2 = [tag for word, tag in sent2]
                    distance = edit_distance(tokens1, tokens2)
                    distances.append(distance)
            # Calculate average Levenshtein distance and 95% confidence interval for this pair
            avg_distance = np.mean(distances)
            std_error = stats.sem(distances)
            confidence_interval = stats.t.interval(0.95, len(distances) - 1, loc=avg_distance, scale=std_error)
            results.append((url1, url2, avg_distance, confidence_interval))

    return results

# Example usage of reading URLs from a file
urls = read_urls_from_file("sample_urls1.txt")
if isinstance(urls, list):
    comparison_results = compare_articles_using_levenshtein(urls)
    for result in comparison_results:
        url1, url2, avg_distance, confidence_interval = result
        print(f"Comparison between {url1} and {url2}:")
        print(f"Average Levenshtein Distance: {avg_distance}")
        print(f"95% Confidence Interval: {confidence_interval}")
else:
    print(urls)
