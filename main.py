import numpy as np
import difflib
import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk import pos_tag, word_tokenize, sent_tokenize

# Download NLTK data files (only required once)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

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

# Example usage
sentence1 = "The quick brown fox jumps over the lazy dog"
sentence2 = "A fast brown animal jumps over a sleeping dog"

aligned1, aligned2 = align_sentences(sentence1, sentence2)
print("Aligned Sentence 1:", aligned1)
print("Aligned Sentence 2:", aligned2)

# Example usage of grab_and_clean_text_from_website
url = "https://www.example.com"
cleaned_text = grab_and_clean_text_from_website(url)
print("Cleaned Text from Website:", cleaned_text)

# Example usage of convert_sentence_to_grammar_tokens
sentence = "The quick brown fox jumps over the lazy dog"
grammar_tokens = convert_sentence_to_grammar_tokens(sentence)
print("Grammar Tokens:", grammar_tokens)

# Example usage of process_text_and_align_tokens
known_tokens = convert_sentence_to_grammar_tokens("The quick brown fox jumps over the lazy dog")
aligned_token_results = process_text_and_align_tokens(url, known_tokens)
for aligned_sentence, aligned_known in aligned_token_results:
    print("Aligned Grammar Tokens:")
    print("Sentence:", aligned_sentence)
    print("Known Tokens:", aligned_known)
