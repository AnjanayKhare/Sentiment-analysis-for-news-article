import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
import re

custom_stopwords_files = [
    "StopWords_Auditor.txt",
    "StopWords_Currencies.txt",
    "StopWords_DatesandNumbers.txt",
    "StopWords_Generic.txt",
    "StopWords_GenericLong.txt",
    "StopWords_Geographic.txt",
    "StopWords_Names.txt"
]

def load_words_from_file(file_path):
    with open(file_path, 'r') as file:
        words = file.read().splitlines()
    return words

positive_words = load_words_from_file("positive-words.txt")
negative_words = load_words_from_file("negative-words.txt")

def fetch_text_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            target_div = soup.find('div', class_='td-post-content tagdiv-type')
            header = soup.find('h1', class_='entry-title')
            temp = []

            if header:
                heading = header.get_text()
                temp.append(heading)

            if target_div:
                p_tags = target_div.find_all('p')
                for p in p_tags:
                    t = p.get_text()
                    temp.append(t)

            return ' '.join(temp)
        else:
            return ""
    except Exception as e:
        print(f"Error fetching URL: {e}")
        return ""

def count_complex_words(text, custom_stopwords):
    words = word_tokenize(text)
    complex_words = [word for word in words if count_syllables(word, custom_stopwords) > 2]
    return len(complex_words)

custom_stopwords = set()
for stopwords_file in custom_stopwords_files:
    with open(stopwords_file, 'r') as file:
        for line in file:
            stop_word = line.split("|")[0].strip()
            custom_stopwords.add(stop_word)

def perform_sentiment_analysis(text, custom_stopwords):
    words = word_tokenize(text)
    words = [word for word in words if word not in custom_stopwords]

    positive_score = sum(1 for word in words if word in positive_words)
    negative_score = sum(1 for word in words if word in negative_words)

    polarity_score = (positive_score - negative_score) / (positive_score + negative_score + 0.000001)
    subjectivity_score = (positive_score + negative_score) / (len(words) + 0.000001)

    return {
        "Positive Score": positive_score,
        "Negative Score": negative_score,
        "Polarity Score": polarity_score,
        "Subjectivity Score": subjectivity_score
    }

def count_syllables(word, custom_stopwords):
    word = word.lower()
    if word in custom_stopwords:
        return 0

    if len(word) <= 3:
        return 1

    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1

    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1

    if word.endswith("e"):
        count -= 1

    if word.endswith("le") and word[-3] not in vowels:
        count += 1

    if count == 0:
        count += 1

    return count


def perform_readability_analysis(text, custom_stopwords):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    words = [word for word in words if word not in custom_stopwords]

    average_sentence_length = len(words) / len(sentences)

    complex_word_count = count_complex_words(text, custom_stopwords)
    percentage_complex_words = complex_word_count / len(words) * 100

    fog_index = 0.4 * (average_sentence_length + percentage_complex_words)
    avg_words_per_sentence = len(words) / len(sentences)
    personal_pronouns = len(re.findall(r'\b(I|we|my|ours|us)\b', text, re.IGNORECASE))
    avg_word_length = sum(len(word) for word in words) / len(words)

    return {
        "Average Sentence Length": average_sentence_length,
        "Percentage of Complex Words": percentage_complex_words,
        "Fog Index": fog_index,
        "Average Number of Words Per Sentence": avg_words_per_sentence,
        "Personal Pronouns": personal_pronouns,
        "Average Word Length": avg_word_length
    }

input_data = pd.read_excel("Output Data Structure.xlsx")
i = 1
# Iterate through the rows in the input data
for index, row in input_data.iterrows():
    url = row["URL"]
    print(i, url)
    i+=1
    text = fetch_text_from_url(url)

    if text:
        sentiment_result = perform_sentiment_analysis(text, custom_stopwords)
        readability_result = perform_readability_analysis(text, custom_stopwords)
        input_data.at[index, "POSITIVE SCORE"] = sentiment_result["Positive Score"]
        input_data.at[index, "NEGATIVE SCORE"] = sentiment_result["Negative Score"]
        input_data.at[index, "POLARITY SCORE"] = sentiment_result["Polarity Score"]
        input_data.at[index, "SUBJECTIVITY SCORE"] = sentiment_result["Subjectivity Score"]
        input_data.at[index, "AVG SENTENCE LENGTH"] = readability_result["Average Sentence Length"]
        input_data.at[index, "PERCENTAGE OF COMPLEX WORDS"] = readability_result["Percentage of Complex Words"]
        input_data.at[index, "FOG INDEX"] = readability_result["Fog Index"]
        input_data.at[index, "AVG NUMBER OF WORDS PER SENTENCE"] = readability_result["Average Number of Words Per Sentence"]
        input_data.at[index, "COMPLEX WORD COUNT"] = count_complex_words(text, custom_stopwords)
        input_data.at[index, "WORD COUNT"] = len(word_tokenize(text))
        input_data.at[index, "COMPLEX WORD COUNT"] = count_complex_words(text, custom_stopwords)
        input_data.at[index, "WORD COUNT"] = len(word_tokenize(text))


input_data.to_excel("Output.xlsx", index=False)
