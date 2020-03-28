from numpy import array
from sklearn.datasets import make_blobs
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
import random
import re
import string
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from nltk.stem.snowball import SnowballStemmer
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import make_scorer, accuracy_score

# Positive tweets from dataset
positive = pd.read_csv('Training\\tweets_positive.csv', usecols=[
    'tweet_text', 'sentiment'], engine='python')

# Negative tweets from dataset
negative = pd.read_csv('Training\\tweets_negative.csv', usecols=[
    'tweet_text', 'sentiment'], engine='python')

# tweets from coronavirus for testing purposes
text = pd.read_csv('Training\\smaller_test.csv', usecols=[
    'data'], engine='python')

january_tweets = pd.read_csv('JSON\January\dataCleaned.csv', usecols=[
    'data', 'text', 'date'], engine='python')

february_tweets = pd.read_csv('JSON\February\dataCleaned.csv', usecols=[
    'data', 'text', 'date'], engine='python')

# Tokenizing positive, negative and text
positive_tokens = positive['tweet_text'].apply(word_tokenize)
negative_tokens = negative['tweet_text'].apply(word_tokenize)

# Italian stopwords
stop_words = stopwords.words('italian')

# Italian Stemmer
stemmer = SnowballStemmer('italian')


# Additional stopwords found online
def additional_stop_words():
    with open('Training\\stopwords.txt', 'r') as f:
        additional_stopwords = f.readlines()
    additional_stopwords = [x.strip() for x in additional_stopwords]
    return additional_stopwords


# Function to remove noise from tokens, removing also stopwords
def remove_noise(tweet_tokens, stop_words=(), additional_stop_words=()):
    cleaned_tokens = []
    for token in tweet_tokens:
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)
        token = stemmer.stem(token)
        if len(token) > 1 and token not in string.punctuation and token.lower() not in stop_words and token.lower() not in additional_stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


# Lists of positive and negative cleaned tokens
positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

# Cleaning positive tokens and adding to list
for tokens in positive_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

# Cleaning negative tokens and adding to list
for tokens in negative_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))


# Defining a generator function that takes a list of tweets as an argument and
# provides a list of words in all of the tweet tokens joined.
def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token


# Positive words
every_positive_word = get_all_words(positive_cleaned_tokens_list)

# Negative words
every_negative_word = get_all_words(negative_cleaned_tokens_list)

# What are the most positive words and how frequent are they?
# freq_dist_positive = FreqDist(every_positive_word)
# print(freq_dist_positive.most_common(10))

# freq_dist_negative = FreqDist(every_negative_word)
# print(freq_dist_negative.most_common(10))


# Converts a list of cleaned tokens to dictionaries
# token as the key and True as values
def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


# Preparing data for training
positive_tokens_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_model = get_tweets_for_model(negative_cleaned_tokens_list)

# Attach label Positive or Negative to each tweet
positive_dataset = [(tweet_dict, 'Positive')
                    for tweet_dict in positive_tokens_model]
negative_dataset = [(tweet_dict, 'Negative')
                    for tweet_dict in negative_tokens_model]

# Create dataset by joining positive and negative
dataset = positive_dataset + negative_dataset

# Shuffle the dataset to avoid bias
random.shuffle(dataset)

# Separating training to test data
train_data = dataset[:9000]
test_data = dataset[9000:]

# LogisticRegression classifier
LRClassifier = SklearnClassifier(LogisticRegression())
LRClassifier.train(train_data)
print("Classifier accuracy percent:",
      (classify.accuracy(LRClassifier, test_data))*100)

# Save model as file for later usage
filename = 'Models\Classifier.pkl'
with open(filename, 'wb') as file:
    pickle.dump(LRClassifier, file)

NaiveBayes = NaiveBayesClassifier.train(train_data)
print("Classifier accuracy percent:",
      (classify.accuracy(NaiveBayes, test_data))*100)

# Save model as file for later usage
filename = 'Models\NaiveBayes.pkl'
with open(filename, 'wb') as file:
    pickle.dump(NaiveBayes, file)

# Adding tweets and tokenized from january to list
january = january_tweets.data.values.tolist()
j_tokenized = january_tweets.text.values.tolist()
j_date = january_tweets.date.values.tolist()
# List of classified tweets from january
classified_january = []


# For each tweet, remove noise and tokenize, calculate the accuracy of a given prediction
# and add to classified list
for tweet in january:
    custom_tokens = remove_noise(word_tokenize(tweet))
    classified_january.append(tuple((tweet, LRClassifier.prob_classify(
        dict([token, True] for token in custom_tokens)).prob('Positive'), LRClassifier.prob_classify(
        dict([token, True] for token in custom_tokens)).prob('Negative'))))

# creating dataframe from classified tweets
df_january = pd.DataFrame(classified_january, columns=[
    'tweet', 'positive', 'negative'])

# Obtain polarity by subtracting positives with negatives values
df_january['polarity'] = df_january['positive'] - df_january['negative']

# Adding tokenized column
df_january['tokenized'] = j_tokenized

# Adding date column
df_january['date'] = j_date

# Reordering columns
df_january = df_january[['date', 'tweet', 'tokenized',
                         'positive', 'negative', 'polarity']]

print(df_january.head(10))

# Saving dataframe to csv
df_january.to_csv('CSV\january_analyzed.csv')


# Adding tweets and tokenized from february to list
february = february_tweets.data.values.tolist()
f_tokenized = february_tweets.text.values.tolist()
# List of classified tweets from february
classified_february = []


# For each tweet, remove noise and tokenize, calculate the accuracy of a given prediction
# and add to classified list
for tweet in february:
    custom_tokens = remove_noise(word_tokenize(tweet))
    classified_february.append(tuple((tweet, LRClassifier.prob_classify(
        dict([token, True] for token in custom_tokens)).prob('Positive'), LRClassifier.prob_classify(
        dict([token, True] for token in custom_tokens)).prob('Negative'))))


# creating dataframe from classified tweets
df_february = pd.DataFrame(classified_february, columns=[
    'tweet', 'positive', 'negative'])

# Obtain polarity by subtracting positives with negatives values
df_february['polarity'] = df_february['positive'] - df_february['negative']

# Adding tokenized column
df_february['tokenized'] = f_tokenized

# Reordering columns
df_february = df_february[['tweet', 'tokenized',
                           'positive', 'negative', 'polarity']]

print(df_february.head(10))

# Saving dataframe to csv
df_february.to_csv('CSV\\february_analyzed.csv')
