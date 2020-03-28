import pandas as pd
from nltk.corpus import stopwords
import re
import string
from sklearn.feature_extraction.text import CountVectorizer

ds = pd.read_csv('Training\italian_dataset.csv', usecols=[
    'tweet_text', 'sentiment'], engine='python')
neutral = pd.read_csv('Training\\tweets_neutral.csv', usecols=[
    'tweet_text', 'sentiment'], engine='python')
ds.tweet_text = ds.tweet_text.replace("https?://[\w/%-.]*", " ", regex=True)

# Removing the space at the beginning
ds.tweet_text = ds.tweet_text.replace('^ ', '', regex=True)
# Removing the space at the end
ds.tweet_text = ds.tweet_text.replace(' $', '', regex=True)
ds.tweet_text = ds.tweet_text.replace('^', ' ', regex=True)
ds.tweet_text = ds.tweet_text.replace('$', ' ', regex=True)
ds.tweet_text = ds.tweet_text.replace(
    '/(?:https?|ftp):\/\/[\n\S]+/g', '', regex=True)
ds.tweet_text = ds.tweet_text.replace('#', '', regex=True)


# To lowercase
ds.tweet_text = ds.tweet_text.apply(
    lambda x: x.lower())


# Removing the space at the beginning and at the end
ds.tweet_text = ds.tweet_text.apply(lambda x: x.strip())

# Removing empty tweets
ds = ds[ds.tweet_text != '']


# Removing usernames
def remove_mentions(text):
    return re.sub(r"(?:\@|https?\://)\S+", "", text)


# Remove punctuation
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ')
    return text


# Changing string to numeri values
ds['sentiment'] = ds['sentiment'].replace(
    ['NEGATIVE', 'NEUTRAL', 'POSITIVE'], [-1, 0, 1])

ds['tweet_text'] = ds['tweet_text'].apply(remove_mentions)

# Removing punctuation
ds['tweet_text'] = ds['tweet_text'].apply(remove_punctuations)


# Removing the spaces in excess
ds['tweet_text'] = ds['tweet_text'].replace(
    '\s+', ' ', regex=True)

# print(ds.head(10))
ds = ds[ds.sentiment != 'MIXED']

df_sorted = ds.sort_values(by=['sentiment'], ascending=False)

print(df_sorted.head(100))
print(neutral.shape)
# Saving to CSV
# df_sorted.to_csv('cleanDatasetSorted.csv')
