import pickle
import pandas as pd
from datetime import datetime

# January tweets
january = pd.read_csv('CSV\january_analyzed.csv', usecols=['date',
                                                           'tokenized', 'polarity'], engine='python')

# February dataset
february = pd.read_csv('CSV\\february_analyzed.csv', usecols=[
    'tokenized', 'polarity'], engine='python')

# Import Naive Bayes classifier
classifier = pickle.load(open('NaiveBayes.pkl', 'rb'))
# classifier.show_most_informative_features(10)

# How frequent is a word?
tweets = january.tokenized.values.tolist()
words = []

for sentence in tweets:
    word = sentence.split()
    words.append(word)


new_list = [j for i in words for j in i]

print(new_list.count('ansia'))

dates = january.date.values.tolist()


def convert_date(dates):
    date_values = []
    for date in dates:
        datetime_object = datetime.strptime(date, "%Y-%m-%d %H:%M:%S%z")
        date_values.append(datetime_object)
    return date_values
