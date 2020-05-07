from nltk.corpus import stopwords
import pandas as pd
import json

# Reading tweets from JSON file and ading them to Pandas Dataframe
tweets = pd.read_json('JSON\March\data.json')


# Deleting duplicate tweets
tweets = tweets.drop_duplicates(subset='text')
tweets['data'] = tweets['text']

# Defining stopwords and adding a space before and after to exclude the case
# in which the stopword is cointained in a word
words = set(stopwords.words('italian'))
stopwords = [' ' + x + ' ' for x in words]

tweets.text = tweets.text.replace(
    "@[\w]*[_-]*[\w]*", " ", regex=True)   # Removing the tags

# Removing the spaces in excess
tweets.text = tweets.text.replace('\s+', ' ', regex=True)
# Removing the space at the beginning
tweets.text = tweets.text.replace('^ ', '', regex=True)
# Removing the space at the end
tweets.text = tweets.text.replace(' $', '', regex=True)
# To lowercase
tweets.text = tweets.text.apply(
    lambda x: x.lower())
tweets.text = tweets.text.replace('^', ' ', regex=True)
tweets.text = tweets.text.replace('$', ' ', regex=True)

for word in stopwords:
    tweets.text = tweets.text.replace(word, ' ', regex=True)

# Removing the space at the beginning and at the end
tweets.text = tweets.text.apply(lambda x: x.strip())
# Removing empty tweets
tweets = tweets[tweets.text != '']

# tweets = tweets[['id', 'date', 'hashtags', 'text', 'data']]
# Saving to CSV
tweets.to_csv('./JSON/March/dataCleaned.csv')
