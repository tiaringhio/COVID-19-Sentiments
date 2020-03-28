import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from wordcloud import WordCloud
from nltk.corpus import stopwords


# January tweets
january = pd.read_csv('CSV\january_analyzed.csv', usecols=[
    'tokenized', 'polarity'], engine='python')

fig, ax = plt.subplots(figsize=(10, 8))

cm = plt.cm.get_cmap('RdYlGn')
# Histogram for january
january.hist(bins=100, ax=ax, color='red', label='Data')

plt.legend(loc="upper left")
plt.ylabel('N of tweets')
plt.xlabel('Polarity')
plt.title('Sentiments of January 2020 about coronavirus')

plt.savefig('Results\january_sentiments.png')
plt.show()
# plt.savefig('Results\january_sentiments.png')

# February dataset
february = pd.read_csv('CSV\\february_analyzed.csv', usecols=[
    'tokenized', 'polarity'], engine='python')

fig, ax = plt.subplots(figsize=(10, 8))

# Histogram for february
february.hist(bins=100, ax=ax, color='blue', label='Data')

plt.legend(loc="upper left")
plt.ylabel('N of tweets')
plt.xlabel('Polarity')
plt.title('Sentiments of February 2020 about coronavirus')

plt.savefig('Results\\february_sentiments.png')
plt.show()

# Italian stopwords from NLTK
stop_words = stopwords.words('italian')


# Additional stopwords found online
def additional_stop_words():
    with open('Training\\stopwords.txt', 'r') as f:
        additional_stopwords = f.readlines()
    additional_stopwords = [x.strip() for x in additional_stopwords]
    return additional_stopwords


# Stop words for wordclouds
stopwords = set(stopwords.words('italian'))
stopwords.update(['coronavirus', 'corona', 'virus',
                  'coronavirusitalia', 'italia', 'cina', 'co', 'ci', 'gi', 'fa', 'cos'])
stopwords.update(additional_stop_words())


# Join every tweet for january
text_january = " ".join(tweet for tweet in january.tokenized)


# Create Wordcloud
wordcloud = WordCloud(width=3000, height=2000, collocations=False,
                      stopwords=stopwords).generate(text_january)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

plt.savefig('Results\january_cloud.png')
plt.show()

# Join every tweet for february
text_february = " ".join(tweet for tweet in february.tokenized)

# Create Wordcloud
wordcloud = WordCloud(width=3000, height=2000, collocations=False,
                      stopwords=stopwords).generate(text_february)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

plt.savefig('Results\\february_cloud.png')
plt.show()
