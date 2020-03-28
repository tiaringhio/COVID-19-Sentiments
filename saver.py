import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import GetOldTweets3 as got
import itertools
import collections
import tweepy as tw
import nltk
from nltk.corpus import stopwords
import re
import networkx
from textblob import TextBlob
import json
import datetime
import time
import yaml

TWITTER_CONFIG_FILE = 'apis.yaml'

# Loading API from yaml file
with open(TWITTER_CONFIG_FILE, 'r') as config_file:
    config = yaml.load(config_file)

# Assignin values from yaml
consumer_key = config['twitter']['consumer_key']
consumer_secret = config['twitter']['consumer_secret']
access_token_key = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token_key, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)


# Function to remove urls from tweets
def remove_url(txt):

    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())


# Number of tweets to retrieve
count = 5000

# List of tweets retrieved
tweet_list = []


# Dictionary to which the tweets are added
def tweet_to_dict(tweet):
    tweet_dict = {
        'id': tweet.id,
        'date': str(tweet.date),
        'hashtags': tweet.hashtags,
        'text': remove_url(tweet.text),
    }
    return tweet_dict


# Adding tweets to dictionary and then to list
def add_to_list():
    for tweet in tweets:
        tweet_dict_temp = tweet_to_dict(tweet)
        tweet_list.append(tweet_dict_temp)
    print('--- Added to list!--- ')


# Date to start from
date_upper = datetime.datetime(2020, 3, 1)
date_lower = datetime.datetime(2020, 2, 29)

date_until = date_upper
date_start = date_lower

start_string = date_start.strftime("%Y-%m-%d")
until_string = date_until.strftime("%Y-%m-%d")


for i in range(4):
    # Create a custom search term and define the number of tweets
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(
        'Coronavirus').setSince(start_string).setUntil(until_string).setLang('it').setMaxTweets(count)
    # Call getTweets and saving in tweets
    print('--- Starting query... ---')
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    print('--- Adding to list... ---')
    add_to_list()
    print('--- Writing JSON... ---')
    # Saving list to JSON file
    json.dump(tweet_list, open('./JSON/saver_output.json', 'w'))
    print('--- Going to sleep... ---\n\n')
    time.sleep(60*5)
    # Add 1 to date after each passage
    date_start += datetime.timedelta(days=1)
    date_until += datetime.timedelta(days=1)
    # Convert dates to string
    start_string = date_start.strftime("%Y-%m-%d")
    until_string = date_until.strftime("%Y-%m-%d")
