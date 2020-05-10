import os
import numpy as np
import pandas as pd
import GetOldTweets3 as got
import itertools
import collections
import re
import json
from datetime import datetime, timedelta
import time

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
dateSince = datetime(2020, 1, 1)
dateUntil = datetime(2020, 2, 1)

# Cycling through days
for day in range(30):
    # Converting dates to strings
    strDateSince = dateSince.strftime("%Y-%m-%d")
    strDateUntil = dateUntil.strftime("%Y-%m-%d")
    # Create a custom search term and define the number of tweets
    print("--- Grabbing " + strDateSince + " tweets... ---")
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(
        'Coronavirus').setSince(strDateSince).setUntil(strDateUntil).setLang('it').setMaxTweets(count)
    # Call getTweets and saving in tweets
    print('--- Starting query... ---')
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    print('--- Adding to list... ---')
    add_to_list()
    print('--- Writing JSON... ---')
    # Saving list to JSON file
    json.dump(tweet_list, open('./Datasets/JSON/April/saver_output1.json', 'w'))
    print('Total number of tweets: ', len(tweet_list))
    print('--- Going to sleep... ---')
    time.sleep(60*5)
    print('--- Changing day ... ---\n\n')
    dateSince += timedelta(days=1)
    dateUntil += timedelta(days=1)
