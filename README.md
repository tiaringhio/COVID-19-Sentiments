<!-- PROJECT LOGO -->
  <br />
    <p align="center">
    <img src="https://image.flaticon.com/icons/svg/2904/2904311.svg" alt="Logo" width="130" height="130">
  </a>
  <h1 align="center">COVID-19 Sentiment Analysis </h1>
  <p align="center">
    This project aims to understand the general sentiments about COVID-19 in Italy.
  </p>

## Table of contents

- [About the project](#about-the-project)
- [Used in this project](#used-in-this-project)
  - [NLTK](#nltk)
  - [Pandas](#pandas)
  - [Re](#re)
  - [Preprocessor](#preprocessor)
  - [Emoji](#emoji)
  - [Scikit-learn](#scikit-learn)
  - [Pickle](#pickle)
- [Datasets](#datasets)
- [The process](#the-process)
  - [Data gathering](#data-gathering)
  - [Cleaning](#cleaning)
  - [Machine Learning](#machine-learning)
  - [Analysis](#analysis)

# About The Project

Sentiment analysis is a type of data mining that measures the inclination of people’s opinions through natural language processing (NLP). This project aims to better understand the general sentiments about COVID-19 by using Twitter to predict, using machine learning, whether a tweet is positive, negative or mixed and in what amount. To achieve this goal a mix of statistics and linguistics have been used.

# Used in this project

## NLTK

[Natural Language Toolkit](https://www.nltk.org/) is extremely helpful for analysing human language data.
It is the core of the project, thanks to its many libraries.

## Pandas

[Pandas](https://pandas.pydata.org/) A data manipulation tool, useful for visualizing and manipulating data.

## Re

A python package, used to remove noise from text such as links and punctuation.

## Preprocessor

[Library](https://pypi.org/project/tweet-preprocessor/) This library makes it easy to clean, parse or tokenize tweets. I has to be installed via cloning and setup.py, pip will give errors.

## Scikit-learn

[Scikit-learn](https://scikit-learn.org/) One of the leading libraries for machine learning, has been used through SklearnClassifier in NLTK in order to use the power of NLTK mixed with the power of sklearn

## Pickle

The pickle module implements binary protocols for serializing and de-serializing a Python object structure. “Pickling” is the process whereby a Python object hierarchy is converted into a byte stream “unpickling” is the inverse operation.

# Datasets

By cloning this repository it is possibile to find the datasets in the correct folders.

- The gathered tweets are found in the _JSON_ folder, separated by month
- The analyzed tweets are found in the _CSV_ folder, separated by month
- _italian_dataset_ has been used to train the models and can be found in the _Training Dataset_ folder.
  - There are subsequent datasets obtained from the original one including:
    - tweets_negative
    - tweets_negative

# The process

## Data gathering

The tweets are gathered through a package called [GetOldTweets3](https://github.com/Mottl/GetOldTweets3), which allows to retrieve data without using Twitter's own API, why you may ask? Because Twitter limits the amount of data that can be downloaded, in quantity and in time, you cannot retrieve information older thank a week. This is done trough a simple script.

## Cleaning

The most extensive part. The text is stripped from puntctuation, links, tags, excessive spaces ecc. using Re, it is then tokenized and stemmed via NLTK to improve the results obtained in the learning phase, furthermore stop words are removed from text using NLTK's built in library and other stopwords found online, specific for the italian language. This process is done throughout the entirety of the project to better mitigate mistakes.

## Machine Learning

For the training part a [dataset of italian tweets](https://github.com/charlesmalafosse/open-dataset-for-sentiment-analysis) has been used, which has been cleaned with the same process as the previously mentioned data. The choice to use this dataset was made because it is extensive and to avoid bias by tagging manually tweets as positive or negative, language is very subjective. After a bit of data preparation, the following models have been trained and pickled:

- Naive Bayes
- Logistic Regression
- Bernoulli Naive Bayes
- Multinomial Naive Bayes
- Stochastic Gradient Descent
- Support Vector Classification
- NuSVC
- LinearSVC

The results were generally better using the Logistic Regression algorithm. An ensemble model has also been made which combines the results obtained with the previous ones, improving them by 0.5% as show in the following table.

<center>

|            _Algorithm_            | _Accuracy_ |
| :-------------------------------: | :--------: |
|          **Naive Bayes**          |    84%     |
|      **Logistic Regression**      |    88%     |
|     **Bernoulli Naive Bayes**     |    85%     |
|    **Multinomial Naive Bayes**    |   86.5%    |
|  **Stochastic Gradient Descent**  |   87.25%   |
| **Support Vector Classification** |   88.1%    |
|             **NuSVC**             |   87.95%   |
|           **LinearSVC**           |    86%     |
|           **Ensemble**            |   88.6%    |

</center>

## Analysis

The final step: showing the result using plots via Matplotlib.
In order to understand the data the following graphs were chosen:

- Histogram for Polarity
- Wordcloud
- Line graph for word frequency through time

### Polarity

<br />
    <p align="center">
    <img src="Results\march_sentiments.png"width="500">

### Wordcloud

<br />
    <p align="center">
    <img src="Results\january_cloud.png" width="500">

### Line graph

<br />
    <p align="center">
    <img src="Results\Keywords\Single\ansia.png" width="500">
