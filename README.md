<!-- PROJECT LOGO -->
  <br />
    <p align="center">
    <img src="https://cdn-icons.flaticon.com/png/512/3214/premium/3214022.png?token=exp=1638525489~hmac=5ef5a143d988d878181e226ef7b2b460" alt="Logo" width="130" height="130">
  </a>
  <h1 align="center">COVID-19 Sentiment Analysis</h1>
  <p align="center">
    This project aims to understand the general sentiments about COVID-19 in Italy.
  </p>

## Table of contents

- [Table of contents](#table-of-contents)
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
    - [Tokenization](#tokenization)
    - [Stemming](#stemming)
    - [Stop words removal](#Stop-words-removal)
    - [Noise removal](#noise-removal)
  - [Machine Learning](#machine-learning)
  - [Analysis](#analysis)
- [Live classifier](#live-classifier)
  - [Polarity](#polarity)
  - [WordCloud](#wordcloud)
- [What's next?](#what's-next)
- [License](#license)
- [Contributors](#contributors)

# About The Project

Sentiment analysis is a type of data mining that measures the inclination of people’s opinions through natural language processing (NLP). This project aims to better understand the general sentiments about COVID-19 by using Twitter to predict, using machine learning, whether a tweet is positive, negative or mixed and in what amount. To achieve this goal a mix of statistics and linguistics have been used.

# Used in this project

## NLTK

[Natural Language Toolkit](https://www.nltk.org/) is extremely helpful for analysing human language data.
It is the core of the project, thanks to its many libraries. More about this powerful tool in the [Cleaning](#cleaning) section

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
- _italian_dataset_ has been used to train the models and can be found in the _Training Dataset_ folder (more about this in the [Machine Learning](#machine-learning) section).
  - There are subsequent datasets obtained from the original one including:
    - negative
    - negative

# The process

## Data gathering

The tweets are gathered through a package called [GetOldTweets3](https://github.com/Mottl/GetOldTweets3), which allows to retrieve data without using Twitter's own API, why you may ask? Because Twitter limits the amount of data that can be downloaded, in quantity and in time, you cannot retrieve information older thank a week. This is done trough a simple script.

## Cleaning

The most extensive part. The text is stripped from puntctuation, links, tags, excessive spaces ecc. using Re, it is then tokenized and stemmed via NLTK to improve the results obtained in the learning phase, furthermore stop words are removed from text using NLTK's built in library and other stopwords found online, specific for the italian language. This process is done throughout the entirety of the project to better mitigate mistakes. This phase includes the following steps:

- Tokenization
- Stemming
- Stop words removal
- Noise removal

### Tokenization

The operation by which the text is divided into words, sentences, symbols or other significant elements called tokens. In computational linguistics the token can be defined simply as any sequence of characters delimited by spaces. This process is essential for preparing the text for the learning phase.

### Stemming

The process of reducing the inflected form of a word to its root form, called "stem" which does not necessarily correspond to the morphological root of the word: normally it is sufficient that the related words are mapped to the same stem. Stemming is a part, together with lemmatization, of the text normalization process.

### Stop words removal

Stop words are words that, given their high frequency in a language, are usually considered to be insignificant in a research, for this reason they are removed during the preparation for learning. There is no official list of Italian stop words but terms such as articles, conjunctions, generic words or widespread verbs are included.

### Noise removal

What does NLP noise mean? Noise is a part of the text that does not add meaning or information to the data. Punctuation, links, special characters that usually provide context to the text are included in this category, context is often difficult to process.

## Machine Learning

For the training part a [dataset of italian tweets](https://github.com/charlesmalafosse/open-dataset-for-sentiment-analysis) has been used, which has been cleaned with the same process as the previously mentioned data. The choice to use this dataset was made because it is extensive and to avoid bias by tagging manually tweets as positive or negative since language is very subjective. After a bit of data preparation, the following models have been trained and _pickled_:

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

The final phase, in which the data downloaded through the models obtained is analyzed. Logistic Regression was chosen as it proved to be the most accurate and consistent. The data was divided by month, each of them was subjected to analysis through the model, the result was then saved as a CSV file.
The results were shown using plots via Matplotlib.
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
    <img src="Results\Keywords\Single\paura.png" width="500">

# Live classifier

You can try the classifier used in the project with a Telegram bot, it's available [here](t.me/covid_sentiment_bot). [Here](https://github.com/tiaringhio/Sentiment-Analyzer) you can find the code needed to run your own version of the bot along with details about the usage.

# What's next?

- Data for the months yet to come
- The models can be further imporoved using deep learning techniques using [Keras](https://keras.io/)
- Improving the cleaning process
- Addition of english language
- ~~Live text classifier~~ Can be found [here](#live-classifier)

# License

Distributed under the GPL License. See `LICENSE` for more information.

Icons made by <a href="https://www.flaticon.com/authors/smashicons" title="Smashicons">Smashicons</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a>

# Contributors

[Mattia Ricci](https://github.com/tiaringhio) - 285237
