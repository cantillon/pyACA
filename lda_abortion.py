#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 13:38:52 2020

@author: cantillon
"""

# Install nltk, gensim and pyldavis
# pip install nltk
# pip install gensim
# pip install pyldavis

# Download data for nltk modules

import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('maxent_treebank_pos_tagger')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Import modules

import csv
import re
from glob import glob
from string import punctuation
import random
random.seed("ic2s2colgne")
from nltk.sentiment import vader
from nltk.corpus import stopwords
from nltk import FreqDist
from gensim import corpora
from gensim import models
import pyLDAvis
import pyLDAvis.gensim
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from datetime import datetime

# Retrieve a list of all articles from a file

filelist = glob('/home/cantillon/Dropbox/ACA/code/the_star_news_output.csv')
print(filelist)
articles = []
for fn in filelist:
    with open (fn) as fi:
        reader=csv.reader(fi)
        for row in reader:
            if row[2]=='True':
                articles.append(row[8])
len(articles)
articles[0][:500]

# Remove HTML tags, remove dashes, remove punctuation, convert to lower case and remove double spaces

articles=[article.replace('<p>',' ').replace('</p>',' ') for article in articles]
articles=[article.replace('-',' ').replace('—',' ') for article in articles]
articles=["".join([l for l in article if l not in punctuation]) for article in articles]
articles=[article.lower() for article in articles]
articles=[" ".join(article.split()) for article in articles]
articles[0][:500]

# Lemmatizing; works better than stemming in this case
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
articles_lemmatized = [" ".join([lemmatizer.lemmatize(word,pos='v') for word in article.split()]) for article in articles]
articles_lemmatized = [" ".join([lemmatizer.lemmatize(word,pos='n') for word in article.split()]) for article in articles_lemmatized]
articles_lemmatized = [" ".join([lemmatizer.lemmatize(word,pos='a') for word in article.split()]) for article in articles_lemmatized]
articles_lemmatized[0][:500]

# Remove stopwords; it works better than TF-IDF scores or filtering extremes in this case

mystopwords = set(stopwords.words('english'))
articles_lemmatized_clean = [" ".join([w for w in article.split() if w not in mystopwords]) for article in articles_lemmatized]
articles_lemmatized_clean[0][:500]

# Identify the most common words and create a custom stopword list:

# Split each article into a list of words
word_lists = []
for article in articles_lemmatized_clean:
    words = [article.split()]
    for word in words:
        word_lists.append(word)
len(word_lists)

# Create a list of all the words from all articles
all_words = []
for l in word_lists:
  for w in l:
    all_words.append(w)
len(all_words)

# Calcuate the frequency distribution of all the words in all articles and identify the 100 most common words
fd = nltk.FreqDist(all_words)
fd.most_common(100)

# Create a dataframe with the 100 most common words and save it to a csv file
df = pd.DataFrame(fd.most_common(100))
df.to_csv('/home/cantillon/Dropbox/ACA/code/FreqDist.csv', encoding='utf-8', index=False)

# Based on the 100 most common words, a costum list of stopwords was identified and saved to a csv file

mystopwords2 = set(open('/home/cantillon/Dropbox/ACA/code/the_star_news_stopwords.txt').read().splitlines())
articles_preprocessed = [" ".join([w for w in article.split() if w not in mystopwords2]) for article in articles_lemmatized_clean]
articles_preprocessed[0][:500]

# LDA model: Convert all strings to lists of words, assing a token id to each word, represent each article by (token_id, token_count) tuples

ldainput_m0 = [article.split() for article in articles_preprocessed]
id2word_m0 = corpora.Dictionary(ldainput_m0)
id2word_m0.filter_extremes(no_below=5)
ldacorpus_m0 = [id2word_m0.doc2bow(doc) for doc in ldainput_m0]
lda_m0 = models.LdaModel(ldacorpus_m0, id2word=id2word_m0, num_topics=10)
lda_m0.print_topics()

# Visualizing and exploring the topic model

vis_data = pyLDAvis.gensim.prepare(lda_m0,ldacorpus_m0,id2word_m0)
pyLDAvis.display(vis_data)
