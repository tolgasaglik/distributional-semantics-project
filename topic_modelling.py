from __future__ import unicode_literals
import re, glob, codecs, os, nltk
import numpy as np
import pandas as pd
from pprint import pprint
from collections import Iterable

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess,lemmatize
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy
import en_core_web_sm

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# Add stop words with roman numbers
def int_to_Roman(num):
   val = (1000, 900,  500, 400, 100,  90, 50,  40, 10,  9,   5,  4,   1)
   syb = ('M',  'CM', 'D', 'CD','C', 'XC','L','XL','X','IX','V','IV','I')
   roman_num = ""
   for i in range(len(val)):
      count = int(num / val[i])
      roman_num += syb[i] * count
      num -= val[i] * count
   return roman_num


for i in range(1,100):
    r = int_to_Roman(i)
    stop_words.extend([r.lower(),r])

from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer  #porterstemmer v2
from nltk.tokenize import sent_tokenize, word_tokenize

import xml.etree.ElementTree as ET
articles_filenames = sorted(glob.glob("./cleanfiles/temp/*.txt"))
articles_filenames = sorted(glob.glob("./cleanfiles/temp/ada*.txt"))
# articles_filenames = sorted(glob.glob("./cleanfiles/temp/adaessa-plain.txt"))

d = list()
for article in articles_filenames:
    with open(article,encoding='utf-8') as f:
        book = os.path.basename(article.split('.')[0])
        d.append(pd.DataFrame({'text': book, 'lines': f.readlines()}))

doc = pd.concat(d)
doc.tail(1000)


# Convert to list
data = doc.lines.values.tolist()

# Remove Emails
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]

# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]

pprint(data[:10])


# def sentence_to_wordlist(raw):
#     #stemmer = PorterStemmer()
#     stemmer = SnowballStemmer('english')
#     words = raw.split()
#     stem_words = ' '.join([stemmer.stem(word) for word in words])
#     return stem_words
#
#
# sentences = []
# for raw_sentence in raw_sentences:
#     if len(raw_sentence) > 0:
#         sentences.append(sentence_to_wordlist(raw_sentence))
#
# stemmer = SnowballStemmer('english')
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


data_words = list(sent_to_words(data))

print(data_words[:10])

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
# print(trigram_mod[bigram_mod[data_words[10]]])


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en

# nlp = spacy.load('en_core_web_sm')
# nlp = English()
nlp = spacy.load('en', disable=['parser', 'ner'])
nlp.max_length = 11000000


def lemmatization(texts, allowed_postags):
#def lemmatization(texts, allowed_postags=['NOUN']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def gensimlemm(texts):
    texts_out = []
    for sent in texts:
        doc = " ".join(sent)
        # print(doc)
        if len(doc) > 0:
            lemmatized_out = [wd.decode('utf-8').split('/')[0] for wd in lemmatize(doc) if wd.decode('utf-8').split('/')[1]=='NN']
            texts_out.append(lemmatized_out)
    return texts_out


# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['VERB', 'ADV', 'NOUN'])
#data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN'])

# Do lemmatization keeping only noun without empty ones (runtime error-does not work)
# data_lemmatized = gensimlemm(data_words_bigrams)

print(data_lemmatized[:10])


# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])


# id2word[100]

# Human readable format of corpus (term-frequency)
# [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:100]]

num_topics = 20

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


# View the topics in LDA model
# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


#Compute Model Perplexity and Coherence Score
# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


#Visualize the topics-keywords
# Visualize the topics
# pyLDAvis.enable_notebook()


vis_lda = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
pyLDAvis.save_html(vis_lda, 'LDA_Vis_'+str(num_topics)+'_Topics.html')
#vis_lda


# Building LDA Mallet Model
mallet_path = os.getenv('MALLET_HOME')+'/bin/mallet' # update this path
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)


# Show Topics
pprint(ldamallet.show_topics(formatted=False))
#
# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)


# Convert mallet to gensim type for pyLDAvis
mallet_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(ldamallet)
# Visualize the topics
vis_mallet = pyLDAvis.gensim.prepare(mallet_model, corpus, id2word)
# For saving html
pyLDAvis.save_html(vis_mallet, 'LDA_Mallet_Vis_'+str(num_topics)+'_Topics.html')
#vis_mallet


''' USE CREATED MODEL ON A SINGLE DOCUMENT TO SEE RESULTS'''

articles_filenames2 = sorted(glob.glob("./cleanfiles/temp/adaessa-plain.txt"))
d2 = list()
for article in articles_filenames2:
    with open(article,encoding='utf-8') as f:
        book = os.path.basename(article.split('.')[0])
        d2.append(pd.DataFrame({'text': book, 'lines': f.readlines()}))

doc2 = pd.concat(d2)
doc2.tail(1000)


# Convert to list
data2 = doc2.lines.values.tolist()

# Remove Emails
data2 = [re.sub('\S*@\S*\s?', '', sent) for sent in data2]

# Remove new line characters
data2 = [re.sub('\s+', ' ', sent) for sent in data2]

# Remove distracting single quotes
data2 = [re.sub("\'", "", sent) for sent in data2]

pprint(data2[:10])

data_words2 = list(sent_to_words(data2))

print(data_words2[:10])

# Build the bigram and trigram models
bigram2 = gensim.models.Phrases(data_words2, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram2 = gensim.models.Phrases(bigram[data_words2], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod2 = gensim.models.phrases.Phraser(bigram2)
trigram_mod2 = gensim.models.phrases.Phraser(trigram2)

data_words_nostops2 = remove_stopwords(data_words2)

data_words_bigrams2 = make_bigrams(data_words_nostops2)
#data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'VERB', 'ADV'])
data_lemmatized2 = lemmatization(data_words_bigrams2, allowed_postags=['NOUN', 'VERB', 'ADV'])
id2word2 = corpora.Dictionary(data_lemmatized2)
texts2 = data_lemmatized2

# Term Document Frequency
corpus_single = [id2word2.doc2bow(text) for text in texts2]
#doc_lda = lda_model[corpus_single]

vis_lda_single = pyLDAvis.gensim.prepare(lda_model, corpus_single, id2word)
pyLDAvis.save_html(vis_lda_single, 'Vis_Single_Doc_'+str(num_topics)+'_Topics.html')

#ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus_single, num_topics=num_topics, id2word=id2word2)
#mallet_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(ldamallet)
vis_mallet_single = pyLDAvis.gensim.prepare(mallet_model, corpus_single, id2word)
pyLDAvis.save_html(vis_mallet_single, 'Vis_Mallet_Single_Doc_'+str(num_topics)+'_Topics.html')



''' ############### END OF ADDED CODE ############### '''




'''

#How to find the optimal number of topics for LDA?
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics
    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics
    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values


# Can take a long time to run.
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=40, step=6)


# Show graph
limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.savefig('./plots/coherence.png',  dpi=500)
plt.show()


# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


# Select the model and print the topics
optimal_model = model_list[3]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))


# Finding the dominant topic in each sentence
def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()
    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.tail(20)


#Find the most representative document for each topic
# Group top 5 sentences under each topic
sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], axis=0)

# Reset Index
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

# Show
sent_topics_sorteddf_mallet.head(10)


#Topic distribution across documents
# Number of Documents for Each Topic
topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

# Percentage of Documents for Each Topic
topic_contribution = round(topic_counts/topic_counts.sum(), 4)

# Topic Number and Keywords
topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

# Concatenate Column wise
df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# Change Column names
df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

# Show
df_dominant_topics

# '''