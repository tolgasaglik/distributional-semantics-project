#Everything should work by changing the file paths.. if you think, more processing is need.. let me know
import os, re
from gensim.parsing.preprocessing import remove_stopwords
import gensim 
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer 
from nltk.tokenize import sent_tokenize, word_tokenize
import glob
import codecs
import nltk
import pandas as pd
import xml.etree.ElementTree as ET

# Data processing
dir_path = "./AustLit/files/Text/"
out_path = "./cleanfiles/temp/"
if not os.path.exists(out_path):
    os.makedirs(out_path)
for file in os.listdir(dir_path):
    with open(dir_path + file, 'r', encoding='utf-8', errors='ignore') as infile, open(out_path + file, "w") as outfile:
        for line in infile:
            if not line.strip(): continue  # skip the empty line
            write_data = re.sub("\&\#[0-9]*\;", " ", line)
            write_data = re.sub(r"\*|\!|\,|\(|\)|\;|\:|\?|\'|\[|\]|\.", "", write_data)  # removes !,();:?.*[] characters
            write_data = remove_stopwords(write_data.lower()) + '\n'  # removes stopwords

            outfile.write(write_data)  # non-empty line. Write it to output:

#reading from processed files
#Tokenizing and Stemming process
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
file_path = out_path
articles_filenames = sorted(glob.glob(file_path+"*.txt"))

'''
#combining books into one string
corpus_raw = u""
for article_filename in articles_filenames:
    print("Reading '{0}'....".format(article_filename))
    with codecs.open(article_filename, "r", "utf-8") as artice_file:
        corpus_raw += artice_file.read()
    print("Corpus is now {0} characters long". format(len(corpus_raw)))
    print()

raw_sentences = tokenizer.tokenize(corpus_raw)

def sentence_to_wordlist(raw):
    #stemmer = PorterStemmer()
    stemmer = SnowballStemmer('english')
    words = raw.split()
    stem_words = ' '.join([stemmer.stem(word) for word in words])
    return stem_words


sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))

token_count = sum([len(sentence) for sentence in sentences])
print("The book corpus contains {0:,} tokens".format(token_count))
'''