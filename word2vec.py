#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function
import codecs
import glob
import multiprocessing
import os
import pprint
import re
import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from gensim.parsing.preprocessing import remove_stopwords
from gensim.models import Doc2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
import sys


# In[2]:


nltk.download('punkt')
nltk.download('stopwords')


# In[65]:


articles_filenames = sorted(glob.glob("./CleanedText/*.txt"))


# In[66]:


print(articles_filenames)


# In[67]:


#combining books into one string
corpus_raw = u""
for article_filename in articles_filenames:
    print("Reading '{0}'....".format(article_filename))
    with codecs.open(article_filename, "r", "utf-8") as artice_file:
        corpus_raw += artice_file.read()
    print("Corpus is now {0} characters long". format(len(corpus_raw)))
    print()


# In[68]:


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# In[69]:


raw_sentences = tokenizer.tokenize(corpus_raw)


# In[70]:


def sentence_to_wordlist(raw):
    raw = remove_stopwords(raw.lower())
    clean = re.sub("[^a-zA-Z]", " ", raw)
    words = clean.split()
    return words


# In[71]:


sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))


# In[73]:


token_count = sum([len(sentence) for sentence in sentences])
print("The book corpus contains {0:,} tokens".format(token_count))


# In[107]:


#Train Word2Vec
#Dimensionality of the resulting word vectors.
num_features = 100

#Minimum word count threshold
min_word_count = 3

#number of threads to run in parallel
num_workers = multiprocessing.cpu_count()

#context window length
context_size = 10

#Downsample setting for frequent words
downsampling = 1e-3

#seed for the RNG, to make the results reproducible
seed = 2


# In[108]:


articles2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)
articles2vec.build_vocab(sentences)


# In[109]:



print("Word2Vec vocabulary length:", len(articles2vec.wv.vocab))


# In[110]:


articles2vec.train(sentences, total_examples=articles2vec.corpus_count, epochs=15)


# In[111]:


if not os.path.exists("./AustLit/files/trained"):
    os.makedirs("./AustLit/files/trained")
    
    


# In[112]:


articles2vec.save(os.path.join("./AustLit/files/trained", "articles2vec.w2v"))


# In[113]:


articles2vec = w2v.Word2Vec.load(os.path.join("./AustLit/files/trained", "articles2vec.w2v"))


# In[114]:


tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)


# In[115]:


all_word_vectors_matrix = articles2vec.wv.vectors


# In[116]:


all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)


# In[119]:


points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[articles2vec.wv.vocab[word].index])
            for word in articles2vec.wv.vocab
        ]
    ],
    columns=["word", "x", "y"]
)


# In[120]:


print(points.head(10))


# In[121]:


sns.set_context("poster")


# In[122]:


fig = points.plot.scatter("x", "y", s=10, figsize=(20, 12)).get_figure()
fig.savefig('./plots/points-seed'+str(seed)+'.png')

# In[123]:


def plot_region(x_bounds, y_bounds):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) & 
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
    ]   
    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)
    ax.get_figure().savefig('./plots/slice-'+str(x_bounds[0])+'_'+str(y_bounds[0])+'.png')


# In[124]:


plot_region(x_bounds=(-20, 0), y_bounds=(-40, -30))


# In[125]:


plot_region(x_bounds=(-15, -10), y_bounds=(-32, -30))


# In[126]:


articles2vec.wv.most_similar("sydney")


# In[127]:


def nearest_similarity_cosmul(start1, end1, end2):
    similarities = articles2vec.wv.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2


# In[129]:


nearest_similarity_cosmul("sydney",  "stories", "melbourne")


# In[130]:


kmeans_model = KMeans(n_clusters=4, init='k-means++', max_iter=100) 
X = kmeans_model.fit(articles2vec.wv.vectors)
labels=kmeans_model.labels_.tolist()


# In[131]:


l = kmeans_model.fit_predict(articles2vec.wv.vectors)
pca = PCA(n_components=2).fit(articles2vec.wv.vectors)
datapoint = pca.transform(articles2vec.wv.vectors)


# In[132]:


plt.figure
# xmin=-5.
# xmax=None
# ymin=-5.
# ymax=None
label1 = ["#FFFF00", "#008000", "#0000FF", "#800080"]
color = [label1[i] for i in labels]
plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)

centroids = kmeans_model.cluster_centers_
centroidpoint = pca.transform(centroids)
plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
plt.axis([-5, 5, -5, 5])
# axes = plt.gca()
# axes.set_xlim([xmin,xmax])
# axes.set_ylim([ymin,ymax])
plt.savefig('./plots/centroid-seed'+str(seed)+'.png', dpi=500)
plt.show()



# In[133]:


def visualize(model, output_path):
    meta_file = "w2x_metadata.tsv"
    placeholder = np.zeros((len(model.wv.index2word), 100))
    with open(os.path.join(output_path,meta_file), 'wb') as file_metadata:
        for i, word in enumerate(model.wv.index2word):
            placeholder[i] = model[word]
            # temporary solution for https://github.com/tensorflow/tensorflow/issues/9094
            if word == '':
                print("Emply Line, should replecaed by any thing else, or will cause a bug of tensorboard")
                file_metadata.write("{0}".format('<Empty Line>').encode('utf-8') + b'\n')
            else:
                file_metadata.write("{0}".format(word).encode('utf-8') + b'\n')
    # define the model without training
    sess = tf.InteractiveSession()
    embedding = tf.Variable(placeholder, trainable = False, name = 'w2x_metadata')
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(output_path, sess.graph)
    # adding into projector
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = 'w2x_metadata'
    embed.metadata_path = meta_file
    # Specify the width and height of a single thumbnail.
    projector.visualize_embeddings(writer, config)
    saver.save(sess, os.path.join(output_path,'w2x_metadata.ckpt'))
    print('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(output_path))


# In[3]:

#HPC
#output_path="/home/users/tsaglik/DistributionalSemanticsProject/CleanedText"
#local
output_path="/Users/tolgasaglik/Desktop/mics-3/MachineLearning/DistributionalSemanticsProject/CleanedText"

model = articles2vec
visualize(model, output_path)

#HPC
#tensorboard --logdir= /home/users/tsaglik/DistributionalSemanticsProject/CleanedText --host localhost --port 8088

#local
#tensorboard --logdir= /Users/tolgasaglik/Desktop/mics-3/MachineLearning/DistributionalSemanticsProject/CleanedText --host localhost --port 8088
#on plain terminal
#python -m tensorboard.main --logdir CleanedText

# In[ ]:


#get_ipython().system(u'tensorboard --logdir= /home/users/tsaglik/DistributionalSemanticsProject/CleanedText --host localhost --port 8088')


# In[ ]:




