# Document clustering with Python

# %%

# Get data

from sklearn.datasets import fetch_20newsgroups

categories = [
    'comp.graphics',
    'rec.autos',
    'rec.sport.baseball',
    'sci.space',
    'soc.religion.christian',
    'talk.politics.guns',
]

newsgroup_train = fetch_20newsgroups(
    subset='train',
    categories=categories,
    remove=('headers', 'footers', 'quotes'))
newsgroup_test = fetch_20newsgroups(
    subset='test',
    categories=categories,
    remove=('headers', 'footers', 'quotes'))
train_data = newsgroup_train['data']
test_data = newsgroup_test['data']
y_train = newsgroup_train['target']
y_test = newsgroup_test['target']

# %%

# Save data
import json
with open("train.jsonl", "w") as w:
    for x, y in zip(train_data, y_train):
        json.dump({"y": int(y), "x": x}, w)
        w.write("\n")

# %%

with open("dev.jsonl", "w") as w:
    for x, y in zip(test_data, y_test):
        json.dump({"y": int(y), "x": x}, w)
        w.write("\n")

# %%

# Initial data exploration

# How many examples per category?

from collections import Counter
Counter(y_train)

# %%

# Distribution of document length?

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import mpld3
lengths = list(map(len, train_data))
le = pd.Series(lengths)
plt.clf()
le[le < 2000].hist()
mpld3.show(open_browser=False)

# %%

# Remove stopwords

import nltk
stopwords = nltk.corpus.stopwords.words('english')

# %%

# Tokenizing + stemming

import spacy
import re
from functools import partial
nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser', 'tagger'])


def remove_spaces(tokens):
    return [token for token in tokens if not token.is_space]


def tokenize(text,
             nlp,
             stemming=False,
             remove_stopwords=False,
             lowercase=False,
             remove_nonwords=False):
    tokens = [
        x.lemma_ if stemming else x.text for x in remove_spaces(nlp(text))
    ]
    if lowercase:
        tokens = [t.lower() for t in tokens]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    if remove_nonwords:
        tokens = [t for t in tokens if re.search('[a-zA-Z]', t)]

    return tokens


# %%

example = train_data[0]

tokenize_stem_fn = partial(
    tokenize,
    nlp=nlp,
    stemming=True,
    remove_stopwords=True,
    lowercase=True,
    remove_nonwords=True)

tokenize_only_fn = partial(
    tokenize,
    nlp=nlp,
    stemming=False,
    remove_stopwords=False,
    lowercase=False,
    remove_nonwords=False)

tokenize_stem_fn(example)
tokenize_only_fn(example)

# %%

# TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(
    max_df=0.8, max_features=200000, tokenizer=tokenize_stem_fn)
X_train = tfidf_vectorizer.fit_transform(train_data)
X_test = tfidf_vectorizer.transform(test_data)
X_train.shape

terms = tfidf_vectorizer.get_feature_names()

# %%

# Interlude: Classify with Naive Bayes

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

clf = MultinomialNB(alpha=0.01)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, pred)
accuracy

# Accuracy : 0.8738

# %%

# Top features?
import numpy as np

for i, category in enumerate(categories):
    best_features = np.argsort(clf.coef_[i])[-10:]
    print(
        "{}: {}".format(category, " ".join(np.asarray(terms)[best_features])))

# %%

# Kmeans

from sklearn.cluster import KMeans

num_clusters = 6
km = KMeans(n_clusters=num_clusters)
km.fit(X_train)
clusters = km.labels_.tolist()

# %%
from itertools import permutations

# Best key on train?

best_key = None
best_score = None
for key in permutations(list(range(num_clusters))):
    score = metrics.accuracy_score(y_train, np.array(key)[clusters])
    if best_score is None or score > best_score:
        best_score = score
        best_key = key

best_key = np.array(best_key)
best_score

# 0.5767

# %%

# Closest k words to each cluster centers

order_centroids = km.cluster_centers_.argsort()[:, ::-1]
num_words = 10

for i in range(num_clusters):
    print("Cluster {} words: ".format(categories[best_key[i]]), end='')
    print("{}".format(",".join([terms[x] for x in order_centroids[i, :10]])))

# %%

# Predicting

kmeans_pred = km.predict(X_test)

pred = best_key[kmeans_pred]
metrics.accuracy_score(y_test, pred)

# F1: 0.5547

# %%

# Use Multidimensional sclaing to plot some examples

from sklearn.manifold import MDS
mds = MDS()
pos = mds.fit_transform(X_train.toarray())
xs, ys = pos[:, 0], pos[:, 1]

# %%

cluster_colors = {
    0: '#1b9e77',
    1: '#d95f02',
    2: '#7570b3',
    3: '#e7298a',
    4: '#66a61e',
    5: '#2233cc'
}
cluster_names = categories

n_points = 100

df = pd.DataFrame(
    dict(
        x=xs[:n_points],
        y=ys[:n_points],
        #  label=best_key[clusters[:n_points]],
        label=y_train[:n_points],
        excerpt=[x[:200] for x in train_data[:100]]))

# %%

#create data frame that has the result of the MDS plus the cluster numbers and titles

#group by cluster
groups = df.groupby('label')

#define custom css to format the font and to remove the axis labeling
css = """
text.mpld3-text, div.mpld3-tooltip {
  font-family:Arial, Helvetica, sans-serif;
}

g.mpld3-xaxis, g.mpld3-yaxis {
display: none; }

svg.mpld3-figure {
margin-left: -200px;
}

.mpld3-tooltip {
weight: 200px;
}
"""


#define custom toolbar location
class TopToolbar(mpld3.plugins.PluginBase):
    """Plugin for moving toolbar to top of figure"""

    JAVASCRIPT = """
    mpld3.register_plugin("toptoolbar", TopToolbar);
    TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
    TopToolbar.prototype.constructor = TopToolbar;
    function TopToolbar(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    TopToolbar.prototype.draw = function(){
      // the toolbar svg doesn't exist
      // yet, so first draw it
      this.fig.toolbar.draw();

      // then change the y position to be
      // at the top of the figure
      this.fig.toolbar.toolbar.attr("x", 150);
      this.fig.toolbar.toolbar.attr("y", 400);

      // then remove the draw function,
      // so that it is not called again
      this.fig.toolbar.draw = function() {}
    }
    """

    def __init__(self):
        self.dict_ = {"type": "toptoolbar"}


# Plot
fig, ax = plt.subplots(figsize=(14, 6))  #set plot size
ax.margins(0.03)  # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    points = ax.plot(
        group.x,
        group.y,
        marker='o',
        linestyle='',
        ms=18,
        label=cluster_names[name],
        mec='none',
        color=cluster_colors[name])
    ax.set_aspect('auto')
    labels = [i for i in group.excerpt]

    #set tooltip using points, labels and the already defined 'css'
    tooltip = mpld3.plugins.PointHTMLTooltip(
        points[0], labels, voffset=10, hoffset=10, css=css)
    #connect tooltip to fig
    mpld3.plugins.connect(fig, tooltip, TopToolbar())

    #set tick marks as blank
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    #set axis as blank
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

ax.legend(numpoints=1)  # show legend with only one dot

mpld3.show()  # show the plot

# %%

# Using LDA

from gensim import corpora, models, similarities

tokenized_train = [tokenize_stem_fn(x) for x in train_data]
tokenized_test = [tokenize_stem_fn(x) for x in test_data]

# %%

dictionary = corpora.Dictionary(tokenized_train)
dictionary.filter_extremes(no_below=1, no_above=0.8)
corpus = [dictionary.doc2bow(text) for text in tokenized_train]

# %%

lda = models.LdaModel(
    corpus,
    num_topics=6,
    id2word=dictionary,
    update_every=5,
    chunksize=10000,
    passes=100)

# %%

topics_matrix = lda.show_topics(formatted=False, num_words=20)
topics_matrix

lda.save("ldamodel")

# %%
