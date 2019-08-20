# Clustering with BERT

# %%

import torch
from pytorch_transformers import BertModel, BertTokenizer
pretrained_weights = "bert-base-multilingual-cased"

# %%

tokenizer = BertTokenizer.from_pretrained(
    pretrained_weights, do_lower_case=False)
model = BertModel.from_pretrained(
    pretrained_weights, output_hidden_states=True)

# %%


def vectorize(x, layer=-2, reduce_fn='avg'):
    if len(x) == 0:
        return vectorize("null", layer, reduce_fn)

    input_ids = torch.tensor([tokenizer.encode(x)])[:, :512]

    if input_ids.size(1) == 0:
        return vectorize("null", layer, reduce_fn)

    with torch.no_grad():
        out = model(input_ids)

    encoded = out[2][layer].squeeze(0)

    if not reduce_fn:
        return encoded

    if reduce_fn == "avg":
        return encoded.mean((0))

    raise ValueError("Reduce function {} not known".format(reduce_fn))


# %%


def similarity(x, y):
    return torch.dot(x, y) / x.norm() / y.norm()


# %%

# Load data

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

import numpy as np
from tqdm import tqdm
X_train = np.array([vectorize(x).numpy() for x in tqdm(train_data)])

# %%
X_test = np.array([vectorize(x).numpy() for x in tqdm(test_data)])

# %%

# Sanity check

from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
clf = GaussianNB()
clf.fit(X_train, y_train)
metrics.accuracy_score(clf.predict(X_train), y_train)

# Accuracy: 0.6230

# %%

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

# %%
