import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from os.path import exists
import re

from elasticsearch import Elasticsearch, helpers
import eland as ed

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union

import hdbscan
import umap
from functools import partial
from hyperopt import fmin, tpe, hp, STATUS_OK, space_eval, Trials

from sentence_transformers import SentenceTransformer
import spacy

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import scipy.sparse as sp
from sklearn.cluster import KMeans
from styleframe import StyleFrame
import requests

app = FastAPI()


class TopicModelingParams(BaseModel):
    min_topics: Union[int, None] = 200
    max_topics: Union[int, None] = 2000
    max_topic_size: Union[int, None] = 20000
    number_of_trials: Union[int, None] = 200
    ngram_range: Union[tuple, None] = (1, 3)
    top_n_words: Union[int, None] = 40
    diversity: Union[float, None] = 0.2
    dataset_hash: Union[str, None] = '...'# TODO


class PredictionParams(BaseModel):
    threshold: Union[float, None] = None
    dataset_hash: Union[str, None] = '...'# TODO


@app.get('/heartbeat')
def heartbeat():
    return 'OK'


def save_data(data, filepath):
    """Save data to disk using pickle."""

    with open(filepath, 'wb') as handle:
        pickle.dump(data, handle)


def load_data(filepath):
    """Load and return the given pickle file."""

    with open(filepath, 'rb') as handle:
        return pickle.load(handle)


def loging_to_dashboard(msg: str, status: str) -> None:
    url = "http://dashboard:3000/api/notification/create"
    timestamped_msg = f'{datetime.now().strftime("%H:%M:%S")} --- {msg}'
    try:
        requests.post(
            url=url,
            json={"type": status, "message": timestamped_msg}
        )
    except Exception as e:
        print("No dashboard:")
    finally:
        print(timestamped_msg)


def query_es_db(es, dataset_hash, is_prediction=False):
    """Query relevant records from ES DB.
        Return them in a Pandas DataFrame."""

    df = ed.DataFrame(es_client=es, es_index_pattern=dataset_hash, columns=['abstract',
                                                                            'is_relevant',
                                                                            'relevance_value',
                                                                            'articletitle',
                                                                            'pathway_number',
                                                                            'benefit',
                                                                            'pathway_benefit',
                                                                            'predicted_pathway_numbers'])
    if is_prediction:
        return df[(df['is_relevant'] == True) &
                  (df['predicted_pathway_numbers'] == 'notchecked')].to_pandas().rename(
            columns={'abstract': 'Abstract'})
    else:
        return df[df['is_relevant'] == True].to_pandas().rename(columns={'abstract': 'Abstract'})


def UMAP_then_HDBSCAN(embeddings,
                      n_neighbors,
                      n_components,
                      min_cluster_size,
                      min_samples=None,
                      random_state=None,
                      low_memory=True,
                      min_dist=0.0,
                      prediction_data=False,
                      final_run=False):
    """Perform dimensionality reduction using UMAP,
        then use HDBSCAN to generate clusters."""

    ### Initialize UMAP model
    umap_model = umap.UMAP(n_neighbors=n_neighbors,
                           n_components=n_components,
                           metric='cosine',
                           random_state=42,
                           low_memory=low_memory,
                           min_dist=min_dist)

    ### Fit UMAP on embeddings
    umap_model.fit(embeddings)

    ### Transform the embeddings using the fitted UMAP model
    umap_embeddings = umap_model.transform(embeddings)

    ### Replace any nan or inf with numbers (0 or high number)
    umap_embeddings = np.nan_to_num(umap_embeddings)

    ### Initialize HDBSCAN model
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                    min_samples=min_samples,
                                    metric='euclidean',
                                    gen_min_span_tree=True,
                                    cluster_selection_method='eom',
                                    prediction_data=prediction_data)

    ### Fit HDBSCAN on UMAP-embeddings
    clusters = hdbscan_model.fit(umap_embeddings)

    if final_run:
        return clusters, umap_model
    else:
        return clusters


def evaluate_clusters(clusters, prob_threshold=0.05):
    """Calculate the label_count, data_loss, and largest_cluster_size,
        which are the number of clusters, the amount of noise, and the
        number of data points placed in the largest cluster, respectively."""

    cluster_labels = clusters.labels_
    label_count = len(np.unique(cluster_labels))
    total_num = len(clusters.labels_)
    data_loss = (np.count_nonzero(clusters.probabilities_ < prob_threshold) / total_num)
    largest_cluster_size = max([sum(cluster_labels == i) for i in range(label_count - 1)])

    return label_count, data_loss, largest_cluster_size


def objective_function(params, embeddings, min_topics, max_topics, max_topic_size):
    """Objective function for hyperopt to minimize."""

    clusters = UMAP_then_HDBSCAN(embeddings,
                                 n_neighbors=params['n_neighbors'],
                                 n_components=params['n_components'],
                                 min_cluster_size=params['min_cluster_size'],
                                 min_samples=params['min_samples'],
                                 random_state=params['random_state'])

    label_count, data_loss, largest_cluster_size = evaluate_clusters(clusters, prob_threshold=0.05)

    ### penalty on the loss function if outside the desired range of topics
    if (label_count < min_topics) | (label_count > max_topics):
        penalty = 0.9
    ### penalty on the loss function if max cluster size is too large
    elif largest_cluster_size > max_topic_size:
        penalty = 0.9
    else:
        penalty = 0

    loss = data_loss + penalty

    return {'loss': loss,
            'label_count': label_count,
            'largest_cluster_size': largest_cluster_size,
            'status': STATUS_OK}


def bayesian_search(embeddings,
                    space,
                    min_topics,
                    max_topics,
                    max_topic_size,
                    max_evals=200):
    """Perform bayesian search on hyperparameter space using hyperopt."""

    trials = Trials()
    fmin_objective = partial(objective_function,
                             embeddings=embeddings,
                             min_topics=min_topics,
                             max_topics=max_topics,
                             max_topic_size=max_topic_size)

    best = fmin(fmin_objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)

    best_params = space_eval(space, best)

    best_hdbscan_model, best_umap_model = UMAP_then_HDBSCAN(embeddings,
                                                            n_neighbors=best_params['n_neighbors'],
                                                            n_components=best_params['n_components'],
                                                            min_cluster_size=best_params['min_cluster_size'],
                                                            min_samples=best_params['min_samples'],
                                                            random_state=best_params['random_state'],
                                                            prediction_data=True,
                                                            final_run=True)

    save_data(trials, 'data/bayesian_search/trials.pickle')
    save_data(best_hdbscan_model, 'data/bayesian_search/best_hdbscan_model.pickle')
    save_data(best_umap_model, 'data/bayesian_search/best_umap_model.pickle')

    return best_hdbscan_model


def find_optimal_topics(documents, min_topics, max_topics, max_topic_size, max_evals):
    """Run Bayesian search if there is not existing HDBSCAN model."""

    hdbscan_model_path = 'data/bayesian_search/best_hdbscan_model.pickle'
    embeddings_path = 'embeddings/test_embeddings.pickle'

    if exists(hdbscan_model_path):
        msg = 'Loading existing model'
        loging_to_dashboard(msg=msg, status='info')
        hdbscan_model = load_data(hdbscan_model_path)
    else:
        msg = 'Existing model is unavailable'
        loging_to_dashboard(msg=msg, status='info')
        msg = 'Starting Bayesian search'
        loging_to_dashboard(msg=msg, status='info')
        if not exists(embeddings_path): return None
        embeddings = load_data(embeddings_path)
        embeddings = {k: embeddings[k] for k in documents.index.values}
        embeddings = np.array(list(embeddings.items()), dtype=object)
        embeddings = np.array([embeddings[i, 1] for i in range(len(embeddings))])

        hspace = {
            "n_neighbors": hp.choice('n_neighbors', range(5, 100)),
            "n_components": hp.choice('n_components', range(3, 10)),
            "min_cluster_size": hp.choice('min_cluster_size', range(10, 200)),
            "min_samples": hp.choice('min_samples', range(5, 50)),
            "random_state": 42
        }

        hdbscan_model = bayesian_search(embeddings=embeddings,
                                        space=hspace,
                                        min_topics=min_topics,
                                        max_topics=max_topics,
                                        max_topic_size=max_topic_size,
                                        max_evals=max_evals)

    return hdbscan_model


def get_topic_sizes(documents):
    """Calculate topic sizes."""

    sizes = documents.groupby(['Topic']).count().sort_values("Abstract", ascending=False).reset_index()
    topic_sizes = dict(zip(sizes.Topic, sizes.Abstract))

    return topic_sizes


def transform_documents(documents, hdbscan_model):
    """Reorganize documents dataframe and creates a single
        document per topic for the ctfidf."""

    ### Add Topic column to documents DataFrame
    ### and remove unnecessary columns / data
    documents['Topic'] = hdbscan_model.labels_
    documents.reset_index(inplace=True)
    documents.rename(columns={'index': '_id'}, inplace=True)
    documents.drop('is_relevant', axis=1, inplace=True)
    documents['relevance_value'] = documents['relevance_value'].map(lambda x: x[0][0])

    ### Calculate the topic sizes
    topic_sizes = get_topic_sizes(documents)

    ### Aggregate documents per topic
    documents_per_topic = documents.groupby(['Topic'], as_index=False, sort=True).agg({'Abstract': ' '.join})
    documents_per_topic = documents_per_topic['Abstract'].values[1:]
    stopwords_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
                      'you', "you're", "you've", "you'll", "you'd", 'your', 'yours',
                      'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
                      "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
                      'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                      'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
                      'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                      'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
                      'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
                      'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                      'through', 'during', 'before', 'after', 'above', 'below', 'to',
                      'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                      'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
                      'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
                      'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                      'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
                      "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
                      've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                      "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
                      'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
                      'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
                      "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
                      'wouldn', "wouldn't"]

    ### Basic preprocessing of text
    ### Steps:
    ###     * Lower text
    ###     * Replace \n and \t with whitespace
    ###     * Only keep alpha-numerical characters
    ###     * Use spacy lemmatization
    documents_per_topic = [doc.lower() for doc in documents_per_topic]
    documents_per_topic = [doc.replace("\n", " ") for doc in documents_per_topic]
    documents_per_topic = [doc.replace("\t", " ") for doc in documents_per_topic]
    documents_per_topic = [re.sub(r'[^A-Za-z0-9 ]+', '', doc) for doc in documents_per_topic]
    documents_per_topic = [doc if doc != "" else "emptydoc" for doc in documents_per_topic]

    for j, doc in enumerate(documents_per_topic):
        subdocs = list()
        for i in range((len(doc) // 1000000) + 1):
            nlp_doc = nlp(doc[i * 1000000:min((i + 1) * 1000000, len(doc))])
            lemma_list = [w.lemma_
                          for w in nlp_doc
                          if (len(w.lemma_) > 2) and (w.lemma_ not in stopwords_list)]
            subdocs.append(' '.join(lemma_list))
        documents_per_topic[j] = ' '.join(subdocs)

    return documents, documents_per_topic, topic_sizes


def get_doc_term_mx(documents_per_topic, ngram_range):
    """Fit Vectorizer model and calculate the doc-term sparse matrix."""

    cv_model = CountVectorizer(ngram_range=ngram_range, stop_words="english")
    cv_model.fit(documents_per_topic)
    words = cv_model.get_feature_names_out()
    X = cv_model.transform(documents_per_topic)

    return words, X


def get_c_tf_idf_mx(X):
    """Learn the idf vector (global term weights)
        and calculate cTFIDF scores."""

    _, n_features = X.shape
    ### Calculate the frequency of words across all classes
    df = np.squeeze(np.asarray(X.sum(axis=0)))

    ### Calculate the average number of samples as regularization
    avg_nr_samples = int(X.sum(axis=1).mean())

    ### Divide the average number of samples by the word frequency
    ### +1 is added to force values to be positive
    idf = np.log((avg_nr_samples / df) + 1)
    idf_diag = sp.diags(idf, offsets=0,
                        shape=(n_features, n_features),
                        format='csr',
                        dtype=np.float64)

    ### Transform a count-based matrix to c-TF-IDF
    X = normalize(X, axis=1, norm='l1', copy=False)
    X = X * idf_diag  # X = c_tf_idf

    return X


def diversify(doc_embedding, word_embeddings, words, top_n, diversity):
    """ Calculate Maximal Marginal Relevance (MMR)
    between candidate keywords and the document.
    """
    ### Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    ### Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        ### Extract similarities within candidates and
        ### between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        ### Calculate MMR
        mmr = (1 - diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        ### Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]


def get_top_n_words(words, X, labels, embedding_model, top_n_words, diversity):
    """Extract the top n words / keywords per topic.
        Number of candidate words (n) = 2 x top_n_words."""

    ### Get the top n indices and values per row in a sparse c-TF-IDF matrix
    ### Return indices of top n values in each row of a sparse matrix
    indices = []
    n = top_n_words * 2
    for le, ri in zip(X.indptr[:-1], X.indptr[1:]):
        n_row_pick = min(n, ri - le)
        values = X.indices[le + np.argpartition(X.data[le:ri], -n_row_pick)[-n_row_pick:]]
        values = [values[index] if len(values) >= index + 1 else None for index in range(n)]
        indices.append(values)
    indices = np.array(indices)

    ### Return the top n values for each row in a sparse matrix
    top_values = []
    for row, values in enumerate(indices):
        scores = np.array([X[row, value] if value is not None else 0 for value in values])
        top_values.append(scores)
    top_values = np.array(top_values)

    sorted_indices = np.argsort(top_values, 1)
    indices = np.take_along_axis(indices, sorted_indices, axis=1)
    scores = np.take_along_axis(top_values, sorted_indices, axis=1)

    ### Get top n words per topic based on c-TF-IDF score
    topics = {label: [(words[word_index], score)
                      if
                      (word_index is not None) and
                      (score > 0) and
                      ('paper' not in words[word_index]) and
                      ('article' not in words[word_index]) and
                      ('publish' not in words[word_index]) and
                      ('author' not in words[word_index])
                      else ("", 0.00001)
                      for word_index, score in zip(indices[index][::-1], scores[index][::-1])
                      ]
              for index, label in enumerate(labels)}

    ### Extract word embeddings for the top n words per topic and compare it
    ### with the topic embedding to keep only the words most similar to the topic embedding
    for topic, topic_words in topics.items():
        words = [word[0] for word in topic_words]
        word_embeddings = embedding_model.encode(words)
        topic_embedding = embedding_model.encode(" ".join(words)).reshape(1, -1)
        topic_words = diversify(topic_embedding, word_embeddings, words,
                                top_n=top_n_words, diversity=diversity)
        topics[topic] = [(word, value) for word, value in topics[topic] if word in topic_words]
    topics = {label: values[:top_n_words] for label, values in topics.items()}

    return topics


def extract_topic_descriptions(documents,
                               hdbscan_model,
                               ngram_range,
                               top_n_words,
                               diversity,
                               embedding_model):
    """Extract cTFIDF and keywords dictionary based
        topic descriptions."""

    documents, documents_per_topic, topic_sizes = transform_documents(documents, hdbscan_model)
    words, X = get_doc_term_mx(documents_per_topic, ngram_range)
    X = get_c_tf_idf_mx(X)
    labels = sorted(list(topic_sizes.keys()))
    labels = labels[1:]
    topics = get_top_n_words(words, X, labels, embedding_model, top_n_words, diversity)

    words = load_data('data/topic_modeling/keywords_dictionary.pickle')
    X = np.zeros((len(documents_per_topic), len(words)))
    for i, doc in enumerate(documents_per_topic):
        for j, kw in enumerate(words):
            X[i, j] = doc.count(kw)
    X = sp.csr_matrix(X)
    topics_kw = get_top_n_words(words, X, labels, embedding_model, top_n_words, diversity)

    return topics, topics_kw


def find_representative_docs(documents, hdbscan_model):
    """ Save the most representative docs (10) per topic.
        The most representative docs are extracted by taking
        the exemplars from the HDBSCAN-generated clusters.
        Full instructions can be found here:
            https://hdbscan.readthedocs.io/en/latest/soft_clustering_explanation.html
    """
    ### Prepare the condensed tree and leaf clusters beneath a given cluster
    condensed_tree = hdbscan_model.condensed_tree_
    raw_tree = condensed_tree._raw_tree
    clusters = sorted(condensed_tree._select_clusters())
    cluster_tree = raw_tree[raw_tree['child_size'] > 1]

    ### Find the points with maximum lambda value in each leaf
    representative_docs = {}
    representative_docs_title = {}
    representative_docs_combos = {}
    for topic in documents['Topic'].unique():

        if topic != -1:

            leaves = hdbscan.plots._recurse_leaf_dfs(cluster_tree, clusters[topic])
            result = np.array([])

            for leaf in leaves:
                max_lambda = raw_tree['lambda_val'][raw_tree['parent'] == leaf].max()
                points = raw_tree['child'][(raw_tree['parent'] == leaf) & (raw_tree['lambda_val'] == max_lambda)]
                result = np.hstack((result, points))

            ### 10x representative docs with the highest relevance value for each topic
            representative_docs[topic] = documents.iloc[result].sort_values('relevance_value', ascending=False)[
                                             'Abstract'].values[:10]
            representative_docs_title[topic] = documents.iloc[result].sort_values('relevance_value', ascending=False)[
                                                   'articletitle'].values[:10]
            representative_docs_combos[topic] = documents.iloc[result].sort_values('relevance_value', ascending=False)[
                                                    'pathway_benefit'].values[:10]

    return representative_docs, representative_docs_title, representative_docs_combos


def get_unique_elements(nested_list):
    """Find unique elements in a nested list."""

    nested_list_list = [ww
                        for w in nested_list
                        if isinstance(w, list)
                        for ww in w]
    nested_list_single = [w
                          for w in nested_list
                          if not isinstance(w, list)]
    nested_list_single.extend(nested_list_list)
    unique_elements = list(set(nested_list_single))

    return unique_elements


def generate_topic_summary(documents, hdbscan_model, embedding_model, topics, topics_kw):
    """Create spreadsheet from topic number, topic size, topic descriptions,
        and representative documents title, abstract, search query combinations."""

    representative_docs, representative_docs_title, representative_docs_combos = find_representative_docs(documents,
                                                                                                          hdbscan_model)

    topic_embeddings = dict()
    topic_sizes = get_topic_sizes(documents)
    topic_num = len(topic_sizes)

    for topic, topic_words in topics.items():
        words = [word[0] for word in topic_words]
        topic_embeddings[topic] = embedding_model.encode(" ".join(words)).reshape(1, -1)

    ### Kmeans to cluster similar topics
    te = np.array([v[0] for k, v in topic_embeddings.items() if k != -1])
    n_clusters = topic_num // 20 + 1
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(te)
    sorting_indices = np.argsort(kmeans.labels_)  # this is the order how topics should be added to the spreadsheet

    pathway_numbers = get_unique_elements(documents['pathway_number'].values)
    benefits = get_unique_elements(documents['benefit'].values)
    combos = list()
    for pn in pathway_numbers:
        for bl in benefits:
            key = str(pn) + str(bl)
            combos.append(key)

    topic_modeling_summary = list()
    topic_modeling_summary.append(('Topic number',
                                   'Topic size',
                                   'Topic description',
                                   'Topic search keywords',
                                   'Search query pathway distribution',
                                   'Search query co-benefit distribution',
                                   'Search query combination distribution (TOP 10)',
                                   'Is topic relevant? (FILL THIS COLUMN)',
                                   'Relevant Pathway(s) (FILL THIS COLUMN)',
                                   'Relevant Co-benefit(s) (FILL THIS COLUMN)',
                                   'Topic label (FILL THIS COLUMN)',
                                   'Comments (FILL THIS COLUMN)',
                                   '',
                                   'Paper #1 - Title', 'Paper #1 - Abstract', 'Paper #1 - Search query combinations',
                                   'Paper #2 - Title', 'Paper #2 - Abstract', 'Paper #2 - Search query combinations',
                                   'Paper #3 - Title', 'Paper #3 - Abstract', 'Paper #3 - Search query combinations',
                                   'Paper #4 - Title', 'Paper #4 - Abstract', 'Paper #4 - Search query combinations',
                                   'Paper #5 - Title', 'Paper #5 - Abstract', 'Paper #5 - Search query combinations',
                                   'Paper #6 - Title', 'Paper #6 - Abstract', 'Paper #6 - Search query combinations',
                                   'Paper #7 - Title', 'Paper #7 - Abstract', 'Paper #7 - Search query combinations',
                                   'Paper #8 - Title', 'Paper #8 - Abstract', 'Paper #8 - Search query combinations',
                                   'Paper #9 - Title', 'Paper #9 - Abstract', 'Paper #9 - Search query combinations',
                                   'Paper #10 - Title', 'Paper #10 - Abstract',
                                   'Paper #10 - Search query combinations'))

    for si in sorting_indices:
        words = [word[0] for word in topics[si]]
        words_kw = [word[0] for word in topics_kw[si]]

        representative_doc_data = list()
        for i in range(len(representative_docs[si])):
            representative_doc_data.extend([representative_docs_title[si][i],
                                            representative_docs[si][i],
                                            representative_docs_combos[si][i]])

        single_topic_documents = documents[hdbscan_model.labels_ == si]

        pathway_proportions = dict()
        benefit_proportions = dict()
        combo_proportions = dict()

        for pn in pathway_numbers:
            pathway_proportions['#' + str(pn)] = 0
        for bl in benefits:
            benefit_proportions['#' + str(bl)] = 0
        for cb in combos:
            combo_proportions['#' + str(cb)] = 0

        number_of_pathways = 0
        number_of_benefits = 0
        number_of_combos = 0

        for uid in single_topic_documents.index.values:

            current_pathway_number = single_topic_documents.loc[uid]['pathway_number']
            if isinstance(current_pathway_number, str):
                pathway_proportions['#' + current_pathway_number] += 1
                number_of_pathways += 1
            else:
                for cpn in current_pathway_number:
                    pathway_proportions['#' + str(cpn)] += 1
                    number_of_pathways += 1

            current_benefit = single_topic_documents.loc[uid]['benefit']
            if isinstance(current_benefit, str):
                benefit_proportions['#' + current_benefit] += 1
                number_of_benefits += 1
            else:
                for cbl in current_benefit:
                    benefit_proportions['#' + str(cbl)] += 1
                    number_of_benefits += 1

            current_combos = single_topic_documents.loc[uid]['pathway_benefit']
            if isinstance(current_combos, str):
                combo_proportions['#' + current_combos] += 1
                number_of_combos += 1
            else:
                for ccb in current_combos:
                    combo_proportions['#' + str(ccb)] += 1
                    number_of_combos += 1

        pathway_occurences = {k: v for k, v in
                              sorted(pathway_proportions.items(), key=lambda item: item[1], reverse=True) if v > 0}
        benefit_occurences = {k: v for k, v in
                              sorted(benefit_proportions.items(), key=lambda item: item[1], reverse=True) if v > 0}
        combo_occurences = {k: v for k, v in sorted(combo_proportions.items(), key=lambda item: item[1], reverse=True)
                            if v > 0}

        pathway_proportions = {k: round(v / number_of_pathways * 100, 2) for k, v in
                               sorted(pathway_proportions.items(), key=lambda item: item[1], reverse=True) if v > 0}
        benefit_proportions = {k: round(v / number_of_benefits * 100, 2) for k, v in
                               sorted(benefit_proportions.items(), key=lambda item: item[1], reverse=True) if v > 0}
        combo_proportions = {k: round(v / number_of_combos * 100, 2) for k, v in
                             sorted(combo_proportions.items(), key=lambda item: item[1], reverse=True) if v > 0}

        first10pairs_occurences = {k: combo_occurences[k] for k in list(combo_occurences)[:10]}
        first10pairs_proportions = {k: combo_proportions[k] for k in list(combo_proportions)[:10]}

        pathway_string = ''
        for k, v in pathway_occurences.items():
            pathway_string += k + ': ' + str(v) + ' (' + str(pathway_proportions[k]) + '%)' + '\n'

        benefit_string = ''
        for k, v in benefit_occurences.items():
            benefit_string += k + ': ' + str(v) + ' (' + str(benefit_proportions[k]) + '%)' + '\n'

        combo_string = ''
        for k, v in first10pairs_occurences.items():
            combo_string += k + ': ' + str(v) + ' (' + str(first10pairs_proportions[k]) + '%)' + '\n'

        topic_info = [si, topic_sizes[si], words, words_kw, pathway_string, benefit_string, combo_string, '', '', '',
                      '', '', '']
        topic_info.extend(representative_doc_data)

        topic_modeling_summary.append(tuple(topic_info))

    topic_modeling_summary_df = pd.DataFrame(topic_modeling_summary)

    return topic_modeling_summary_df


def get_elasticsearch_object():
    """Create the ES object."""

    # TODO
    """elasticPort = ...
    elasticScheme = ...
    elasticTimeout = ...
    elasticHost = ...
    elasticUsername = ...
    elasticPassword = ..."""

    es = Elasticsearch(
        [elasticHost],
        http_auth=(elasticUsername, elasticPassword),
        scheme=elasticScheme,
        port=elasticPort,
        request_timeout=elasticTimeout,
    )

    return es


@app.post("/topic_modeling")
def run_tm(params: TopicModelingParams):
    """
        Step 1: Run bayesian search to find optimal parameters
        Step 2: Run topic modeling with the optimal parameters
        Step 3: Extract topic descriptions both with cTFIDF and with keywords
        Step 4: Generate topic summary spreadsheet
    """

    msg = 'Initializing'
    loging_to_dashboard(msg=msg, status='info')
    es = get_elasticsearch_object()

    ### Parameter to select Dataset
    dataset_hash = params.dataset_hash

    ### Parameters for Bayesian search
    min_topics = params.min_topics
    max_topics = params.max_topics
    max_topic_size = params.max_topic_size
    max_evals = params.number_of_trials

    ### Parameters for Topic descriptions
    ngram_range = tuple(params.ngram_range)
    top_n_words = params.top_n_words
    diversity = params.diversity

    embedding_model_type = 'all-mpnet-base-v1'
    embedding_model = SentenceTransformer(embedding_model_type)

    msg = 'Query relevant documents'
    loging_to_dashboard(msg=msg, status='info')
    documents = query_es_db(es, dataset_hash)
    if not documents.shape[0]:
        msg = 'There are no relevant papers in the database'
        loging_to_dashboard(msg=msg, status='info')
        msg = 'Run Relevance Classification first'
        loging_to_dashboard(msg=msg, status='info')
        return None
    msg = 'Find optimal topics'
    loging_to_dashboard(msg=msg, status='info')
    hdbscan_model = find_optimal_topics(documents, min_topics, max_topics, max_topic_size, max_evals)
    if hdbscan_model is None:
        msg = 'Pre-computed embeddings can not be found'
        loging_to_dashboard(msg=msg, status='info')
        msg = 'Run Relevance Classification first'
        loging_to_dashboard(msg=msg, status='info')
        return None
    if documents.shape[0] != len(hdbscan_model.labels_):
        msg = 'Mismatch between number of papers and number of label'
        loging_to_dashboard(msg=msg, status='info')
        msg = 'Make sure to use the exact same papers ' \
              'when continuing from a saved Bayesian search result'
        loging_to_dashboard(msg=msg, status='info')
        return None
    msg = 'Extract topic descriptions'
    loging_to_dashboard(msg=msg, status='info')
    topics, topics_kw = extract_topic_descriptions(documents,
                                                   hdbscan_model,
                                                   ngram_range,
                                                   top_n_words,
                                                   diversity,
                                                   embedding_model)
    msg = 'Generate topic summary spreadsheet'
    loging_to_dashboard(msg=msg, status='info')
    topic_modeling_summary_df = generate_topic_summary(documents, hdbscan_model, embedding_model, topics, topics_kw)

    training_documents = {documents.iloc[i]['_id']: documents.iloc[i]['Topic']
                          for i in range(documents.shape[0])
                          if documents.iloc[i]['Topic'] != -1}
    save_data(training_documents, 'data/topic_modeling/training_documents.pickle')
    StyleFrame(topic_modeling_summary_df).to_excel('data/topic_modeling/topic_modeling_summary.xlsx').save()

    msg = 'Topic modeling is done'
    loging_to_dashboard(msg=msg, status='info')


def update_data_generator(list_of_update_actions):
    """Generator function that yields next update action upon invoking."""

    for update_action in list_of_update_actions:
        yield update_action


@app.post("/prediction")
def run_pred(params: PredictionParams):
    """Predict NCS pathway and co-benefit categories of papers
        based on existing topic models and their labels."""

    msg = 'Start: TOPIC MODELING'
    loging_to_dashboard(msg=msg, status='info')
    es = get_elasticsearch_object()

    ### Parameter to select Dataset
    dataset_hash = params.dataset_hash

    msg = 'Querying data from Elasticsearch database'
    loging_to_dashboard(msg=msg, status='info')
    documents = query_es_db(es, dataset_hash, True)
    if documents.shape[0]:
        msg = f'{documents.shape[0]} records were queried from Elasticsearch database'
        loging_to_dashboard(msg=msg, status='info')
    else:
        msg = '-> Either there are no relevant papers in the database'
        loging_to_dashboard(msg=msg, status='info')
        msg = '---> Then run Relevance Classification first'
        loging_to_dashboard(msg=msg, status='info')
        msg = '-> Or all relevant papers have been processed already'
        loging_to_dashboard(msg=msg, status='info')
        msg = 'Exiting...'
        loging_to_dashboard(msg=msg, status='info')
        return

    training_documents_path = 'data/topic_modeling/training_documents.pickle'
    if exists(training_documents_path):
        training_documents = load_data(training_documents_path)
        training_uids = [uid for uid in documents.index.values if uid in training_documents]
        documents_to_process = documents.drop(training_uids)
    else:
        documents_to_process = documents

    embeddings_path = 'embeddings/test_embeddings.pickle'
    umap_model_path = 'data/bayesian_search/best_umap_model.pickle'
    hdbscan_model_path = 'data/bayesian_search/best_hdbscan_model.pickle'
    topic_modeling_summary_path = 'data/topic_modeling/topic_modeling_summary.xlsx'

    if (not exists(embeddings_path)) or (not exists(umap_model_path)) or (not exists(hdbscan_model_path)) or (
            not exists(topic_modeling_summary_path)):
        msg = 'Required files can not be found'
        loging_to_dashboard(msg=msg, status='info')
        msg = 'Run Topic Modeling first'
        loging_to_dashboard(msg=msg, status='info')
        return None

    msg = 'Loading pre-computed embeddings'
    loging_to_dashboard(msg=msg, status='info')
    embeddings = load_data(embeddings_path)
    uids_to_process = list(documents_to_process.index.values)
    embeddings = {k: embeddings[k] for k in uids_to_process}
    embeddings = np.array(list(embeddings.items()), dtype=object)
    embeddings = np.array([embeddings[i, 1] for i in range(len(embeddings))])

    msg = 'Loading UMAP model and transform embeddings'
    loging_to_dashboard(msg=msg, status='info')
    umap_model = load_data(umap_model_path)
    embeddings = umap_model.transform(embeddings)

    msg = 'Loading HDBSCAN model and calculate cluster membership probabilities'
    loging_to_dashboard(msg=msg, status='info')
    hdbscan_model = load_data(hdbscan_model_path)
    num_topics = len(set(hdbscan_model.labels_))
    probabilities = hdbscan.membership_vector(hdbscan_model, embeddings)
    ### Determine cluster assignment based on highest probability /// using a probability threshold
    predictions = [np.argmax(x) if np.max(x) > 1.0 / num_topics else -1
                   for x in probabilities]
    predictions = {uids_to_process[i]: p for i, p in enumerate(predictions)}

    msg = 'Starting to determine NCS pathway(s) and Co-benefit(s)'
    loging_to_dashboard(msg=msg, status='info')
    topic_modeling_summary = pd.read_excel(topic_modeling_summary_path, keep_default_na=False)
    topic_modeling_summary.set_index('Topic number', inplace=True)

    ### Use topic labels to determine NCS pathway and Co-benefits
    ### Upload predictions to elasticsearch
    msg = f'{documents.shape[0]} records to be updated'
    loging_to_dashboard(msg=msg, status='info')
    uids = list(documents.index.values)
    iter_num = int(documents.shape[0] / 100000) + 1
    for i in range(iter_num):

        msg = f'Prediction batch #{i+1} out of {iter_num} prediction batches'
        loging_to_dashboard(msg=msg, status='info')

        list_of_update_actions = list()

        for uid in uids[i * 100000:min(documents.shape[0], i * 100000 + 100000)]:

            if uid in training_uids:
                prediction = training_documents[uid]
            else:
                prediction = predictions[uid]

            if prediction == -1:
                pathways = ['N/A']
                benefits = ['N/A']
            else:
                pathways = str(
                    topic_modeling_summary.loc[prediction]['Relevant Pathway(s) (FILL THIS COLUMN)']).replace(' ',
                                                                                                              '').split(
                    ',')
                benefits = str(
                    topic_modeling_summary.loc[prediction]['Relevant Co-benefit(s) (FILL THIS COLUMN)']).replace(' ',
                                                                                                                 '').split(
                    ',')

            fields_to_update = dict()
            fields_to_update['predicted_pathway_numbers'] = pathways
            fields_to_update['predicted_benefits'] = benefits

            list_of_update_actions.append({'_id': uid,
                                           '_index': dataset_hash,
                                           '_source': {'doc': fields_to_update},
                                           '_op_type': 'update'})

        msg = f'Uploading bulk #{i+1}, records from {i*100000} to {min(documents.shape[0], i*100000+100000)}'
        loging_to_dashboard(msg=msg, status='info')
        helpers.bulk(es, update_data_generator(list_of_update_actions), chunk_size=400, request_timeout=180)

    msg = f'{documents.shape[0]} records are updated'
    loging_to_dashboard(msg=msg, status='info')

    del hdbscan_model
    del umap_model
    del embeddings
    del documents

    msg = 'Prediction is done'
    loging_to_dashboard(msg=msg, status='info')


def main():
    # logging.basicConfig(level=logging.DEBUG)
    uvicorn.run(app, host="0.0.0.0", port=int(3333), log_config=None)


if __name__ == "__main__":
    main()
