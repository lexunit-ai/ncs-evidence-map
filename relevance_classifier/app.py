import numpy as np
import pandas as pd
from os import listdir
import pickle
from os.path import exists
from elasticsearch import Elasticsearch, helpers
import eland as ed
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from datetime import datetime
import uvicorn
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
import requests

app = FastAPI()


class Params(BaseModel):
    threshold: Union[float, None] = None
    dataset_hash: Union[str, None] = '...'# TODO


class HashParams(BaseModel):
    dataset_hash: Union[str, None] = '...'# TODO


@app.get('/heartbeat')
def heartbeat():
    return 'OK'


@app.post('/read_specific')
def read_from_hash(params: HashParams):
    """Read the training data from hash info, 
    convert it into a single Pandas DataFrame,
    and then save it at its designated location."""

    dataset_hash = params.dataset_hash

    dirname = 'training_data/' + str(dataset_hash)
    combo_info = pd.read_csv('training_data_hashes/hashes-' + str(dataset_hash) + '.csv',
                             index_col=0)
    combo_info.set_index('string_hash', inplace=True)
    dfs_list = list()
    combo_folders = listdir(dirname)
    pathway_names = list()

    for combo_folder in combo_folders:

        combo_dfs = listdir('%s/%s' % (dirname, combo_folder))

        for combo_df in combo_dfs:
            df = pd.read_excel('%s/%s/%s' % (dirname, combo_folder, combo_df))[
                ['Article Title', 'Abstract', 'Author Keywords', 'Document Type', 'UT (Unique WOS ID)']]
            pathway_names.append(combo_info.loc[combo_folder]['pathway_number'])
            df['pathway_number'] = combo_info.loc[combo_folder]['pathway_number']
            df['benefit'] = combo_info.loc[combo_folder]['benefit']
            dfs_list.append(df[df['Abstract'].notnull()])

    df = pd.concat(dfs_list)
    df.set_index('UT (Unique WOS ID)', inplace=True)

    df = df[~df['Document Type'].isin(['Article; Early Access',
                                       'Article; Retracted Publication',
                                       'Book Review',
                                       'Correction',
                                       'Editorial Material',
                                       'Letter',
                                       'Meeting Abstract',
                                       'News Item',
                                       'Note',
                                       'Proceedings Paper; Retracted Publication',
                                       'Reprint',
                                       'Review',
                                       'Review; Book Chapter',
                                       'Review; Early Access'])]

    for i in set(pathway_names):
        pathway_df = df[df['pathway_number'] == i]
        pathway_df.to_excel('data/ground_truth/%s.xlsx' % (str(i)))


def read_training_data_from_ground_truth():
    """Read the ground truth Excel files from disk.
        These records will be used for training the relevance models."""

    dirname = 'data/ground_truth'
    excel_files = listdir(dirname)
    dfs_list = list()

    for gt in excel_files:
        dfs_list.append(pd.read_excel(dirname + '/' + gt, index_col=0))

    df = pd.concat(dfs_list)

    return df


def query_es_db(es, dataset_hash, new_threshold):
    """Query records from ES DB that does not have relevance value.
        Return them in a Pandas DataFrame."""

    df = ed.DataFrame(es_client=es, es_index_pattern=dataset_hash, columns=['abstract', 'relevance_value'])
    if not new_threshold:
        return df[df['relevance_value'] == -100].to_pandas().rename(columns={'abstract': 'Abstract'})
    else:
        return df.to_pandas().rename(columns={'abstract': 'Abstract'})


def create_embeddings_batch(model, df):
    """Calculate embeddings of abstract texts in 'df' using the given 'model' with batch_size = 32."""

    embeddings = model.encode(df['Abstract'].values, show_progress_bar=True)
    embeddings_dict = dict()

    for i in range(df.shape[0]):
        embeddings_dict[df.iloc[i].name] = embeddings[i]

    return embeddings_dict


def create_embeddings_single(model, df):
    """Calculate embeddings of abstract texts in 'df' using the given 'model' with batch_size = 1."""

    embeddings_dict = dict()

    for i in range(df.shape[0]):
        embeddings_dict[df.iloc[i].name] = model.encode(df.iloc[i]['Abstract'])

    return embeddings_dict


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


def update_embeddings(df, embedding_model, load_flag, filename):
    """Check if there are existing embeddings on disk.
        Create embeddings for only new uids."""

    embeddings_folder = 'data/embeddings/'

    if load_flag and exists(embeddings_folder + filename):
        embeddings = load_data(embeddings_folder + filename)
        new_uids = list(set(df.index.values).difference(set(embeddings.keys())))
        if not len(new_uids): return embeddings
    else:
        embeddings = dict()
        new_uids = list(df.index.values)

    new_embeddings = create_embeddings_batch(embedding_model, df.loc[new_uids])

    for uid in new_uids:
        embeddings[uid] = new_embeddings[uid]

    save_data(embeddings, embeddings_folder + filename)
    return embeddings


def aggregate_combo_dictionary(combo_dict):
    """Aggregate along benefits in a combo dictionary."""

    aggregated_dict = dict()

    for k, v in combo_dict.items():

        agg_k = str(k)[:-1]
        if agg_k not in aggregated_dict: aggregated_dict[agg_k] = list()
        aggregated_dict[agg_k].append(v)

    return aggregated_dict


def build_embedding_based_models(df, embeddings):
    """Build baseline and weighted models for each NCS pathway 
        based on the mean of the corresponding embeddings."""

    pathway_models_weighted = dict()
    pathway_models = dict()

    for i in range(df.shape[0]):

        path = str(df.iloc[i]['pathway_number'])
        benefit = str(df.iloc[i]['benefit'])
        key = path + benefit
        uid = df.iloc[i].name

        if path in pathway_models:
            pathway_models[path].append(embeddings[uid])
        else:
            pathway_models[path] = [embeddings[uid]]

        if key in pathway_models_weighted:
            pathway_models_weighted[key].append(embeddings[uid])
        else:
            pathway_models_weighted[key] = [embeddings[uid]]

    pathway_models.update((k, np.array(v).mean(0)) for k, v in pathway_models.items())
    pathway_models_weighted.update((k, np.array(v).mean(0)) for k, v in pathway_models_weighted.items())
    pathway_models_weighted = aggregate_combo_dictionary(pathway_models_weighted)
    pathway_models_weighted.update((k, np.array(v).mean(0)) for k, v in pathway_models_weighted.items())

    return pathway_models, pathway_models_weighted


def get_pathway_similarities(embedding, pathway_models, similarity_type='cos'):
    """Calculate the similarity to each model and return in a dictionary."""

    sims = dict()

    for k, v in pathway_models.items():
        if similarity_type == 'cos':
            sims[k] = cosine_similarity(embedding.reshape(1, -1), v.reshape(1, -1))
        elif similarity_type == 'euc':
            sims[k] = np.array([[1 / (1 + np.linalg.norm(embedding.reshape(1, -1) - v.reshape(1, -1)))]])
        elif similarity_type == 'rbf':
            sigma = 1
            sims[k] = np.array(
                [[np.exp(-np.linalg.norm(embedding.reshape(1, -1) - v.reshape(1, -1)) / (2 * sigma * sigma))]])
        else:
            print('Not supported similarity function!')

    return sims


def get_max_similarity(sims):
    """Find the maximum similarity in a dictionary and 
        return the corresponding key:value pair (mvk:mk)"""

    mvk = max(sims, key=sims.get)
    mv = sims[mvk]

    return mvk, mv


def get_similarity_info(embeddings, pathway_models, similarity_type='cos'):
    """Calculate the similarities between each text and each model.
        Also finds the most similar pathway and its number.
        Return everything in a dictionary of tuples."""

    similarity_info = dict()

    for k, v in embeddings.items():
        sims = get_pathway_similarities(v, pathway_models, similarity_type)
        mvk, mv = get_max_similarity(sims)

        similarity_info[k] = (mvk, mv)

    return similarity_info


def get_pathway_maxsims(df, similarity_info):
    """Collect similarity values of most similar articles per pathway."""

    combo_maxsims = dict()

    for i in range(df.shape[0]):

        path = str(df.iloc[i]['pathway_number'])
        benefit = str(df.iloc[i]['benefit'])
        key = path + benefit
        uid = df.iloc[i].name

        if similarity_info[uid][0] == path:
            if key not in combo_maxsims:
                combo_maxsims[key] = list()
            combo_maxsims[key].append(similarity_info[uid][1])

    return aggregate_combo_dictionary(combo_maxsims)


def get_thresholds(pathway_maxsims, pathway_levels, user_defined_threshold):
    """Calculate the thresholds for each pathway and return in a dictionary."""

    pathway_thresholds = dict()

    for k, v in pathway_maxsims.items():
        p = pathway_levels[k] if not user_defined_threshold else pathway_levels
        flat_list = [x for xs in v for x in xs]
        th_index = int(len(flat_list) * p / 100) if p != 0 else 0
        pathway_thresholds[k] = np.sort(np.array(flat_list).reshape(1, -1)[0])[th_index]

        ### Old method (finding threshold first on benefit, then on pathway level) ###
        # pathway_thresholds[k] = list()
        # for c in v:
        #     if len(c)>0:
        #         th_index = int(len(c)*pathway_levels[k]/100) if pathway_levels[k]!=0 else 0
        #         pathway_thresholds[k].append(np.sort(np.array(c).reshape(1,-1)[0])[th_index])
    # pathway_thresholds.update((kk, np.sort(np.array(vv).reshape(1,-1)[0])[1]) for kk,vv in pathway_thresholds.items())

    return pathway_thresholds


def optimize_threshold(maxsims):
    """Run the optimization to get pathway specific threshold level."""

    flat_list = [x for xs in maxsims for x in xs]
    sorted_maxsims = np.sort(np.array(flat_list).reshape(1, -1)[0])
    sim_interval = sorted_maxsims[-1] - sorted_maxsims[0]

    sim_dist = list()
    for i in range(sorted_maxsims.shape[0] - 1):
        sim_dist.append(100 * (sorted_maxsims[i + 1] - sorted_maxsims[i]) / sim_interval)
    sim_dist = np.array(sim_dist)

    percentaged_sim_dist = list()
    window_size = round(len(sim_dist) / 100)
    for i in range(0, len(sim_dist) - window_size, window_size):
        percentaged_sim_dist.append(sim_dist[i:i + window_size].sum())

    window_size = 5
    start_checking = False
    for i in range(len(percentaged_sim_dist) - window_size):
        if 1 > percentaged_sim_dist[i]:
            start_checking = True
        if start_checking:
            traveled_dist = np.array(percentaged_sim_dist[i:i + window_size]).sum()
            if window_size > traveled_dist:
                return i + 1
    return 10  # 10 is the default threshold level


def get_pathway_levels(pathway_maxsims):
    """Running the optimization for each pathway."""

    pathway_levels = dict()

    for k, v in pathway_maxsims.items():
        pathway_levels[k] = optimize_threshold(v)

    return pathway_levels


def test_relevance(test_embedding, pathway_models, pathway_thresholds, similarity_type='cos'):
    """Determine if the given text is relevant or not.
        Return the most similar pathway (mvk) 
        and the corresponding similarity value (mv)."""

    sims = get_pathway_similarities(test_embedding, pathway_models, similarity_type)
    mvk, mv = get_max_similarity(sims)

    return mvk, mv, mv >= pathway_thresholds[mvk]


def check_threshold(threshold, user_defined_threshold):
    """Determine if the given threshold is different from the previous one."""

    threshold_path = 'data/relevance_models/threshold.pickle'

    if exists(threshold_path):
        previous_data = load_data(threshold_path)
        previous_threshold = previous_data[0]
        previous_user_defined_threshold = previous_data[1]
        if (not user_defined_threshold) and (not previous_user_defined_threshold):
            return False
        elif (not user_defined_threshold) and (previous_user_defined_threshold):
            return True
        elif (user_defined_threshold) and (threshold == previous_threshold):
            return False
        elif (user_defined_threshold) and (threshold != previous_threshold):
            return True
        else:
            return True
    else:
        return True


def get_relevance_models(embedding_model,
                         load_flag,
                         embeddings_filename,
                         threshold,
                         user_defined_threshold,
                         similarity_type):
    """Load training data and build the models."""

    relevance_models_path = 'data/relevance_models/relevance_models.pickle'

    if exists(relevance_models_path):
        msg = 'Loading relevance models'
        loging_to_dashboard(msg=msg, status='info')
        relevance_models = load_data(relevance_models_path)
        pathway_models = relevance_models['pathway_models']
        if user_defined_threshold:
            msg = 'Calculating thresholds based on user defined threshold level.'
            loging_to_dashboard(msg=msg, status='info')
            pathway_thresholds = get_thresholds(relevance_models['pathway_maxsims'],
                                                threshold,
                                                user_defined_threshold)
        else:
            msg = 'Loading thresholds from optimization'
            loging_to_dashboard(msg=msg, status='info')
            pathway_thresholds = relevance_models['pathway_thresholds']
    else:
        msg = 'Loading ground truth data'
        loging_to_dashboard(msg=msg, status='info')
        df = read_training_data_from_ground_truth()
        msg = 'Calculating embeddings'
        loging_to_dashboard(msg=msg, status='info')
        embeddings = update_embeddings(df, embedding_model, load_flag, embeddings_filename)
        msg = 'Building relevance models'
        loging_to_dashboard(msg=msg, status='info')
        pathway_models, _ = build_embedding_based_models(df, embeddings)  # Frequency weights by default
        similarity_info = get_similarity_info(embeddings, pathway_models, similarity_type)
        pathway_maxsims = get_pathway_maxsims(df, similarity_info)
        msg = 'Running optimization to get thresholds'
        loging_to_dashboard(msg=msg, status='info')
        pathway_levels = get_pathway_levels(pathway_maxsims)
        pathway_thresholds = get_thresholds(pathway_maxsims, pathway_levels, False)

        relevance_models = dict()
        relevance_models['pathway_models'] = pathway_models
        relevance_models['pathway_thresholds'] = pathway_thresholds
        relevance_models['pathway_maxsims'] = pathway_maxsims
        save_data(relevance_models, relevance_models_path)

        if user_defined_threshold:
            msg = 'Calculating thresholds based on user defined threshold level'
            loging_to_dashboard(msg=msg, status='info')
            pathway_thresholds = get_thresholds(pathway_maxsims, threshold, user_defined_threshold)

    return pathway_models, pathway_thresholds


def update_data_generator(list_of_update_actions):
    """Generator function that yields next update action upon invoking."""

    for update_action in list_of_update_actions:
        yield update_action


def test_and_upload(df,
                    es,
                    dataset_hash,
                    embedding_model,
                    test_load_flag,
                    test_embeddings_filename,
                    pathway_models,
                    pathway_thresholds,
                    similarity_type):
    """Load test data (records from ES DB without relevance value) 
        and determine if they are relevant or not."""

    msg = 'Calculate / read embeddings'
    loging_to_dashboard(msg=msg, status='info')
    embeddings = update_embeddings(df, embedding_model, test_load_flag, test_embeddings_filename)

    uids = list(df.index.values)
    iter_num = int(df.shape[0] / 100000) + 1
    for i in range(iter_num):

        msg = f'Running relevance classification batch #{i+1} out of {iter_num} batches'
        loging_to_dashboard(msg=msg, status='info')

        list_of_update_actions = list()

        for uid in uids[i * 100000:min(df.shape[0], i * 100000 + 100000)]:
            relevant_pathway, relevance_value, is_relevant = test_relevance(embeddings[uid],
                                                                            pathway_models,
                                                                            pathway_thresholds,
                                                                            similarity_type)

            fields_to_update = dict()
            fields_to_update['is_relevant'] = is_relevant
            fields_to_update['relevance_value'] = relevance_value
            fields_to_update['relevant_pathway'] = relevant_pathway

            list_of_update_actions.append({'_id': uid,
                                           '_index': dataset_hash,
                                           '_source': {'doc': fields_to_update},
                                           '_op_type': 'update'})

        msg = f'Uploading bulk #{i+1}, records from {i*100000} to {min(df.shape[0], i*100000+100000)}'
        loging_to_dashboard(msg=msg, status='info')
        helpers.bulk(es, update_data_generator(list_of_update_actions), chunk_size=400, request_timeout=180)


@app.post("/ping")
def run_rc(params: Params):
    msg = 'Start: RELEVANCE CLASSIFICATION'
    loging_to_dashboard(msg=msg, status='info')

    dataset_hash = params.dataset_hash

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

    train_embeddings_filename = 'train_embeddings.pickle'
    test_embeddings_filename = 'test_embeddings.pickle'
    train_load_flag = True
    test_load_flag = True
    similarity_type = 'cos'
    try:
        threshold = int(params.threshold)
        user_defined_threshold = True
    except:
        threshold = params.threshold
        user_defined_threshold = False
    new_threshold = check_threshold(threshold, user_defined_threshold)

    msg = 'Querying data from elastic database'
    loging_to_dashboard(msg=msg, status='info')
    df = query_es_db(es, dataset_hash, new_threshold)
    if df.shape[0]:
        msg = f'{df.shape[0]} records were queried from Elasticsearch database'
        loging_to_dashboard(msg=msg, status='info')
    else:
        msg = 'All relevant papers have been processed already'
        loging_to_dashboard(msg=msg, status='info')
        msg = 'Exiting...'
        loging_to_dashboard(msg=msg, status='info')
        return

    msg = 'Loading BERT model'
    loging_to_dashboard(msg=msg, status='info')
    embedding_model_type = 'all-mpnet-base-v1'
    embedding_model = SentenceTransformer(embedding_model_type)

    pathway_models, pathway_thresholds = get_relevance_models(embedding_model,
                                                              train_load_flag,
                                                              train_embeddings_filename,
                                                              threshold,
                                                              user_defined_threshold,
                                                              similarity_type)

    test_and_upload(df,
                    es,
                    dataset_hash,
                    embedding_model,
                    test_load_flag,
                    test_embeddings_filename,
                    pathway_models,
                    pathway_thresholds,
                    similarity_type)

    if new_threshold:
        save_data((threshold, user_defined_threshold), 'data/relevance_models/threshold.pickle')

    msg = f'{df.shape[0]} records were updated'
    loging_to_dashboard(msg=msg, status='info')

    del embedding_model
    del df

    msg = 'Relevance classification is done'
    loging_to_dashboard(msg=msg, status='info')


def main():
    # logging.basicConfig(level=logging.DEBUG)
    uvicorn.run(app, host="0.0.0.0", port=int(3333), log_config=None)


if __name__ == "__main__":
    main()
