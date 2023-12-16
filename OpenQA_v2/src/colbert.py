import colbert
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
import random
import pickle
import pandas as pd
from tqdm import tqdm
import sys


def www2fb(in_str):
    if in_str.startswith("www.freebase.com"):
        in_str = 'fb:%s' % (in_str.split('www.freebase.com/')[-1].replace('/', '.'))
    if in_str == 'fb:m.07s9rl0':
        in_str = 'fb:m.02822'
    if in_str == 'fb:m.0bb56b6':
        in_str = 'fb:m.0dn0r'
    # Manual Correction
    if in_str == 'fb:m.01g81dw':
        in_str = 'fb:m.01g_bfh'
    if in_str == 'fb:m.0y7q89y':
        in_str = 'fb:m.0wrt1c5'
    if in_str == 'fb:m.0b0w7':
        in_str = 'fb:m.0fq0s89'
    if in_str == 'fb:m.09rmm6y':
        in_str = 'fb:m.03cnrcc'
    if in_str == 'fb:m.0crsn60':
        in_str = 'fb:m.02pnlqy'
    if in_str == 'fb:m.04t1f8y':
        in_str = 'fb:m.04t1fjr'
    if in_str == 'fb:m.027z990':
        in_str = 'fb:m.0ghdhcb'
    if in_str == 'fb:m.02xhc2v':
        in_str = 'fb:m.084sq'
    if in_str == 'fb:m.02z8b2h':
        in_str = 'fb:m.033vn1'
    if in_str == 'fb:m.0w43mcj':
        in_str = 'fb:m.0m0qffc'
    if in_str == 'fb:m.07rqy':
        in_str = 'fb:m.0py_0'
    if in_str == 'fb:m.0y9s5rm':
        in_str = 'fb:m.0ybxl2g'
    if in_str == 'fb:m.037ltr7':
        in_str = 'fb:m.0qjx99s'
    return in_str



to_index = []
count = 0

def freebase2triple(fbpath):
    with open(fbpath, 'r') as f:
        for i, line in enumerate(f):
            items = line.strip().split("\t")
            if len(items) != 3:
                print("ERROR: line - {}".format(line))
                continue
            subject = www2fb(items[0])
            predicate = www2fb(items[1])
            object = www2fb(items[2])
            try:
                subject_list = mid2name[subject]
                temp_pred = predicate.replace('fb:', '').replace('.', ' ').replace('_', ' ')
                object_list = mid2name[object]
                plain_text = f"{random.choice(subject_list)} {temp_pred} {random.choice(object_list)}"
                to_index.append((plain_text, subject, predicate))
            except:
                count+=1
            if i%1000000==0:
                print(i)
    return to_index

def reverb2triple(rvpath):
    df = pd.read_csv(rvpath, sep='\t', header=None)
    reverb_columns_name = ['ExID', 'arg1', 'rel', 'arg2', 'narg1', 'nrel', 'narg2', 'csents', 'conf', 'urls']
    df.columns = reverb_columns_name
    df = df.dropna()
    df = df.drop_duplicates()
    df['text'] = df.apply(lambda row:' '.join([row['arg1'], row['rel'], row['arg2']]).strip(), axis=1)
    df['docno'] = (df.index).astype(str)
    df = df[['text', 'docno']]
    list_of_tuples = [tuple(row) for row in df.values]
    return list_of_tuples

def index(index_name, nbits, doc_maxlen, to_index):
    checkpoint = 'colbert-ir/colbertv2.0'

    with Run().context(RunConfig(nranks=1, experiment='notebook')):  # nranks specifies the number of GPUs to use
        config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits, kmeans_niters=4) # kmeans_niters specifies the number of iterations of k-means clustering; 4 is a good and fast default.
                                                                                    # Consider larger numbers for small datasets.

        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=[item[0] for item in to_index[:1000000]], overwrite=True)
        # indexer.get_index() # You can get the absolute path of the index, if needed.
    return indexer.get_index() # You can get the absolute path of the index, if needed.

def create_searcher(index_name):
    with Run().context(RunConfig(experiment='notebook')):
        searcher = Searcher(index=index_name, collection=[item[0] for item in to_index[:1000000]])
    return searcher

def retrieve(searcher, to_search):
    candidates = []
    answers = []
    for i, item in tqdm(enumerate(to_search), total=len(to_search)):
        temp = []
        results = searcher.search([item], k=100)

        
        for passage_id, passage_rank, passage_score in zip(*results):
            temp.append(' '.join(to_index[passage_id][1:]))
        candidates.append(temp)
        answer = to_search.iloc[i][['Question', 'Answer', 'relation_type']].to_dict()
        answers.append(' '.join((answer['Answer'], answer['relation_type'])))
    
    return candidates, answers

def calculate_hit_at_n(retrieved_lists, answer_lists):
    hit_at_n_results = {}

    for n in [1, 3, 5, 10, 20, 50, 100]:
        hits = 0

        for retrieved, answers in zip(retrieved_lists, answer_lists):
            top_n_retrieved = retrieved[:n]

            # Check if there is at least one relevant item in the top N retrieved
            if answers in top_n_retrieved:
                hits += 1

        hit_at_n = hits / len(retrieved_lists) * 100.0
        hit_at_n_results[f'Hit@{n}'] = hit_at_n

    return hit_at_n_results


if __name__=='__name__':
    
    knowledge_graph = int(sys.argv[1])
    dataset = int(sys.argv[2])
    count = int(sys.argv[3])

    # loading index related to freebase names 
    with open('/content/drive/MyDrive/indexes/names_2M.pkl', 'rb') as f:
        mid2name = pickle.load(f)

    fbpath = '/content/drive/MyDrive/QA/freebase-FB2M.txt'
    rvpath = '/content/drive/MyDrive/QA/reverb_wikipedia_tuples-1.1.txt'
    freebase = freebase2triple(fbpath, mid2name)
    reverb = reverb2triple(rvpath)

    fb_test_queries = pd.read_excel('/content/OpenQA/data/freebase/test_useful_records.xlsx')[['Question', 'Answer', 'relation_type']]
    fb_valid_queries = pd.read_excel('/content/OpenQA/data/freebase/valid_useful_records.xlsx')[['Question', 'Answer', 'relation_type']]
    rv_test_queries = pd.read_excel('/content/OpenQA/data/reverb/test.xlsx')[['Question', 'Reverb_no']]
    rv_valid_queries = pd.read_excel('/content/OpenQA/data/reverb/valid.xlsx')[['Question', 'Reverb_no']]

    if knowledge_graph==0: # reverb
        to_index = [item[0] for item in freebase]
        if dataset==0: # testset
            to_search = rv_test_queries
        if dataset==1: # validset
            to_search = rv_valid_queries
    if knowledge_graph==1: # freebase 
        to_index = [item[0] for item in reverb]
        if dataset==0: # testset
            to_search = fb_test_queries
        if dataset==1: # validset
            to_search = fb_valid_queries
    if knowledge_graph==0:
        to_index = [item[0] for item in freebase+reverb]
        if dataset==0: # testset
            to_search = pd.concat([fb_test_queries, rv_test_queries], ignore_index=True)
        if dataset==1: # validset
            to_search = pd.concat([fb_valid_queries, rv_valid_queries], ignore_index=True)
    if count>-1:
        to_index = to_index[:count]
        to_search = to_search[:count]


    nbits = 2   # encode each dimension with 2 bits
    doc_maxlen = 50 # truncate passages at 50 tokens
    index_name = f'index.{nbits}bits'

    index(index_name, nbits, doc_maxlen, to_index)
    searcher = create_searcher(index_name)
    retrieved_lists, answer_lists = retrieve(searcher, to_search)
    HITs = calculate_hit_at_n(retrieved_lists, answer_lists)
    print(HITs)




