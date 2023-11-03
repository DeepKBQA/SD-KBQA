import pandas as pd
import networkx as nx
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pattern.en import conjugate, lemma, lexeme,PRESENT,SG,PAST
from transformers import Trainer, TrainingArguments
from transformers import AutoConfig, AutoTokenizer, AutoModel, BertForSequenceClassification
import sys
from sklearn.preprocessing import LabelEncoder
import numpy as np

def get_tf_idf_query_similarity(vectorizer, docs_tfidf, query):
    """
    vectorizer: TfIdfVectorizer model
    docs_tfidf: tfidf vectors for all docs
    query: query doc
    return: cosine similarity between query and all docs
    """
    query_tfidf = vectorizer.transform([query])
    cosineSimilarities = cosine_similarity(query_tfidf, docs_tfidf).flatten()
    return cosineSimilarities

#### pattern python>=3.7 compatibility problem
def pattern_stopiteration_workaround():
    try:
        print(lexeme('gave'))
    except:
        pass
# pattern_stopiteration_workaround()

class ReverbKnowledgeBase:
	def __init__(self, path='../data/reverb_wikipedia_tuples-1.1.txt'):
		super().__init__()
		df = pd.read_csv(path, sep='\t', header=None)
		reverb_columns_name = ['ExID', 'arg1', 'rel', 'arg2', 'narg1', 'nrel', 'narg2', 'csents', 'conf', 'urls']
		df.columns = reverb_columns_name
		df = df.dropna()
		df = df.drop_duplicates()
		self.KB = df
		self.is_facts = self.KB[(self.KB.rel.apply(lambda rg:rg.find('is ')!=-1))|(self.KB.rel.apply(lambda rg:rg.find('Is ')!=-1))]
		self.nodes = self.KB['arg1'].to_list()+self.KB['arg2'].to_list()
		self.edges = self.KB['rel'].to_list()
		self.nodes_vectorizer = TfidfVectorizer()
		self.edges_vectorizer = TfidfVectorizer()
		self.nodes_tfidf = self.nodes_vectorizer.fit_transform(self.nodes)
		self.edges_tfidf = self.edges_vectorizer.fit_transform(self.edges)
		self.relations = {}
		for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Indexing ...'):
			if row['rel'] in self.relations:
				self.relations[row['rel']].append((row['arg1'], index, row['conf']))
				self.relations[row['rel']].append((row['arg2'], index, row['conf']))
			else:
				self.relations[row['rel']] = [(row['arg1'], index, row['conf'])]
				self.relations[row['rel']].append((row['arg2'], index, row['conf']))
		


	def tfidf_nodes_query(self, search_phrase, cutoff=50):
		similarities = get_tf_idf_query_similarity(self.nodes_vectorizer, self.nodes_tfidf, search_phrase)
		ranks = {k:v for k,v in zip(self.nodes, similarities)}
		sorted_ranks = {k: v for k, v in sorted(ranks.items(), key=lambda item:item[1], reverse=True)[:min(len(ranks), cutoff)]}

		return sorted_ranks

	def tfidf_edges_query(self, search_phrase, cutoff=50):
		similarities = get_tf_idf_query_similarity(self.edges_vectorizer, self.edges_tfidf, search_phrase)
		ranks = {k:v for k,v in zip(self.edges, similarities)}
		sorted_ranks = {k: v for k, v in sorted(ranks.items(), key=lambda item:item[1], reverse=True)[:min(len(ranks), cutoff)]}
		return sorted_ranks
		
	def tfidf_query(self, node='Bill Gates', edge='Born'):
		edge_list = edge.split()
		if len(edge_list)>=2 and edge_list[0]=='did':
			edge_list[1] = conjugate(verb=edge_list[1],tense=PAST)
			edge = ' '.join(edge_list[1:])
		else:
			edge = ' '.join(edge_list)
		edges = self.tfidf_edges_query(edge)
		return edges

# convert raw text file to proper dataset object (based on task)
import torch
class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        # initialization
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        # slicing method X[index]
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

if __name__=='__main__':
  dataset = str(sys.argv[1])
  le = LabelEncoder()
  le.classes_ = np.load('/content/drive/MyDrive/data_freebase/classes_v2.npy')
  config = AutoConfig.from_pretrained("/content/drive/MyDrive/data_freebase/classifier_v2")
  tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
  model = BertForSequenceClassification.from_pretrained("/content/drive/MyDrive/data_freebase/classifier_v2", num_labels=len(le.classes_))
  test_texts = pd.read_excel(f'/content/drive/MyDrive/data_freebase/{dataset}_sbs.xlsx')['Question'].to_list()
  test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=200)
  test_dataset = ClassificationDataset(test_encodings, [1 for _ in test_texts])
  training_args = TrainingArguments(
    output_dir='./results',          # output directory
    per_device_train_batch_size=128,  # batch size per device during training
    per_device_eval_batch_size=128,   # batch size for evaluation
  )
  trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
  )
  pred = trainer.predict(test_dataset)
  freebase = []
  for idx, item in enumerate(pred.predictions):
    temp = softmax(item)
    indices = temp.argsort()
    fb = [(le.inverse_transform([indices[-i]])[0], temp[indices[-i]]) for i in range(10)]
    freebase.append(sorted(fb, key=lambda item:item[-1], reverse=True))

  debug = pd.DataFrame({
                      'Question':test_texts,
                      'Freebase':freebase,
                     })
  pattern_stopiteration_workaround()
  RKBG = ReverbKnowledgeBase(r'/content/drive/MyDrive/data_freebase/reverb_wikipedia_tuples-1.1.txt') #	'./sample_reverb_tuples.txt'
  test_df = pd.read_excel(f'/content/drive/MyDrive/data_freebase/{dataset}_sbs.xlsx')
  reverb = []
  for idx, row in tqdm(test_df.iterrows(), total=test_df.shape[0], desc='Predicting ...'):
    # pass
    reverb.append(RKBG.tfidf_query(node=row['node'], edge=row['edges']))
  debug['Reverb'] = reverb
  debug.to_excel(f'/content/drive/MyDrive/data_freebase/{dataset}_dbg.xlsx', index=False)
