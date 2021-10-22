import pandas as pd 
from transformers import BertTokenizer
import os
import numpy as np 
import torch

def create_intermediate(dataframe, datatype='train'):
	intermediate = pd.DataFrame()
	tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
	# one-line function to add special tokens
	addspecialtokens = lambda string:f'[CLS] {string.lower()} [SEP]'
	# one-line function to tokenize the question
	wordstoberttokens = lambda string:tokenizer.tokenize(string)
	# one-line function for token2ids converting
	berttokenstoids = lambda tokens:tokenizer.convert_tokens_to_ids(tokens)
	# performing all of above functions 
	intermediate['token_matrix'] = dataframe.question.apply(addspecialtokens).apply(wordstoberttokens).apply(berttokenstoids)
	intermediate['Question'] = dataframe.question
	intermediate['tokenized_question'] = dataframe.question.apply(addspecialtokens).apply(wordstoberttokens)
	intermediate['Answer'] = dataframe.answer_mid
	intermediate['first_entity_ids'] = dataframe['entity'].apply(addspecialtokens).apply(wordstoberttokens).apply(berttokenstoids)
	intermediate['second_entity_ids'] = dataframe['entity'].apply(lambda string:'').apply(addspecialtokens).apply(wordstoberttokens).apply(berttokenstoids)
	intermediate['entity'] = dataframe['entity']
	intermediate['tokenized_question'] = dataframe.entity.apply(addspecialtokens).apply(wordstoberttokens)
	intermediate.to_excel(f"../data/freebase/{datatype}_intermediate.xlsx")

QUESTION_WORDS = ['what', 'which', 'where', 'when', 'why', 'who', 'how', 'whom']

def get_question_words_ids():
	'''
		converting question words to their counterpart ids!
	'''
	tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
	question_words_ids = tokenizer.convert_tokens_to_ids(QUESTION_WORDS)
	question_words_ids += [101, 102]
	return question_words_ids
	
def contains(small, big):
    for idx in range(len(big)-len(small)):
        if big[idx:idx+len(small)]==small:
            return [idx, idx+len(small)]
    return [-1, -1]

def get_relation(token_ids, entity_borders, question_words_ids):
	'''
	retrieving relation using a simple heuristic, not entity, not question word, this is the relation span!
	'''
	relation = token_ids[:entity_borders[0]]+token_ids[entity_borders[1]:]
	relation = [item for item in relation if item not in question_words_ids]
	answer = []
	for item in token_ids:
		if item in relation:
			answer.append(1)
		else:
			answer.append(0)
	return answer

def create_bertified_dataset( input_excel_dir = r'../data/freebase/',
							  output_pkl_dir = r'../bertified/freebase/',
							  datatype='train'):
	'''
	This is where all the magics have happened. 
	the function is designed to convert inter-mediate data to what BERT network can be fed by.
	'''
	dataframe = pd.read_excel(os.path.join(input_excel_dir, f'{datatype}_intermediate.xlsx'), engine='openpyxl')
	# retrieving maximum length sample
	maxlen = dataframe['token_matrix'].apply(lambda x:len(eval(x))).max()
	# print(maxlen)
	# converting the data into nd.numpyarray 
	token_mat = np.zeros((len(dataframe), maxlen), dtype="int32")
	for i, row in enumerate(dataframe['token_matrix'].to_list()):
		token_mat[i, :len(eval(row))] = eval(row)
	# adding labels for start|end of entites. 
	entity_borders = np.zeros((len(dataframe), 2), dtype='int32')
	for i, (bigger, ent1) in enumerate(zip(dataframe['token_matrix'].to_list(), 
										   dataframe['first_entity_ids'].to_list())):
		entity_borders[i]=contains(eval(ent1)[1:-1], eval(bigger))
		
	relation_borders = np.zeros((len(dataframe), maxlen), dtype='int32')
	question_words_ids = get_question_words_ids()
	for i, (token_array, ent_borders) in enumerate(zip(dataframe['token_matrix'].to_list(), 
											  		   entity_borders)):
		relation_borders[i, :len(eval(token_array))] = get_relation(eval(token_array), ent_borders, question_words_ids)
	dumb_samples = []
	
	for i, (tokens, relation, entity) in enumerate(zip(token_mat, relation_borders, entity_borders)):
		if sum(relation)==0 or entity[0]==entity[-1]:
			dumb_samples.append(i)
	dataframe['relation_span'] = relation_borders.tolist() 
	dumb_records = dataframe.iloc[dumb_samples, :]
	dumb_records.to_excel(os.path.join(input_excel_dir, 'dumb_records.xlsx'))
	useful_records = dataframe[~(dataframe.index.isin(dumb_samples))]
	useful_records.to_excel(f'../data/freebase/{datatype}_useful_records.xlsx')
	relation_borders = np.delete(relation_borders, dumb_samples, axis=0)
	entity_borders = np.delete(entity_borders, dumb_samples, axis=0)
	token_mat = np.delete(token_mat, dumb_samples, axis=0)
	# saving the results 
	with open(os.path.join(output_pkl_dir, f'{datatype}_tokenmat.npy'), 'wb') as f:
		np.save(f, token_mat)
	with open(os.path.join(output_pkl_dir, f'{datatype}_entities.npy'), 'wb') as f:
		np.save(f, entity_borders)
	with open(os.path.join(output_pkl_dir, f'{datatype}_relations.npy'), 'wb') as f:
		np.save(f, relation_borders)
  

def read_data(input_pkl_dir=r'../bertified/freebase/', datatype='train'):
	tokens = np.load(os.path.join(input_pkl_dir, f'{datatype}_tokenmat.npy'))
	relations = np.load(os.path.join(input_pkl_dir, f'{datatype}_relations.npy'))
	entities = np.load(os.path.join(input_pkl_dir, f'{datatype}_entities.npy'))
	labels = np.hstack((entities, relations))
	data = [torch.from_numpy(tokens).long(), torch.from_numpy(labels).long()]
	return data


if __name__=='__main__':
	columns = ['identifier', 'entity_mid', 'entity', 'relation_type', 'answer_mid', 'question', 'entity_label']
	train_df = pd.read_csv(
	    '../data/freebase/questions/train.txt',
	    sep='\t',
	    header=None, 
	    index_col=None
	            )
	train_df.columns = columns
	valid_df = pd.read_csv(
	    '../data/freebase/questions/valid.txt',
	    sep='\t',
	    header=None, 
	    index_col=None
	            )
	valid_df.columns = columns
	test_df = pd.read_csv(
	    '../data/freebase/questions/test.txt',
	    sep='\t',
	    header=None, 
	    index_col=None
	            )
	test_df.columns = columns


	create_intermediate(train_df, datatype='train')
	print('create_intermediate')
	create_bertified_dataset(datatype='train')
	print('create_bertified_dataset')
	create_intermediate(valid_df, datatype='valid')
	print('create_intermediate')
	create_bertified_dataset(datatype='valid')
	print('create_bertified_dataset')
	create_intermediate(test_df, datatype='test')
	print('create_intermediate')
	create_bertified_dataset(datatype='test')
	print('create_bertified_dataset')
	