# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 22:23:21 2020

@author: meti
"""

import pandas as pd
import numpy as np 
import os
from transformers import BertTokenizer
from itertools import chain, combinations
from sklearn.model_selection import train_test_split
from tqdm import tqdm


QUESTION_WORDS = ['what', 'which', 'where', 'when', 'why', 'who', 'how', 'whom']


def powerset(iterable):
	'''
	gets an iterable and create power set of it!
	'''
	power_set = chain.from_iterable(combinations(iterable, r) for r in range(len(iterable)+1))
	power_set = [list(item) for item in power_set]
	power_set = sorted(power_set, key=lambda x:len(x), reverse=True)
	power_set.remove([iterable[0], iterable[-1]])
	return power_set


def read_reverb(reverb_path):
	'''
	Function to read reverb file and return a list of striped lines!
	'''
	lines = []
	with open(reverb_path) as fin:
		for line in fin:
			temp = line.strip().split('\t')
			lines.append(temp)
	return lines

def get_triple(record_list, index):
	'''
	Gets records of the KB and returns the desired record triple
	'''
	temp = record_list[index]
	return (temp[1], temp[2], temp[3])

def get_normalized_triple(record_list, index):
	'''
	Gets records of the KB and returns the desired record normalized triple
	'''
	temp = record_list[index]
	return (temp[4], temp[5], temp[6])



def combine_with_reverb(questions_path=r'../data/reverb/questions/Final_Sheet_990824.xlsx',
				   reverb_path=r'../data/reverb/questions/reverb_wikipedia_tuples-1.1.txt'):
	# reading Questions file 
	dataframe = pd.read_excel(questions_path, engine='openpyxl')
	# reading KB into list of lines (string)
	reverb = read_reverb(reverb_path)
	# filtering strategy 
	dataframe = get_tuple_frequency(reverb, dataframe)
	dataframe = dataframe[(dataframe['Frequency']<10)&(dataframe.Meaningful==1)]
	# in this stage, we DO have the desired questions

	# loading up bert tokenizer
	tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
	# one-line function to add special tokens
	addspecialtokens = lambda string:f'[CLS] {string} [SEP]'
	# one-line function to tokenize the question
	wordstoberttokens = lambda string:tokenizer.tokenize(string)
	# one-line function for token2ids converting
	berttokenstoids = lambda tokens:tokenizer.convert_tokens_to_ids(tokens)
	# performing all of above functions 
	dataframe['token_matrix'] = dataframe.Question.apply(addspecialtokens).apply(wordstoberttokens).apply(berttokenstoids)
	
	# adding triple|normalized triple of the question
	dataframe['triple'] = dataframe.Reverb_no.apply(lambda x:get_triple(reverb, x))
	dataframe['normalized_triple'] = dataframe.Reverb_no.apply(lambda x:get_normalized_triple(reverb, x))
	# determining the max length 
	maxlen = dataframe['token_matrix'].apply(lambda x:len(x)).max()
	
	# converting first entity | second entity | relation into convenient bert sub-token IDs 
	dataframe['first_entity_ids'] = dataframe['triple'].apply(lambda x:x[0]).apply(addspecialtokens).apply(wordstoberttokens).apply(berttokenstoids)
	dataframe['second_entity_ids'] = dataframe['triple'].apply(lambda x:x[-1]).apply(addspecialtokens).apply(wordstoberttokens).apply(berttokenstoids)
	dataframe['relation_ids'] = dataframe['triple'].apply(lambda x:x[1]).apply(addspecialtokens).apply(wordstoberttokens).apply(berttokenstoids)
	# saving the outputs!
	dataframe.to_excel("../data/reverb/intermediate.xlsx")
	
def get_borders(bigger, smaller):
	power_set = powerset(smaller)
	for smaller_subset in power_set[:-1]:
		net_bigger = bigger[1:-1]; net_smaller = smaller_subset[1:-1]
		for i in range(1, len(net_bigger)-len(net_smaller)+1):
			if net_bigger[i:i+len(net_smaller)]==net_smaller:
				return [i+1, i+len(net_smaller)+1]
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

def get_question_words_ids():
	'''
		converting question words to their counterpart ids!
	'''
	tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
	question_words_ids = tokenizer.convert_tokens_to_ids(QUESTION_WORDS)
	question_words_ids += [101, 102]
	return question_words_ids
	



def create_bertified_dataset( input_excel_dir = r'../data/reverb/',
							  output_pkl_dir = r'../bertified/reverb/',
							  data_folder = r'data/reverb'):
	'''
	This is where all the magics have happened. 
	the function is designed to convert inter-mediate data to what BERT network can be fed by.
	'''
	dataframe = pd.read_excel(os.path.join(input_excel_dir, 'intermediate.xlsx'), engine='openpyxl')
	# retrieving maximum length sample
	maxlen = dataframe['token_matrix'].apply(lambda x:len(eval(x))).max()
	# converting the data into nd.numpyarray 
	token_mat = np.zeros((len(dataframe), maxlen), dtype="int32")
	for i, row in enumerate(dataframe['token_matrix'].to_list()):
		token_mat[i, :len(eval(row))] = eval(row)
	# adding labels for start|end of entites. 
	answer_entity = []
	entity_borders = np.zeros((len(dataframe), 2), dtype='int32')
	for i, (bigger, ent1, ent2) in enumerate(zip(dataframe['token_matrix'].to_list(), 
										   dataframe['first_entity_ids'].to_list(),
										   dataframe['second_entity_ids'].to_list())):
		temp1 = get_borders(eval(bigger), eval(ent1))
		temp2 = get_borders(eval(bigger), eval(ent2))
		if (temp1[-1]-temp1[0])>(temp2[-1]-temp2[0]):
			entity_borders[i] = temp1
			answer_entity.append(1)
		else:
			entity_borders[i] = temp2
			answer_entity.append(2)
	dataframe['answer_entity'] = answer_entity
	# adding labels for relation spans 
	relation_borders = np.zeros((len(dataframe), maxlen), dtype='int32')
	question_words_ids = get_question_words_ids()
	for i, (token_array, ent_borders) in enumerate(zip(dataframe['token_matrix'].to_list(), 
											  		   entity_borders)):
		relation_borders[i, :len(eval(token_array))] = get_relation(eval(token_array), ent_borders, question_words_ids)
	dumb_samples = []
	# filtering out dumb samples (samples with no / defected labels)
	for i, (tokens, relation, entity) in enumerate(zip(token_mat, relation_borders, entity_borders)):
		if sum(relation)==0 or entity[0]==entity[-1]:
			dumb_samples.append(i)
	dumb_records = dataframe.iloc[dumb_samples, :]
	dumb_records.to_excel(os.path.join(input_excel_dir, 'dumb_records.xlsx'))
	useful_records = dataframe[~(dataframe.index.isin(dumb_samples))]
	# splitting usefull records into train|valid|test
	train, test = train_test_split(useful_records, test_size=0.30, random_state=42)
	train, valid = train_test_split(train, test_size=0.15, random_state=42)
	train.to_excel(os.path.join(input_excel_dir, 'train.xlsx')); valid.to_excel(os.path.join(input_excel_dir, 'valid.xlsx')); test.to_excel(os.path.join(input_excel_dir, 'test.xlsx'))
	relation_borders = np.delete(relation_borders, dumb_samples, axis=0)
	entity_borders = np.delete(entity_borders, dumb_samples, axis=0)
	token_mat = np.delete(token_mat, dumb_samples, axis=0)
	# saving the results 
	with open(os.path.join(output_pkl_dir, 'tokenmat.npy'), 'wb') as f:
		np.save(f, token_mat)
	with open(os.path.join(output_pkl_dir, 'entities.npy'), 'wb') as f:
		np.save(f, entity_borders)
	with open(os.path.join(output_pkl_dir,'relations.npy'), 'wb') as f:
		np.save(f, relation_borders)
			
def read_data(token_path=r'../bertified/reverb/tokenmat.npy',	
			relation_path=r'../bertified/reverb/relations.npy', 
			entity_path=r'../bertified/reverb/entities.npy'):
	tokens = np.load(token_path)
	relations = np.load(relation_path)
	entities = np.load(entity_path)
	labels = np.hstack((entities, relations))
	X_train, X_test, y_train, y_test = train_test_split(tokens, labels, test_size=0.30, random_state=42)
	X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
	tselected = [torch.from_numpy(X_train).long(), torch.from_numpy(y_train[:, 2:]).long()]
	vselected = [torch.from_numpy(X_valid).long(), torch.from_numpy(y_valid[:, 2:]).long()]
	xselected = [torch.from_numpy(X_test).long(), torch.from_numpy(y_test[:, 2:]).long()]
	traindata = TensorDataset(*tselected)
	devdata = TensorDataset(*vselected)
	testdata = TensorDataset(*xselected)

	ret = (traindata, devdata, testdata)
	return ret

def get_tuple_frequency(dataset_lines, questions):
    # Performing Question filtering as described in Article
    index = {}
    for idx, line in tqdm(enumerate(dataset_lines), total=len(dataset_lines), desc='Indexing ...'):
        left = line[4]+'|'+line[5]
        right = line[5]+'|'+line[6]
        for item in [left, right]:
            if item in index:
                index[item]+=1
            else:
                index[item]=1
    frequencies = []
    for idx, row in tqdm(questions.iterrows(), total=questions.shape[0], desc='Filtering ...'):
        reverb_number = row['Reverb_no']
        left = dataset_lines[reverb_number][4]+'|'+dataset_lines[reverb_number][5]
        right = dataset_lines[reverb_number][5]+'|'+dataset_lines[reverb_number][6]
        frequency = max(index[left], index[right])
        frequencies.append(frequency)
    
    questions['Frequency']=frequencies
    return questions


if __name__ == '__main__':

	# reverb_lines = read_reverb('reverb_wikipedia_tuples-1.1.txt')
	# questions = pd.read_excel(r'data/Final_Sheet_990824.xlsx', sheet_name=1, engine='openpyxl')
	# index = get_tuple_frequency(reverb_lines, questions)
	# index[index['Frequency']<10].to_excel('')
	combine_with_reverb()

	create_bertified_dataset()
	# print(get_question_words_ids())
	# get_relation([101, 2129, 2172, 2769, 2001, 2139, 29510, 2005, 2604, 102], [8, 9], [2054, 2029, 2073, 2043, 2339, 2040, 2129, 3183])
