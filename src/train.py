from transformers import BertModel, BertTokenizer
from transformers import AdamW
import torch
from torch.nn.functional import nll_loss
from torch.utils.data import Dataset, DataLoader
from utils import freebase_read_data, reverb_read_data, nodes_get_f1, edges_get_f1
from torch.nn.functional import mse_loss
from torch.nn.functional import one_hot
import numpy as np 
np.seterr(divide='ignore', invalid='ignore')
from tabulate import tabulate
from tqdm import tqdm
import sys



import logging
logging.basicConfig(level=logging.DEBUG)



class NodeEdgeDetector(torch.nn.Module):
	'''
	Neural Network architecture!
	'''
	def __init__(self, bert, tokenizer, dropout=0.5, clip_len=True, **kw):
		super().__init__(**kw)
		self.bert = bert
		dim = self.bert.config.hidden_size
		self.nodestart = torch.nn.Linear(dim, 1)
		self.nodeend = torch.nn.Linear(dim, 1)
		
		# self.edgestart = torch.nn.Linear(dim, 1)
		# self.edgeend = torch.nn.Linear(dim, 1)
		self.edgespan = torch.nn.Linear(dim, 1)
		
		self.dropout = torch.nn.Dropout(p=dropout)
		self.clip_len = clip_len

		self.tokenizer = tokenizer

	def forward(self, x):	   # x: (batsize, seqlen) ints
		mask = (x != 0).long()
		if self.clip_len:
			maxlen = mask.sum(1).max().item()
			maxlen = min(x.size(1), maxlen + 1)
			mask = mask[:, :maxlen]
			x = x[:, :maxlen]
		bert_outputs = self.bert(x, attention_mask=mask, output_hidden_states=False)
		lhs = bert_outputs.last_hidden_state
		a = self.dropout(lhs)
		logits_node_start = self.nodestart(lhs)
		logits_node_end = self.nodeend(lhs)
		logits_edge_span = self.edgespan(lhs)
		# logits_edge_start = self.edgestart(lhs)
		# logits_edge_end = self.edgeend(lhs)
		# print(logits_node_start.size(), logits_node_end.size(), logits_edge_span.size())
		logits = torch.cat([logits_node_start.transpose(1, 2), logits_node_end.transpose(1, 2), 
							logits_edge_span.transpose(1, 2)], 1)
		return logits


class BordersDataset(Dataset):
	'''
		Convert Data to proper Tensor dataset
	'''
	def __init__(self, data):
		# convert into tensors
		self.tokens_matrix = torch.from_numpy(data[0]).long()
		self.borders = torch.from_numpy(data[1]).long()
		self.n_samples = data[0].shape[0]

	def __getitem__(self, index):
		# returns specific item
		return self.tokens_matrix[index], self.borders[index] 
	def __len__(self):
		return self.n_samples
		# returns dataset length



class TrainingLoop:
	'''
	Everything related to model training
	'''
	def __init__(self, model, optimizer, freezeemb=True, 
				 epochs=6, save_path='./models/', **kw):
		self.model = model
		params = []
		for paramname, param in self.model.named_parameters():
			if paramname.startswith("bert.embeddings.word_embeddings"):
				if not freezeemb:
					params.append(param)
			else:
				params.append(param)
		self.optimizer = optimizer(params, **kw)
		self.epochs = epochs
		self.save_path = save_path
		self.predicts = None
	def train(self, dataloader, eval_dataloader, loss_function):
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.model.to(device)
		
		for epoch in range(self.epochs):
			self.model.train()
			losses = []

			for _, batch in enumerate(tqdm(dataloader, desc=f"Train Epoch Number {epoch+1}")):
				self.model.zero_grad()
				X, y = batch
				X = X.to(device); y = y.to(device)
				logits = self.model(X) 
				nodes_onehot = one_hot(y[:, :2], num_classes=logits.size()[-1]).float()
				
				maxlen = logits.size()[-1]
				actual = torch.cat((nodes_onehot, torch.unsqueeze(y[:, 2:], 1)[:, :, :maxlen]), 1)
				

				loss = loss_function(logits, actual, reduction='sum')
				losses.append(loss)
				loss.backward()
				self.optimizer.step()
			logging.info(f'Epoch number: {epoch+1} Train Loss is equal: {sum(losses)/len(losses)}') 
			self.eval(eval_dataloader, loss_function, epoch, device)


	def eval(self, dataloader, loss_function, epoch, device):
		self.model.eval()
		losses = []
		for _, batch in enumerate(tqdm(dataloader, desc=f"Eval Epoch Number {epoch+1}")):
			with torch.no_grad():
				X, y = batch
				X = X.to(device); y = y.to(device)
				logits = self.model(X) 
				nodes_onehot = one_hot(y[:, :2], num_classes=logits.size()[-1]).float()
				
				maxlen = logits.size()[-1]
				actual = torch.cat((nodes_onehot, torch.unsqueeze(y[:, 2:], 1)[:, :, :maxlen]), 1)
				

				loss = loss_function(logits, actual, reduction='sum')
				losses.append(loss)
		logging.info(f'Epoch number: {epoch+1} Eval Loss is equal: {sum(losses)/len(losses)}')

	def predict(self, dataloader, device, evaluate=True):
		self.model.eval()
		predicts = []
		for _, batch in enumerate(tqdm(dataloader, desc=f"Predicting ...")):
			with torch.no_grad():
				X, _ = batch
				X = X.to(device)
				logits = self.model(X)
				nodes_borders = torch.argmax(logits[:, :2], dim=2).cpu().detach().numpy().tolist()
				edges_spans = np.where(logits[:, 2].cpu().detach().numpy() > 0.5, 1, 0)
				
				[predicts.append((node_borders, edge_spans)) for node_borders, edge_spans in zip(nodes_borders, edges_spans)]
		self.predicts = predicts
		if evaluate:
			node_goldens = []
			for batch in dataloader:
				_, y = batch
				nodes_borders = y[:, :2].cpu().detach().numpy().tolist()
				[node_goldens.append(item) for item in nodes_borders]
			gold_nodes_border = np.array(node_goldens)
			pred_nodes_border = np.array([item[0] for item in self.predicts])
			nodes_get_f1(pred_nodes_border, gold_nodes_border)
			
			edge_goldens = []
			for batch in dataloader:
				_, y = batch
				edges_spans = y[:, 2:].cpu().detach().numpy().tolist()
				[edge_goldens.append(item) for item in edges_spans]
			gold_edges_span = np.array(edge_goldens)
			pred_edges_span = [item[1].tolist()+[0 for _ in range(35-len(item[1]))] for item in self.predicts]
			pred_edges_span = np.asarray(pred_edges_span)
			# print(gold_edges_span.shape, pred_edges_span.shape)
			edges_get_f1(pred_edges_span, gold_edges_span)

	
	def save(self, save_path='../models/node_edge_bert.pt'):
		torch.save(self.model, save_path)
	
	def load(self, save_path='../models/node_edge_bert.pt'):
		self.model = torch.load(save_path)

	def readable_predict(self, device, _input='Where was Bill Gates Born?', print_result=True):
		addspecialtokens = lambda string:f'[CLS] {string} [SEP]'
		wordstoberttokens = lambda string:self.model.tokenizer.tokenize(string)
		berttokenstoids = lambda tokens:self.model.tokenizer.convert_tokens_to_ids(tokens)
		input_token_ids = berttokenstoids(wordstoberttokens(addspecialtokens(_input)))
		input_tensors = torch.tensor([input_token_ids]).long()
		input_tensors = input_tensors.to(device)
		with torch.no_grad():
			logits = self.model(input_tensors)
		nodes_borders = torch.argmax(logits[:, :2], dim=2).cpu().detach().numpy().tolist()
		edges_spans = np.where(logits[:, 2].cpu().detach().numpy() > 0.5, 1, 0)

		node = self.model.tokenizer.convert_ids_to_tokens(input_token_ids[nodes_borders[0][0]:nodes_borders[0][1]])
		edge = self.model.tokenizer.convert_ids_to_tokens(np.array(input_token_ids)[edges_spans[0]==1])
		if print_result:
			data = [[_input, node, edge]]
			print(tabulate(data, headers=["Question", "Node", "Edge"]))
		else:
			return node, edge
	def readable_predict_article(self, device, _input='Where was Bill Gates Born?', print_result=True):
    ### getting node and edge
		addspecialtokens = lambda string:f'[CLS] {string} [SEP]'
		wordstoberttokens = lambda string:self.model.tokenizer.tokenize(string)
		berttokenstoids = lambda tokens:self.model.tokenizer.convert_tokens_to_ids(tokens)
		input_token_ids = berttokenstoids(wordstoberttokens(addspecialtokens(_input)))
		input_tensors = torch.tensor([input_token_ids]).long()
		input_tensors = input_tensors.to(device)
		with torch.no_grad():
			logits = self.model(input_tensors)
		nodes_borders = torch.argmax(logits[:, :2], dim=2).cpu().detach().numpy().tolist()
		edges_spans = np.where(logits[:, 2].cpu().detach().numpy() > 0.5, 1, 0)

		node = self.model.tokenizer.convert_ids_to_tokens(input_token_ids[nodes_borders[0][0]:nodes_borders[0][1]])
		edge = self.model.tokenizer.convert_ids_to_tokens(np.array(input_token_ids)[edges_spans[0]==1])
		if print_result:
			data = [[_input, node, edge]]
			print(tabulate(data, headers=["Question", "Node", "Edge"]))
		else:
			return wordstoberttokens, berttokenstoids, input_token_ids, nodes_borders, edges_spans, node, edge

if __name__=='__main__':

	train_reverb, valid_reverb, test_reverb = reverb_read_data()
	train_freebase = freebase_read_data(datatype='train')
	valid_freebase = freebase_read_data(datatype='valid')
	test_freebase = freebase_read_data(datatype='test')
	# print(train_reverb[0].shape, train_freebase[0].shape)
	train_offset = train_freebase[0].shape[1]-train_reverb[0].shape[1]

	# print(train_offset)
	train = (np.vstack(	
						(np.hstack((train_reverb[0], np.zeros(shape=(	train_reverb[0].shape[0],
																		train_offset	)))),
												train_freebase[0])),
			np.vstack(	
						(np.hstack((train_reverb[1], np.zeros(shape=(	train_reverb[0].shape[0],
																		train_offset	)))),
												train_freebase[1]))
			)
	valid_offset = valid_freebase[0].shape[1]-valid_reverb[0].shape[1]
	# print(valid_offset)
	valid = (np.vstack(	
						(np.hstack((valid_reverb[0], np.zeros(shape=(	valid_reverb[0].shape[0],
																		valid_offset	)))),
												valid_freebase[0])),
			np.vstack(	
						(np.hstack((valid_reverb[1], np.zeros(shape=(	valid_reverb[0].shape[0],
																		valid_offset	)))),
												valid_freebase[1]))
			)
	test_offset = test_freebase[0].shape[1]-test_reverb[0].shape[1]
	# print(test_offset)
	test = (np.vstack(	
						(np.hstack((test_freebase[0], np.zeros(shape=(	test_freebase[0].shape[0],
																		-test_offset	)))),
												test_reverb[0])),
			np.vstack(	
						(np.hstack((test_freebase[1], np.zeros(shape=(	test_freebase[1].shape[0],
																		-test_offset	)))),
												test_reverb[1]))
			)
	

	# print(len(train), type(train))
	# print(train[0].shape, train[1].shape)
	# print(train[1][:10, :2], train[1][:10, -2:])
	
	# print(len(valid), type(valid))
	# print(valid[0].shape, valid[1].shape)
	# print(valid[1][:10, :2], valid[1][:10, -2:])
	
	# print(len(test), type(test))
	# print(test[0].shape, test[1].shape)
	# print(test[1][:10, :2], test[1][:10, -2:])
	# sys.exit()
	bert = BertModel.from_pretrained("bert-base-uncased")
	tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
	node_edge_detector = NodeEdgeDetector(bert, tokenizer, dropout=torch.tensor(0.5))
	optimizer = AdamW
	kw = {'lr':0.0002, 'weight_decay':0.1}
	tl = TrainingLoop(node_edge_detector, optimizer, True, **kw)
	
	train_dataset = BordersDataset(train)
	train_dataloader = DataLoader(dataset=train_dataset, batch_size=200, shuffle=True, pin_memory=True)
	valid_dataset = BordersDataset(valid)
	valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=200, shuffle=False, pin_memory=True)
	test_dataset = BordersDataset(test)
	test_dataloader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False, pin_memory=True)
	
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	loss = mse_loss
	tl.train(train_dataloader, valid_dataloader, loss)
	tl.save()
	##################################################
	tl.load()
	tl.predict(test_dataloader, device)
	##################################################
	tl.readable_predict(device, print_result=True)
	