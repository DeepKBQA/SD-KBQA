from graph import *
from train import *
import pandas as pd
from utils import get_hit
import sys

'''
	A simple script to fill article table 
'''
if __name__=='__main__':
	dataset = str(sys.argv[1])
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	bert = BertModel.from_pretrained("bert-base-uncased")
	tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
	node_edge_detector = NodeEdgeDetector(bert, tokenizer, dropout=torch.tensor(0.5))
	optimizer = AdamW
	kw = {'lr':0.0002, 'weight_decay':0.1}
	tl = TrainingLoop(node_edge_detector, optimizer, True, **kw)
	loss = mse_loss
	tl.load('/content/drive/MyDrive/data_freebase/node_edge_bert_v2.pt')

	RKBG = ReverbKnowledgeBase('/content/drive/MyDrive/data_freebase/reverb_wikipedia_tuples-1.1.txt')
	wordstoberttokens_array, berttokenstoids_array, input_token_ids_array, nodes_borders_array, edges_spans_array, node_array, edge_array = [], [], [], [], [], [], []
	questions_array = []
	freebase_df = pd.read_excel(f'/content/OpenQA/data/freebase/{dataset}_useful_records.xlsx')
	reverb_df = pd.read_excel(f'/content/OpenQA/data/reverb/{dataset}.xlsx')
	test_df = pd.DataFrame({
      'Question':freebase_df.Question.to_list()+reverb_df.Question.to_list(),
      'Reverb_no': [-1 for _ in range(len(freebase_df))]+reverb_df.Reverb_no.to_list()
    })
	# actual = test_df['Reverb_no'].to_list()
	# test_df = test_df[:100]
	system_results, candidates_array, actual_answer_array = [], [], []
	for index, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
		wordstoberttokens, berttokenstoids, input_token_ids, nodes_borders, edges_spans, node, edge = tl.readable_predict_article(
                                                device, _input=row['Question'], print_result=False)
		wordstoberttokens_array.append(wordstoberttokens)
		berttokenstoids_array.append(berttokenstoids)
		input_token_ids_array.append(input_token_ids)
		nodes_borders_array.append(nodes_borders)
		edges_spans_array.append(edges_spans)
    
		node = ' '.join(node); edge = ' '.join(edge)
		node = node.replace(' ##', ''); edge = edge.replace(' ##', '')

		node_array.append(node)
		edge_array.append(edge)
		questions_array.append(row['Question'].lower().split())
		temp = RKBG.query(node=node, edge=edge)
		candidates_array.append(temp[:min(len(temp), 25)])
		actual_answer_array.append(row['Reverb_no'])
	output_data = {
        'bert_tokenizer_output':wordstoberttokens_array,
        'bert_token_ids':berttokenstoids_array,
        'Question':test_df.Question.to_list(),
        'input_token_ids':input_token_ids_array,
        'nodes_borders':nodes_borders_array,
        'edges_spans':edges_spans_array,
        'node':node_array,
        'edges':edge_array,
        'question':questions_array,
        'candidates':candidates_array, 
        'actual_answer':actual_answer_array
        }

	pd.DataFrame(output_data).to_excel(f'{dataset}_step_by_step_output.xlsx')