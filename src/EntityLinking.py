# importing necessary libraries
import pandas as pd
import pickle
from tqdm import tqdm
from fuzzywuzzy import fuzz
import os
from collections import defaultdict
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import sys

# reading indices
def read_indices(base_path='/content/drive/MyDrive/indexes/'):
  # mapping between MIDs and names in the form of dict['MID']=['str1', 'str2', ...,  'strN']
  with open(os.path.join(base_path, 'names_2M.pkl'), 'rb') as f:
    mid2name = pickle.load(f)
  # reverse mapping between string and MID in the form of dict['string']=[('MID', 'actual string', 'freebase type') ...] 
  with open(os.path.join(base_path, 'entity_2M.pkl'), 'rb') as f:
    entity_2M = pickle.load(f)
  # In/Out degree for each MID in the form of dict['MID']=[In degree, Out degree]
  with open(os.path.join(base_path, 'degrees_2M.pkl'), 'rb') as f:
    degrees_2M = pickle.load(f)
  # mapping between MIDs and Relations in the form of dict['MID']=[{'fb:common.topic.notable_types', 'fb:people.person.gender', 'fb:people.person.profession'}]
  with open(os.path.join(base_path, 'reachability_2M.pkl'), 'rb') as f:
    reachability_2M = pickle.load(f)
  return mid2name, entity_2M, degrees_2M, reachability_2M


# reading reverb freebase combination file
def read_reverb2freebase(path='/content/drive/MyDrive/reverb2freebase.csv'):
  reverb2freebace = pd.read_csv(path)
  reverb2freebace['freebase_ID_argument1'] = reverb2freebace['freebase_ID_argument1'].apply(lambda string:'fb:m.'+str(string))
  reverb2freebace['conf'] = reverb2freebace['conf'].astype(float)
  return reverb2freebace

# adding new nodes to KB
def combine(mid2name, entity_2M, degrees_2M, reachability_2M, reverb2freebace):
  matched_mids, unmatched_mids, added_entity_strings = 0, 0, 0
  string_count = sum([len(name) for name in entity_2M.values()])


  for index, row in tqdm(reverb2freebace.iterrows(), total=reverb2freebace.shape[0]):
    if row['freebase_ID_argument1'] in mid2name:
      mid1 = mid = row['freebase_ID_argument1'] 
      matched_mids+=1
    else:
      mid1 = mid = row['argument1_uuid']
      unmatched_mids+=1
    mid2 = mid = row['argument2_uuid']
    unmatched_mids+=1
    reverb_string1 = str(row['arg1']).lower()
    reverb_string2 = str(row['arg2']).lower()
    relation = row['rel']
    conf = str(row['conf'])
    linking = str(row['link_score'])
    try:
      temp = entity_2M[reverb_string1]
    except:
      entity_2M[reverb_string1] = set()
      temp = entity_2M[reverb_string1]
      added_entity_strings+=1
    temp.add((mid1, reverb_string1, conf))
    try:
      temp = entity_2M[reverb_string2]
    except:
      entity_2M[reverb_string2] = set()
      temp = entity_2M[reverb_string2]
      added_entity_strings+=1
    temp.add((mid2, reverb_string2, conf))
  
  print(f'\nTotal MIDs before augmentation: {len(mid2name)}\tUnmatched (Added) MIDs: {unmatched_mids}\t Matched MIDs: {matched_mids}')
  print(f'Total Entity Strings before augmentation: {string_count}\tAdded Entity Strings: {added_entity_strings}') 
  return mid2name, entity_2M, degrees_2M, reachability_2M

# creating N-grams from input text
def get_ngram(text):
    ngram = []
    tokens = str(text).split()
    for i in range(len(tokens)+1):
        for j in range(i):
            if i-j <= 3:
                temp = " ".join(tokens[j:i])
                if temp not in ngram:
                    ngram.append(temp)
    ngram = sorted(ngram, key=lambda x: len(x.split()), reverse=True)
    return ngram

def get_stat_inverted_index(reverse_index):
    """
    Get the number of entry and max length of the entry (How many mid in an entry)
    """
    with open(filename, "rb") as handler:
        global  inverted_index
        inverted_index = pickle.load(handler)
        inverted_index = defaultdict(str, inverted_index)
    print("Total type of text: {}".format(len(inverted_index)))
    max_len = 0
    _entry = ""
    for entry, value in inverted_index.items():
        if len(value) > max_len:
            max_len = len(value)
            _entry = entry
    print("Max Length of entry is {}, text is {}".format(max_len, _entry))

# creating invese index
def reverse_index(entity_2M):
  inverted_index = defaultdict(str, entity_2M)
  print("Total type of text: {}".format(len(inverted_index)))
  max_len = 0
  _entry = ""
  for entry, value in inverted_index.items():
    if len(value) > max_len:
      max_len = len(value)
      _entry = list(value)
  print("Max Length of entry is {}, text is {}".format(max_len, _entry[:100]))
  return inverted_index

def saving_memory(mid2name, entity_2M, degrees_2M, reachability_2M):
  del entity_2M
  del degrees_2M
  del reachability_2M
  del mid2name

# Entity linking and evaluation 
def entity_linking(data_type, predicteds, golds, HITS_TOP_ENTITIES, output, inverted_index):
    stopword = set(stopwords.words('english'))
    fout = open(output, 'w')
    total = 0
    top1 = 0
    top3 = 0
    top5 = 0
    top10 = 0
    top20 = 0
    top50 = 0
    top100 = 0
    hits = []
    candidates = []
    for idx, (predicted, gold_id) in tqdm(enumerate(zip(predicteds, golds))):
        bflag = True
        total += 1
        C = []
        C_scored = []
        tokens = get_ngram(predicted)
        # print(tokens)
        if len(tokens) > 0:
            maxlen = len(tokens[0].split())
            # print(maxlen)
        for item in tokens:
            if len(item.split()) < maxlen and len(C) == 0:
                maxlen = len(item.split())
            if len(item.split()) < maxlen and len(C) > 0:
                break
            if item in stopword:
                # print('his is stopword', item)
                continue
            C.extend(inverted_index[item])
            # print(inverted_index[item])
        for mid_text_type in set(C):
            score = fuzz.ratio(mid_text_type[1], predicted.strip()) / 100.0
            C_scored.append((mid_text_type, score))
        C_scored.sort(key=lambda t: t[1], reverse=True)
        # print(C_scored[:100])
        # sys.exit()
        candidates.append(C_scored[:100])
        cand_mids = C_scored[:HITS_TOP_ENTITIES]
        for mid_text_type, score in cand_mids:
            fout.write(" %%%% {}\t{}\t{}\t{}".format(mid_text_type[0], mid_text_type[1], mid_text_type[2], score))
        fout.write('\n')
        
        midList = [x[0][0] for x in cand_mids]
        if gold_id in midList[:1]:
            top1 += 1
            if bflag:
              hits.append(1)
              bflag=False
        if gold_id in midList[:3]:
            top3 += 1
            if bflag:
              hits.append(3)
              bflag=False            
        if gold_id in midList[:5]:
            top5 += 1
            if bflag:
              hits.append(5)
              bflag=False
        if gold_id in midList[:10]:
            top10 += 1
            if bflag:
              hits.append(10)
              bflag=False
        if gold_id in midList[:20]:
            top20 += 1
            if bflag:
              hits.append(20)
              bflag=False
        if gold_id in midList[:50]:
            top50 += 1
            if bflag:
              hits.append(50)
              bflag=False
        if gold_id in midList[:100]:
            top100 += 1
            if bflag:
              hits.append(100)
              bflag=False
        if bflag:
          hits.append(-1)
          bflag=False

    print(data_type)
    print("Top1 Entity Linking Accuracy: {}".format(top1 / total))
    print("Top3 Entity Linking Accuracy: {}".format(top3 / total))
    print("Top5 Entity Linking Accuracy: {}".format(top5 / total))
    print("Top10 Entity Linking Accuracy: {}".format(top10 / total))
    print("Top20 Entity Linking Accuracy: {}".format(top20 / total))
    print("Top50 Entity Linking Accuracy: {}".format(top50 / total))
    print("Top100 Entity Linking Accuracy: {}".format(top100 / total))
    return hits, candidates


if __name__=='__main__':
  # reading step-by-step output
  test_df = pd.read_excel('/content/drive/MyDrive/data_freebase/valid_sbs.xlsx')
  # adding actual entity for reverb
  temp_df = pd.read_excel('/content/OpenQA/data/reverb/valid.xlsx')[['Question', 'answer_entity']]
  test_df = test_df.merge(temp_df, how='inner', left_on='Question', right_on='Question')
  reverb2freebace = read_reverb2freebase()
  questions_fact = reverb2freebace.merge(test_df, how='inner', left_on='reverb_no', right_on='Reverb_no')
  
  # reading freebase questions 
  freebase = pd.read_excel('/content/OpenQA/data/freebase/valid_useful_records.xlsx')
  golds = []
  for idx, row in questions_fact.iterrows():
    if row['freebase_ID_argument1']=='fb:m.nan':
      ans_ent = row['answer_entity']
      golds.append(row[f'argument{ans_ent}_uuid'])
    else:
      golds.append(row['freebase_ID_argument1'])
  predicteds = questions_fact.node.astype(str).to_list()
  print(f'Reverb Questions Count: {len(golds)}\tFreebase Questions Count: {len(freebase)}')
  golds+=freebase.Answer.to_list()
  predicteds+=freebase.entity.to_list()
  
  questions =  questions_fact.Question.to_list()+freebase.Question.to_list()

  mid2name, entity_2M, degrees_2M, reachability_2M = read_indices()
  mid2name, entity_2M, degrees_2M, reachability_2M = combine(mid2name, entity_2M, degrees_2M, reachability_2M, reverb2freebace)
  inverted_index = reverse_index(entity_2M)
  saving_memory(mid2name, entity_2M, degrees_2M, reachability_2M)
  hits, candidates = entity_linking('test', predicteds, golds, 100, "./results", inverted_index)
  pd.DataFrame({
    'Question':questions,
    'NER':predicteds,
    'Answer':golds,
    'HIT@':hits,
    'Candidates':candidates
  }).to_excel('Linking.xlsx', index=False)
