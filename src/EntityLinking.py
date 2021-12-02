#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/meti-94/OpenQA/blob/main/EntityLinking.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


get_ipython().system('pip install transformers -q')
get_ipython().system('pip install fuzzywuzzy -q')
get_ipython().system('pip install python-Levenshtein -q')


# In[ ]:


get_ipython().system('pip install fuzzywuzzy -q')
get_ipython().system('git clone https://github.com/castorini/BuboQA.git')
get_ipython().run_line_magic('cd', '/content/BuboQA')
get_ipython().system('bash setup.sh ')


# In[ ]:


cp -r /content/BuboQA/indexes /content/drive/MyDrive


# In[ ]:


cp -r /content/BuboQA/data/processed_simplequestions_dataset /content/drive/MyDrive


# In[2]:


import pandas as pd
import pickle
from tqdm import tqdm
from fuzzywuzzy import fuzz


# In[18]:


# mapping between MIDs and names in the form of dict['MID']=['str1', 'str2', ...,  'strN']
with open('/content/drive/MyDrive/indexes/names_2M.pkl', 'rb') as f:
    mid2name = pickle.load(f)


# In[19]:


# In/Out degree for each MID in the form of dict['MID']=[In degree, Out degree]
with open('/content/drive/MyDrive/indexes/degrees_2M.pkl', 'rb') as f:
    degrees_2M = pickle.load(f)


# In[20]:


# reverse mapping between string and MID in the form of dict['string']=[('MID', 'actual string', 'freebase type') ...] 
with open('/content/drive/MyDrive/indexes/entity_2M.pkl', 'rb') as f:
    entity_2M = pickle.load(f)


# In[21]:


# mapping between MIDs and Relations in the form of dict['MID']=[{'fb:common.topic.notable_types', 'fb:people.person.gender', 'fb:people.person.profession'}]
with open('/content/drive/MyDrive/indexes/reachability_2M.pkl', 'rb') as f:
    reachability_2M = pickle.load(f)


# In[22]:


reverb2freebace = pd.read_csv('/content/drive/MyDrive/reverb2freebase.csv')
reverb2freebace['freebase_ID_argument1'] = reverb2freebace['freebase_ID_argument1'].apply(lambda string:'fb:m.'+string)
reverb2freebace['conf'] = reverb2freebace['conf'].astype(float)


# In[23]:


mid_count = len(mid2name)
string_count = sum([len(name) for name in mid2name.values()])
original_relations = sum([len(relation) for relation in reachability_2M.values()])

matched_mids = 0
new_entity_strings = 0
new_relations = 0

for index, row in tqdm(reverb2freebace.iterrows(), total=reverb2freebace.shape[0]):
  mid = row['freebase_ID_argument1']
  reverb_string = row['arg1']
  relation = row['rel']
  conf = str(row['conf'])
  try:
    temp = mid2name[mid]
    matched_mids+=1
  except:
    continue
  if reverb_string not in temp:
    temp.append((reverb_string, conf))
    new_entity_strings+=1
  temp = reachability_2M[mid]
  if relation not in temp:
    temp.add((relation, conf))
    new_relations+=1
  try:
    temp = entity_2M[reverb_string]
    temp.add((mid, reverb_string, conf))
  except:
    entity_2M[reverb_string] = set([(mid, reverb_string, conf)])
  
  degrees_2M[mid][1]+=1


print(f'\nOriginal MIDs: {mid_count}\tMatched MIDs: {matched_mids}')
print(f'Original Entity Strings: {string_count}\tAdded Entity Strings: {new_entity_strings}') 
print(f'Original Relations Count: {original_relations}\tAdded Relations Count: {new_relations}') 


# In[24]:


def get_ngram(text):
    #ngram = set()
    ngram = []
    tokens = str(text).split()
    for i in range(len(tokens)+1):
        for j in range(i):
            if i-j <= 3:
                #ngram.add(" ".join(tokens[j:i]))
                temp = " ".join(tokens[j:i])
                if temp not in ngram:
                    ngram.append(temp)
    #ngram = list(ngram)
    ngram = sorted(ngram, key=lambda x: len(x.split()), reverse=True)
    return ngram


# In[25]:


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


# In[26]:


from collections import defaultdict
inverted_index = defaultdict(str, entity_2M)
print("Total type of text: {}".format(len(inverted_index)))
max_len = 0
_entry = ""
for entry, value in inverted_index.items():
  if len(value) > max_len:
    max_len = len(value)
    _entry = list(value)
print("Max Length of entry is {}, text is {}".format(max_len, _entry[:100]))


# In[27]:


del entity_2M
del degrees_2M
del reachability_2M
del mid2name


# In[37]:


reverb2freebace.head()


# In[36]:


reverb2freebace.iloc[168]


# In[31]:


pred_df_freebase = pd.read_excel('/content/freebase_answers.xlsx')
pred_df_reverb = pd.read_excel('/content/reverb_answers.xlsx')
print(len(pred_df_freebase), len(pred_df_reverb))
pred_df = pd.concat([pred_df_freebase, pred_df_reverb])
print(len(pred_df))
# pred_df['Question'] = pred_df['questions'].apply(lambda item:' '.join(eval(item)).replace(' #', ''))
# gold_df_freebase = pd.read_excel('/content/drive/MyDrive/data_freebase/test_useful_records.xlsx')
# gold_df_
# print(len(pred_df), len(gold_df))
# df = pred_df.merge(gold_df, on='Question', how ='inner')
# print(len(df))

# predicteds = df['node'].astype(str).to_list()
# golds = df['Answer'].to_list()


# In[29]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import sys
def entity_linking(data_type, predicteds, golds, HITS_TOP_ENTITIES, output):
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

    for idx, (predicted, gold_id) in tqdm(enumerate(zip(predicteds, golds))):
        total += 1
        C = []
        C_scored = []
        tokens = get_ngram(predicted)

        if len(tokens) > 0:
            maxlen = len(tokens[0].split())
        for item in tokens:
            if len(item.split()) < maxlen and len(C) == 0:
                maxlen = len(item.split())
            if len(item.split()) < maxlen and len(C) > 0:
                break
            if item in stopword:
                continue
            C.extend(inverted_index[item])
        for mid_text_type in set(C):
            score = fuzz.ratio(mid_text_type[1], predicted.strip()) / 100.0
            C_scored.append((mid_text_type, score))
        C_scored.sort(key=lambda t: t[1], reverse=True)
        cand_mids = C_scored[:HITS_TOP_ENTITIES]
        for mid_text_type, score in cand_mids:
            fout.write(" %%%% {}\t{}\t{}\t{}".format(mid_text_type[0], mid_text_type[1], mid_text_type[2], score))
        fout.write('\n')
        
        midList = [x[0][0] for x in cand_mids]
        if gold_id in midList[:1]:
            top1 += 1
        if gold_id in midList[:3]:
            top3 += 1
        if gold_id in midList[:5]:
            top5 += 1
        if gold_id in midList[:10]:
            top10 += 1
        if gold_id in midList[:20]:
            top20 += 1
        if gold_id in midList[:50]:
            top50 += 1
        if gold_id in midList[:100]:
            top100 += 1

    print(data_type)
    print("Top1 Entity Linking Accuracy: {}".format(top1 / total))
    print("Top3 Entity Linking Accuracy: {}".format(top3 / total))
    print("Top5 Entity Linking Accuracy: {}".format(top5 / total))
    print("Top10 Entity Linking Accuracy: {}".format(top10 / total))
    print("Top20 Entity Linking Accuracy: {}".format(top20 / total))
    print("Top50 Entity Linking Accuracy: {}".format(top50 / total))
    print("Top100 Entity Linking Accuracy: {}".format(top100 / total))


# In[30]:


entity_linking('test', predicteds, golds, 100, "./results")


# In[ ]:




