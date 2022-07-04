import json
from collections import defaultdict
import pickle as pkl
from tqdm.auto import tqdm
import os.path
from tqdm import tqdm
import numpy as np

import gensim

corpus = []
with open('inspired/train_data_processed.jsonl', 'r', encoding = 'utf-8') as f:
    for line in tqdm(f):
        dialog = json.loads(line)
        for word in dialog['contexts']:
            #for letter in word:
                corpus.append(word)
             
        for word in dialog['response']:
            #for letter in word:
                corpus.append(word)

#print(type(corpus))
#print(len(corpus))
#print(corpus[:5])

modelinspired=gensim.models.word2vec.Word2Vec(sentences = corpus,vector_size=300,min_count=1)
modelinspired.save('word2vec_inspired')
word2index = {word: i + 4 for i, word in enumerate(modelinspired.wv.index_to_key)}
#word2index['_split_']=len(word2index)+4
#json.dump(word2index, open('word2index_redial.json', 'w', encoding='utf-8'), ensure_ascii=False)
word2embedding = [[0] * 300] * 4 + [modelinspired.wv[word] for word in word2index]+[[0]*300]
import numpy as np
        
word2index['_split_']=len(word2index)+4
json.dump(word2index, open('word2index_inspired.json', 'w', encoding='utf-8'), ensure_ascii=False)

print(np.shape(word2embedding))
np.save('word2vec_inspired.npy', word2embedding)



