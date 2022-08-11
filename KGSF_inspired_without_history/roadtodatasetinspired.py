import json
from collections import defaultdict
import pickle as pkl
from tqdm.auto import tqdm
import os.path
from tqdm import tqdm
import numpy as np

import gensim

#corpus = []
#with open('inspired/train_data_processed.jsonl', 'r', encoding = 'utf-8') as f:
#    for line in tqdm(f):
#        dialog = json.loads(line)
#        for word in dialog['contexts']:
#            #for letter in word:
#                corpus.append(word)
             
#        for word in dialog['response']:
#            #for letter in word:
#                corpus.append(word)

##print(type(corpus))
##print(len(corpus))
##print(corpus[:5])

#modelinspired=gensim.models.word2vec.Word2Vec(sentences = corpus,vector_size=300,min_count=1)
#modelinspired.save('word2vec_inspired')
#word2index = {word: i + 4 for i, word in enumerate(modelinspired.wv.index_to_key)}
##word2index['_split_']=len(word2index)+4
##json.dump(word2index, open('word2index_redial.json', 'w', encoding='utf-8'), ensure_ascii=False)
#word2embedding = [[0] * 300] * 4 + [modelinspired.wv[word] for word in word2index]+[[0]*300]
#import numpy as np
        
#word2index['_split_']=len(word2index)+4
#json.dump(word2index, open('word2index_inspired.json', 'w', encoding='utf-8'), ensure_ascii=False)

#print(np.shape(word2embedding))
#np.save('word2vec_inspired.npy', word2embedding)

#mask4movie = np.load('mask4movie.npy')
#print(np.shape(mask4movie))
#mask4movieinspired = np.zeros(30598)
#print(mask4movie)
#print(np.shape(mask4movieinspired))
#print(mask4movieinspired)
#np.save('mask4movieinspired2.npy', mask4movieinspired)

#mask4key = np.load('mask4key.npy')
#print(np.shape(mask4key))
#mask4keyinspired = np.zeros(30598)
#print(mask4key)
#print(np.shape(mask4keyinspired))
#print(mask4keyinspired)
#np.save('mask4keyinspired2.npy', mask4keyinspired)


#entity2entityId=pkl.load(open('data/entity2entityId.pkl','rb'))
##print(dict(list(entity2entityId.items())[0:2]))
#print(len(entity2entityId))


#text_dict=pkl.load(open('data/movie_ids.pkl','rb'))
##print(len(text_dict))
##print(type(text_dict))
#print(text_dict[:20])

##id2entityId=pkl.load(open('data/id2entity.pkl','rb'))
##print(dict(list(id2entityId.items())[0:2]))

#import random
#listing = list(range(0,17178))

##random.shuffle(listing)


#file = open('movie_ids_new.pkl', 'wb')

### dump information to that file
#pkl.dump(listing, file)

### close the file
#file.close()

listing = pkl.load(open('movie_ids_new.pkl','rb'))
print(listing[:2])
#rec_list=[]

#with open('train_data_processed.jsonl', 'r', encoding='utf-8') as f:
#    for line in tqdm(f):
#            trainprocessed = json.loads(line)
#            #rec_list.append(trainprocessed['movie'])
#            for line1 in trainprocessed['rec']:
#                rec_list.append(line1)

#with open('valid_data_processed.jsonl', 'r', encoding='utf-8') as f:
#    for line in tqdm(f):
#            validprocessed = json.loads(line)
#            #rec_list.append(validprocessed['movie'])
#            for line1 in validprocessed['rec']:
#                rec_list.append(line1)

#with open('test_data_processed.jsonl', 'r', encoding='utf-8') as f:
#    for line in tqdm(f):
#            testprocessed = json.loads(line)
#            #rec_list.append(testprocessed['movie'])
#            for line1 in testprocessed['rec']:
#                rec_list.append(line1)

#print(len(rec_list))

#rec_list = list(dict.fromkeys(rec_list))

#print(len(rec_list))

#print(sorted(rec_list))

#file = open('movie_ids_new.pkl', 'wb')

## dump information to that file
#pkl.dump(rec_list, file)

## close the file
#file.close()

##text_dict=pkl.load(open('data/movie_ids.pkl','rb'))
##print(len(text_dict))
####print(type(text_dict))
###print(text_dict[:20])

##text_dict = list(dict.fromkeys(text_dict))

##print(len(text_dict))

#movieid = pkl.load(open("inspired/item_ids.pkl", "rb"))
#print(type(movieid))
#print(movieid[:20])

#movieid = pkl.load(open("movie_ids.pkl", "rb"))
#print(type(movieid))
#print(movieid[:20])

#with open('inspired/item_ids.json', 'r', encoding='utf-8') as f:
#    item_ids = json.load(f)

#print(type(item_ids))