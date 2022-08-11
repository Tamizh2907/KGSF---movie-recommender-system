import numpy as np
from tqdm import tqdm
import pickle as pkl
import json
from nltk import word_tokenize
import re
from torch.utils.data.dataset import Dataset
import numpy as np
from copy import deepcopy

#entity2entityId=pkl.load(open('data/entity2entityId.pkl','rb'))
#print(dict(list(entity2entityId.items())[0:2]))
#id2entityId=pkl.load(open('data/id2entity.pkl','rb'))
#print(dict(list(id2entityId.items())[0:2]))
subkg=pkl.load(open('data/subkg.pkl','rb'))
print(dict(list(subkg.items())[0:2]))
#text_dict=pkl.load(open('data/text_dict.pkl','rb'))
#print(dict(list(text_dict.items())[0:5]))
#text_dict=pkl.load(open('data/movie_ids.pkl','rb'))
#print(text_dict[:2])
#exit()

filename='data/train_data.jsonl'
f=open(filename,encoding='utf-8')
for line in tqdm(f):
    lines=json.loads(line.strip())
    #print(lines)
    #break
    contexts=lines['messages']
    #movies=lines['movieMentions']
    #altitude=lines['respondentQuestions']
    #initial_altitude=lines['initiatorQuestions']
    ##print(movies)
    ##print(altitude)
    #print(contexts)
    for message in contexts:
        #print(message)
        entities=[]
        try:
            for entity in self.text_dict[message['text']]:
                try:
                    entities.append(self.entity2entityId[entity])
                    #print(entities)
                except:
                    pass
                #break
        except:
            #print("outside")
            pass
        #break
    #print(entities)
    ##print(initial_altitude)
    #break