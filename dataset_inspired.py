########################################################################################################################################################################
########################################################################################################################################################################
#####                                                                                                                                                             ######
#####                     This code is originally used by Zhou, Kun, Wayne Xin Zhao, Shuqing Bian, Yuanhang Zhou, Ji-Rong Wen, and Jingsong Y                     ######
#####                     "Improving conversational recommender systems via knowledge graph based semantic fusion." In Proceedings of the 26th                    ######
#####                      ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 1006-1014. 2020.                                         ######                                            
#####                                                                                                                                                             ######
#####                                                                                                                                                             ######
#####                              Some part of the code was taken from from Wang, Xiaolei, Kun Zhou, Ji-Rong Wen, and Wayne Xin Zhao.                            ######                                          
#####                "Towards Unified Conversational Recommender Systems via Knowledge-Enhanced Prompt Learning." arXiv preprint arXiv:2206.09363 (2022).         ######
#####                                                                                                                                                             ######
#####                                                                                                                                                             ######
#####                                                                                                                                                             ######
#####                         It is modified further according to my need and used for this project. Cited and Acknowleged.                                       ######                             
#####                                                                                                                                                             ######
########################################################################################################################################################################
########################################################################################################################################################################
import json
from collections import defaultdict
import pickle as pkl
from tqdm.auto import tqdm
import os.path
from tqdm import tqdm
from nltk import word_tokenize
import gensim

#def get_item_set(file):
#    entity = set()
#    with open('inspired/' + file, 'r', encoding='utf-8') as f:
#        for line in tqdm(f):
#            line = json.loads(line)
#            for turn in line:
#                for e in turn['movie_link']:
#                    entity.add(e)
#    return entity


#def extract_subkg(kg, seed_set, n_hop):
#    subkg = defaultdict(list)  # {head entity: [(relation, tail entity)]}
#    subkg_hrt = set()  # {(head_entity, relation, tail_entity)}

#    ripple_set = None
#    for hop in range(n_hop):
#        memories_h = set()  # [head_entity]
#        memories_r = set()  # [relation]
#        memories_t = set()  # [tail_entity]

#        if hop == 0:
#            tails_of_last_hop = seed_set  # [entity]
#        else:
#            tails_of_last_hop = ripple_set[2]  # [tail_entity]

#        for entity in tqdm(tails_of_last_hop):
#            for relation_and_tail in kg[entity]:
#                h, r, t = entity, relation_and_tail[0], relation_and_tail[1]
#                if (h, r, t) not in subkg_hrt:
#                    subkg_hrt.add((h, r, t))
#                    subkg[h].append((r, t))
#                memories_h.add(h)
#                memories_r.add(r)
#                memories_t.add(t)

#        ripple_set = (memories_h, memories_r, memories_t)

#    return subkg


#def kg2id(kg):
#    entity_set = all_item

#    with open('inspired/relation_set.json', encoding='utf-8') as f:
#        relation_set = json.load(f)

#    for head, relation_tails in tqdm(kg.items()):
#        for relation_tail in relation_tails:
#            if relation_tail[0] in relation_set:
#                entity_set.add(head)
#                entity_set.add(relation_tail[1])

#    entity2id = {e: i for i, e in enumerate(entity_set)}
#    print(f"# entity: {len(entity2id)}")
#    relation2id = {r: i for i, r in enumerate(relation_set)}
#    relation2id['self_loop'] = len(relation2id)
#    print(f"# relation: {len(relation2id)}")

#    kg_idx = {}
#    for head, relation_tails in kg.items():
#        if head in entity2id:
#            head = entity2id[head]
#            kg_idx[head] = [(relation2id['self_loop'], head)]
#            for relation_tail in relation_tails:
#                if relation_tail[0] in relation2id and relation_tail[1] in entity2id:
#                    kg_idx[head].append((relation2id[relation_tail[0]], entity2id[relation_tail[1]]))

#    return entity2id, relation2id, kg_idx


#all_item = set()
#file_list = [
#    'test.jsonl',
#    'valid.jsonl',
#    'train.jsonl',
#]
#for file in file_list:
#    all_item |= get_item_set(file)
#print(f'# all item: {len(all_item)}')

#with open('inspired/kg.pkl', 'rb') as f:
#    kg = pkl.load(f)
#subkg = extract_subkg(kg, all_item, 2)
#entity2id, relation2id, subkg = kg2id(subkg)

#file = open('inspired/dbpedia_subkg.pkl', 'wb')
### dump information to that file
#pkl.dump(subkg, file)
### close the file
#file.close()

#file = open('inspired/entity2id.pkl', 'wb')
### dump information to that file
#pkl.dump(entity2id, file)
### close the file
#file.close()

#file = open('inspired/relation2id.pkl', 'wb')
### dump information to that file
#pkl.dump(relation2id, file)
### close the file
#file.close()

##with open('inspired/dbpedia_subkg.json', 'w', encoding='utf-8') as f:
##    json.dump(subkg, f, ensure_ascii=False)
##with open('inspired/entity2id.json', 'w', encoding='utf-8') as f:
##    json.dump(entity2id, f, ensure_ascii=False)
##with open('inspired/relation2id.json', 'w', encoding='utf-8') as f:
##    json.dump(relation2id, f, ensure_ascii=False)

##with open('inspired/entity2id.json', encoding='utf-8') as f:
##    entity2id = json.load(f)

#entity2id = pkl.load(open('inspired/entity2id.pkl','rb'))


#def remove(src_file, tgt_file):
#    tgt = open('inspired/' + tgt_file, 'w', encoding='utf-8')
#    with open('inspired/' + src_file, encoding='utf-8') as f:
#        for line in tqdm(f):
#            line = json.loads(line)
#            for i, message in enumerate(line):
#                new_entity, new_entity_name = [], []
#                for j, entity in enumerate(message['entity_link']):
#                    if entity in entity2id:
#                        new_entity.append(entity)
#                        new_entity_name.append(message['entity_name'][j])
#                line[i]['entity_link'] = new_entity
#                line[i]['entity_name'] = new_entity_name

#                new_movie, new_movie_name = [], []
#                for j, movie in enumerate(message['movie_link']):
#                    if movie in entity2id:
#                        new_movie.append(movie)
#                        new_movie_name.append(message['movie_name'][j])
#                line[i]['movie_link'] = new_movie
#                line[i]['movie_name'] = new_movie_name

#            tgt.write(json.dumps(line, ensure_ascii=False) + '\n')
#    tgt.close()


#src_files = ['test.jsonl', 'valid.jsonl', 'train.jsonl']
#tgt_files = ['test_data_dbpedia.jsonl', 'valid_data_dbpedia.jsonl', 'train_data_dbpedia.jsonl']
#for src_file, tgt_file in zip(src_files, tgt_files):
#    remove(src_file, tgt_file)

#def process(data_file, out_file, movie_set):
#    with open('inspired/' + data_file, 'r', encoding='utf-8') as fin, open('inspired/' + out_file, 'w', encoding='utf-8') as fout:
#        for line in tqdm(fin):
#            dialog = json.loads(line)

#            context, response, text = [], [], ''
#            entity_list, movie_list = [], []

#            for turn in dialog:
#                #print(turn)
#                #exit()
#                text = turn['text']
#                entity_link = [entity2id[entity] for entity in turn['entity_link'] if entity in entity2id]
#                movie_link = [entity2id[movie] for movie in turn['movie_link'] if movie in entity2id]
                
#                #if movie_link != ['']:
#                #    print(movie_link)

#                if turn['role'] == 'SEEKER':
#                    context.append(word_tokenize(text))
#                    #entity_list.extend(entity_link + movie_link)
#                else:
#                    response.append(word_tokenize(text))
#                    #entity_list.extend(entity_link + movie_link)
                    
#                #if len(context) == 0:
#                #    context.append('')
                    
#                entity_list.extend(entity_link + movie_link)
#                movie_set |= set(movie_link)
#                movie_list.extend(movie_link)

#            for movie in movie_list:
#                lastmovie = movie

#            turn = {
#                    'contexts': context,
#                    'response': response,
#                    'entity': list(set(entity_list)),
#                    'movie': lastmovie,
#                    'rec': 1
#                }
#            fout.write(json.dumps(turn, ensure_ascii=False) + '\n')

##with open('inspired/entity2id.json', 'r', encoding='utf-8') as f:
##    entity2id = json.load(f)
#item_set = set()

#process('test_data_dbpedia.jsonl', 'test_data_processed.jsonl', item_set)
#process('valid_data_dbpedia.jsonl', 'valid_data_processed.jsonl', item_set)
#process('train_data_dbpedia.jsonl', 'train_data_processed.jsonl', item_set)

##with open('inspired/item_ids.json', 'w', encoding='utf-8') as f:
##    json.dump(list(item_set), f, ensure_ascii=False)
#file = open('inspired/item_ids.pkl', 'wb')
### dump information to that file
#pkl.dump(list(item_set), file)
### close the file
#file.close()

#print(f'#item: {len(item_set)}')

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


from nltk.tokenize import word_tokenize
#import pandas as pd
#csvfile = pd.read_csv('inspired/dialogpt.csv', encoding = 'utf-8', header=None)
#print(csvfile)
#print(len(csvfile))
#text = nltk.word_tokenize(csvfile.values)
#corpus.append(text)
import csv
#reader = csv.reader(open('inspired/dialogpt.csv', 'r'), delimiter= ",")
#for line in reader:
#    for field in line:
#        tokens = word_tokenize(field)
#        corpus.append(tokens)

#print(type(corpus))
#import itertools
#corpuslist = list(itertools.chain(*corpus))
reader = csv.reader(open('inspired/dialogpt.csv', 'r'), delimiter= ",")
corpus = []
for line in reader:
    for field in line:
        tokens = word_tokenize(field)
        corpus.append(tokens)
#print(corpus[1:100])
#import itertools
#corpuslist = list(itertools.chain(*corpus))
#print(corpuslist[1:100])
file_content = open("inspired/dialogpt.txt").read()
tokenstxt = word_tokenize(file_content)
file_content = open("inspired/glutenbergbooks.txt", encoding = 'utf-8').read()
tokensglutenberg = word_tokenize(file_content)
file_content = open("inspired/morebooks.txt", encoding = 'utf-8').read()
tokensmorebooks = word_tokenize(file_content)
corpuslist = []
for line in tokenstxt:
    corpuslist.append(line)
for line in tokensglutenberg:
    corpuslist.append(line)
for line in tokensmorebooks:
    corpuslist.append(line)
#print(len(corpus))
#print(len(corpuslist))
#print(corpuslist[1:10])
corpusupdate = [corpuslist[a:a+15] for a in range(0, len(corpuslist), 15)]
#corpuslist.append(tokenstxt)
#print(corpusupdate[1:10])
#print(len(corpus))
#print(len(corpusupdate))
for line in corpusupdate:
    corpus.append(line)

print(len(corpus))

#corpuslist.append(tokensglutenberg)
#for line in tokensglutenberg:
#    corpuslist.append(line)
#print(corpuslist[-100:-1])
#corpuslistfinal = list(itertools.chain(*corpuslist[-1]))
#print(len(corpuslist))
#print(corpuslistfinal[1:100])
#print(corpuslistfinal[:2])
#file_content = open("inspired/dialogpt.txt").read()
#tokens = word_tokenize(file_content)
#corpus.append(tokens)

#file_content = open("inspired/glutenbergbooks.txt", encoding = 'utf-8').read()
#tokenstxt = word_tokenize(file_content)
#corpus.append(tokens)
##corpus.append('_split_')
##print(type(corpus))
#print(len(tokenstxt))
#print(len(corpus))
#print(tokens[:5])
#print(corpus[:-1])
#exit()
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
#print(len(corpus))
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


