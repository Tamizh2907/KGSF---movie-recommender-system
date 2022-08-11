from calendar import c
import numpy as np
import csv
from tqdm import tqdm
import pickle as pkl
import json
from nltk import word_tokenize
import re
from torch.utils.data.dataset import Dataset
from copy import deepcopy
import gensim
from collections import defaultdict
from tqdm.auto import tqdm
import os.path
from tqdm import tqdm

#entity2entityId=pkl.load(open('data/entity2entityId.pkl','rb'))
#print(dict(list(entity2entityId.items())[0:2]))
#id2entityId=pkl.load(open('data/id2entity.pkl','rb'))
#print(dict(list(id2entityId.items())[0:2]))
#subkg=pkl.load(open('data/subkg.pkl','rb'))
#print(dict(list(subkg.items())[0:2]))
#text_dict=pkl.load(open('data/text_dict.pkl','rb'))
#print(dict(list(text_dict.items())[0:1]))
#exit()
#text_dict=pkl.load(open('data/movie_ids.pkl','rb'))
#print(text_dict[:2])
#exit()

class dataset(object):
    def __init__(self,filename,opt):
        self.entity2entityId=pkl.load(open('data/entity2entityId.pkl','rb'))
        self.entity_max=len(self.entity2entityId)

        self.id2entity=pkl.load(open('data/id2entity.pkl','rb'))
        self.subkg=pkl.load(open('data/subkg.pkl','rb'))    #need not back process
        self.text_dict=pkl.load(open('data/text_dict.pkl','rb'))

        #all_item = set()
        #file_list = [
        #    'test.jsonl',
        #    'dev.jsonl',
        #    'train.jsonl',
        #]
        #for file in file_list:
        #    self.all_item |= self.get_item_set(file)

        self.item = self._get_item_set(filename)

        #print(f'# all item: {len(all_item)}')

        with open('inspired/kg.pkl', 'rb') as f:
            kg = pkl.load(f)
        #self.subkg = self._extract_subkg(kg, self.item, 2)
        #self.entity2id, self.relation2id, self.subkg = self._kg2id(self.subkg)

        #with open('inspired/dbpedia_subkg.json', 'w', encoding='utf-8') as f:
        #    json.dump(self.subkg, f, ensure_ascii=False)
        #with open('inspired/entity2id.json', 'w', encoding='utf-8') as f:
        #    json.dump(self.entity2id, f, ensure_ascii=False)
        #with open('inspired/relation2id.json', 'w', encoding='utf-8') as f:
        #    json.dump(self.relation2id, f, ensure_ascii=False)

        #with open('inspired/entity2id.json', encoding='utf-8') as f:
        #    self.entity2id = json.load(f)
        
        self.subkg = pkl.load(open('inspired/dbpedia_subkg.pkl','rb'))
        self.relation2id = pkl.load(open('inspired/relation2id.pkl','rb'))
        self.entity2id = pkl.load(open('inspired/entity2id.pkl','rb'))

        length = len(filename)
        self.src_file = filename
        self.tgt_file = filename[:length - 6] + '_data_dbpedia.jsonl'
        self._remove(self.src_file, self.tgt_file)

        self.item_set = set()

        #self.process('test_data_dbpedia.jsonl', 'test_data_processed.jsonl', item_set)
        #self.process('valid_data_dbpedia.jsonl', 'valid_data_processed.jsonl', item_set)

        self.data_file = filename[:length - 6] + '_data_dbpedia.jsonl'
        self.out_file = filename[:length - 6] + '_data_processed.jsonl'
        self._process(self.data_file, self.out_file, self.item_set)

        #with open('inspired/item_ids.json', 'w', encoding='utf-8') as f:
        #    json.dump(list(self.item_set), f, ensure_ascii=False)
        #print(f'#item: {len(self.item_set)}')

        #rec_list=[]

        #with open('train_data_processed.jsonl', 'r', encoding='utf-8') as f:
        #    for line in tqdm(f):
        #        trainprocessed = json.loads(line)
        #        #rec_list.append(trainprocessed['movie'])
        #        for line1 in trainprocessed['rec']:
        #            rec_list.append(line1)

        self.batch_size=opt['batch_size']
        self.max_c_length=opt['max_c_length']
        self.max_r_length=opt['max_r_length']
        self.max_count=opt['max_count']
        self.entity_num=opt['n_entity']
        #self.word2index=json.load(open('word2index.json',encoding='utf-8'))

        #print(type(corpus))
        #print(len(corpus))
        #print(corpus[:5])
        self.data=[]
        cases=[]
        #self.corpus=[]
        with open('inspired/' + self.out_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                lines=json.loads(line)
                contexts=lines['contexts']
                response=lines['response']
                entity=lines['entity']
                movie=lines['movie']
                rec=lines['rec']
                cases.append({'contexts': contexts, 'response': response, 'entity': entity, 'movie': movie, 'rec': rec})
                self.data.extend(cases)

          
            #for values in lines:
            #    self.data.extend(values)


        #print(self.data[0])

        #print(type(self.corpus))
        #print(len(self.corpus))
        #print(self.corpus[:5])

        #if 'train' in filename:

        #self.prepare_word2vec()
        #self.word2index = json.load(open('word2index_redial.json', encoding='utf-8'))
        self.word2index = json.load(open('word2index_inspired.json', encoding='utf-8'))
        self.key2index=json.load(open('key2index_3rd.json',encoding='utf-8'))

        self.stopwords=set([word.strip() for word in open('stopwords.txt',encoding='utf-8')])

        #self.co_occurance_ext(self.data)
        #exit()

    def _get_item_set(self, file):
        entity = set()
        with open('inspired/' + file, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = json.loads(line)
                for turn in line:
                    for e in turn['movie_link']:
                        entity.add(e)
        return entity

    def _extract_subkg(self, kg, seed_set, n_hop):
        subkg = defaultdict(list)  # {head entity: [(relation, tail entity)]}
        subkg_hrt = set()  # {(head_entity, relation, tail_entity)}

        ripple_set = None
        for hop in range(n_hop):
            memories_h = set()  # [head_entity]
            memories_r = set()  # [relation]
            memories_t = set()  # [tail_entity]

            if hop == 0:
                tails_of_last_hop = seed_set  # [entity]
            else:
                tails_of_last_hop = ripple_set[2]  # [tail_entity]

            for entity in tqdm(tails_of_last_hop):
                for relation_and_tail in kg[entity]:
                    h, r, t = entity, relation_and_tail[0], relation_and_tail[1]
                    if (h, r, t) not in subkg_hrt:
                        subkg_hrt.add((h, r, t))
                        subkg[h].append((r, t))
                    memories_h.add(h)
                    memories_r.add(r)
                    memories_t.add(t)

            ripple_set = (memories_h, memories_r, memories_t)

        return subkg

    def _kg2id(self, kg):
        entity_set = self.item

        with open('inspired/relation_set.json', encoding='utf-8') as f:
            relation_set = json.load(f)

        for head, relation_tails in tqdm(kg.items()):
            for relation_tail in relation_tails:
                if relation_tail[0] in relation_set:
                    entity_set.add(head)
                    entity_set.add(relation_tail[1])

        entity2id = {e: i for i, e in enumerate(entity_set)}
        print(f"# entity: {len(entity2id)}")
        relation2id = {r: i for i, r in enumerate(relation_set)}
        relation2id['self_loop'] = len(relation2id)
        print(f"# relation: {len(relation2id)}")

        kg_idx = {}
        for head, relation_tails in kg.items():
            if head in entity2id:
                head = entity2id[head]
                kg_idx[head] = [(relation2id['self_loop'], head)]
                for relation_tail in relation_tails:
                    if relation_tail[0] in relation2id and relation_tail[1] in entity2id:
                        kg_idx[head].append((relation2id[relation_tail[0]], entity2id[relation_tail[1]]))

        return entity2id, relation2id, kg_idx


    def _remove(self, src_file, tgt_file):
        tgt = open('inspired/' + tgt_file, 'w', encoding='utf-8')
        with open('inspired/' + src_file, encoding='utf-8') as f:
            for line in tqdm(f):
                line = json.loads(line)
                for i, message in enumerate(line):
                    new_entity, new_entity_name = [], []
                    for j, entity in enumerate(message['entity_link']):
                        if entity in self.entity2id:
                            new_entity.append(entity)
                            new_entity_name.append(message['entity_name'][j])
                    line[i]['entity_link'] = new_entity
                    line[i]['entity_name'] = new_entity_name

                    new_movie, new_movie_name = [], []
                    for j, movie in enumerate(message['movie_link']):
                        if movie in self.entity2id:
                            new_movie.append(movie)
                            new_movie_name.append(message['movie_name'][j])
                    line[i]['movie_link'] = new_movie
                    line[i]['movie_name'] = new_movie_name

                tgt.write(json.dumps(line, ensure_ascii=False) + '\n')
        tgt.close()

    def _process(self, data_file, out_file, movie_set):
        #reader = csv.reader(open('inspired/dialogpt.csv', 'r'), delimiter= ",")
        #corpus = []
        #for line in reader:
        #    for field in line:
        #        tokens = word_tokenize(field)
        #        corpus.append(tokens)
        #import itertools
        #corpuslist = list(itertools.chain(*corpus))
        #file_content = open("inspired/dialogpt.txt").read()
        #tokenstxt = word_tokenize(file_content)
        #file_content = open("inspired/glutenbergbooks.txt", encoding = 'utf-8').read()
        #tokensglutenberg = word_tokenize(file_content)
        #file_content = open("inspired/morebooks.txt", encoding = 'utf-8').read()
        #tokensmorebooks = word_tokenize(file_content)
        #corpuslist = []
        #for line in tokenstxt:
        #    corpuslist.append(line)
        #for line in tokensglutenberg:
        #    corpuslist.append(line)
        #for line in tokensmorebooks:
        #    corpuslist.append(line)
        #corpuslist.append(tokenstxt)
        #corpusupdate = [corpuslist[a:a+15] for a in range(0, len(corpuslist), 15)]
        #for line in corpusupdate:
        #    corpus.append(line)
        #for line in tokensglutenberg:
        #    corpuslist.append(line)
        #x = sum(1 for line in open('inspired/' + data_file, 'r', encoding='utf-8'))
        #chunkvalue = len(corpus)//x
        #chunks = [corpus[a:a+chunkvalue] for a in range(0, len(corpus), chunkvalue)]
        ##print(len(chunks[0]))
        ##exit()
        with open('inspired/' + data_file, 'r', encoding='utf-8') as fin, open('inspired/' + out_file, 'w', encoding='utf-8') as fout:
            #print('file opened')
            #exit()
            #x = len(fin.readlines())
            #chunkvalue = len(corpuslist)//x
            #chunks = [corpuslist[a:a+chunkvalue] for a in range(0, len(corpuslist), chunkvalue)]
            ##print(len(chunks[0]))
            ##exit()
            #count = 0
            for line in tqdm(fin):
                #print('line opened')
                #exit()
                dialog = json.loads(line)

                context, response, text = [], [], ''
                entity_list, movie_list = [], []

                #context.append(tokens)
                #response.append(tokenstxt)
                #response.append(tokensglutenberg)

                #chunkshalf = chunks[count]
                #half = len(chunkshalf)//2
                #context.append(chunkshalf[:half]) 
                #response.append(chunkshalf[half:])
                #count+=1

                for turn in dialog:
                    text = turn['text']
                    entity_link = [self.entity2id[entity] for entity in turn['entity_link'] if entity in self.entity2id]
                    movie_link = [self.entity2id[movie] for movie in turn['movie_link'] if movie in self.entity2id]
                
                    #if movie_link != ['']:
                    #print(movie_link)

                    if turn['role'] == 'SEEKER':
                        context.append(word_tokenize(text))
                        #entity_list.extend(entity_link + movie_link)
                    else:
                        response.append(word_tokenize(text))
                        #entity_list.extend(entity_link + movie_link)
                    
                    #if len(context) == 0:
                    #context.append('')
                    
                    entity_list.extend(entity_link + movie_link)
                    movie_set |= set(movie_link)
                    movie_list.extend(movie_link)

                for movie in movie_list:
                    lastmovie = movie


                turn = {
                        'contexts': context,
                        'response': response,
                        'entity': list(set(entity_list)),
                        'movie': lastmovie,
                        'rec': 1
                    }
                fout.write(json.dumps(turn, ensure_ascii=False) + '\n')

        #with open('inspired/' + out_file, 'w', encoding='utf-8') as fag:
        #    corpus = []
        #    corpus.append(tokens)
        #    corpus.append(tokenstxt)
        #    corpus.append(tokensglutenberg)



    def prepare_word2vec(self):
        corpus = []
        with open('inspired/' + self.out_file, 'r', encoding = 'utf-8') as f:
            for line in tqdm(f):
                dialog = json.loads(line)
                for word in dialog['contexts']:
                    #for letter in word:
                    corpus.append(word)
             
                for word in dialog['response']:
                    #for letter in word:
                    corpus.append(word)

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

        #import gensim
        #model=gensim.models.word2vec.Word2Vec(self.corpus,size=300,min_count=1)
        #model.save('word2vec_redial')
        #word2index = {word: i + 4 for i, word in enumerate(model.wv.index2word)}
        ##word2index['_split_']=len(word2index)+4
        ##json.dump(word2index, open('word2index_redial.json', 'w', encoding='utf-8'), ensure_ascii=False)
        #word2embedding = [[0] * 300] * 4 + [model[word] for word in word2index]+[[0]*300]
        #import numpy as np
        
        #word2index['_split_']=len(word2index)+4
        #json.dump(word2index, open('word2index_redial.json', 'w', encoding='utf-8'), ensure_ascii=False)

        #print(np.shape(word2embedding))
        #np.save('word2vec_redial.npy', word2embedding)


    def padding_w2v(self,response,max_length,transformer=True,pad=0,end=2,unk=3):
        vector=[]
        concept_mask=[]
        dbpedia_mask=[]
        #print(len(response))
        for word in response:
            for letters in word:
                for digit in letters:
            #print(len(response))
            #print(word)
                #print(letters)
                #exit()
                    vector.append(self.word2index.get(digit,unk))
            ##if word.lower() not in self.stopwords:
                    concept_mask.append(self.key2index.get(digit.lower(),0))
            #exit()
            #else:
            #    concept_mask.append(0)
            #if '@' in word:
            #    try:
            #        entity = self.id2entity[int(word[1:])]
            #        id=self.entity2entityId[entity]
            #    except:
            #        id=self.entity_max
            #    dbpedia_mask.append(id)
            #else:
                    dbpedia_mask.append(self.entity_max)
        vector.append(end)
        concept_mask.append(0)
        dbpedia_mask.append(self.entity_max)

        if len(vector)>max_length:
            if transformer:
                return vector[-max_length:],max_length,concept_mask[-max_length:],dbpedia_mask[-max_length:]
            else:
                return vector[:max_length],max_length,concept_mask[:max_length],dbpedia_mask[:max_length]
        else:
            length=len(vector)
            return vector+(max_length-len(vector))*[pad],length,\
                   concept_mask+(max_length-len(vector))*[0],dbpedia_mask+(max_length-len(vector))*[self.entity_max]

    def padding_context(self,contexts,pad=0,transformer=True):
        vectors=[]
        vec_lengths=[]
        if transformer==False:
            if len(contexts)>self.max_count:
                for sen in contexts[-self.max_count:]:
                    vec,v_l=self.padding_w2v(sen,self.max_r_length,transformer)
                    vectors.append(vec)
                    vec_lengths.append(v_l)
                return vectors,vec_lengths,self.max_count
            else:
                length=len(contexts)
                for sen in contexts:
                    vec, v_l = self.padding_w2v(sen,self.max_r_length,transformer)
                    vectors.append(vec)
                    vec_lengths.append(v_l)
                return vectors+(self.max_count-length)*[[pad]*self.max_c_length],vec_lengths+[0]*(self.max_count-length),length
        else:
            #print(len(contexts))
            #print(self.max_count)
            contexts_com=[]
            for sen in contexts[-self.max_count:-1]:
                #print(contexts[-5:-1])
                #print(sen)
                contexts_com.extend(sen)
                contexts_com.append('_split_')
            contexts_com.extend(contexts[-1])
            #print(contexts_com)
            #exit()
            vec,v_l,concept_mask,dbpedia_mask=self.padding_w2v(contexts_com,self.max_c_length,transformer)
            #print(vec)
            return vec,v_l,concept_mask,dbpedia_mask,0

    def response_delibration(self,response,unk='MASKED_WORD'):
        new_response=[]
        for word in response:
            if word in self.key2index:
                new_response.append(unk)
            else:
                new_response.append(word)
        return new_response

    def data_process(self,is_finetune=False):
        data_set = []
        context_before = []
        for line in self.data:
            #if len(line['contexts'])>2:
            #    continue
            if is_finetune and line['contexts'] == context_before:
                continue
            else:
                context_before = line['contexts']
            context,c_lengths,concept_mask,dbpedia_mask,_=self.padding_context(line['contexts'])
            #print(context)
            #print(line['response'])
            response,r_length,_,_=self.padding_w2v(line['response'],self.max_r_length)
            #print(response)
            if False:
                mask_response,mask_r_length,_,_=self.padding_w2v(self.response_delibration(line['response']),self.max_r_length)
            else:
                mask_response, mask_r_length=response,r_length
            assert len(context)==self.max_c_length
            assert len(concept_mask)==self.max_c_length
            assert len(dbpedia_mask)==self.max_c_length

            data_set.append([np.array(context),c_lengths,np.array(response),r_length,np.array(mask_response),mask_r_length,line['entity'],
                             line['movie'],concept_mask,dbpedia_mask,line['rec']])
        return data_set

    def co_occurance_ext(self,data):
        stopwords=set([word.strip() for word in open('stopwords.txt',encoding='utf-8')])
        keyword_sets=set(self.key2index.keys())-stopwords
        movie_wordset=set()
        for line in data:
            movie_words=[]
            if line['rec']==1:
                for word in line['response']:
                    if '@' in word:
                        try:
                            num=self.entity2entityId[self.id2entity[int(word[1:])]]
                            movie_words.append(word)
                            movie_wordset.add(word)
                        except:
                            pass
            line['movie_words']=movie_words
        new_edges=set()
        for line in data:
            if len(line['movie_words'])>0:
                before_set=set()
                after_set=set()
                co_set=set()
                for sen in line['contexts']:
                    for word in sen:
                        if word in keyword_sets:
                            before_set.add(word)
                        if word in movie_wordset:
                            after_set.add(word)
                for word in line['response']:
                    if word in keyword_sets:
                        co_set.add(word)

                for movie in line['movie_words']:
                    for word in list(before_set):
                        new_edges.add('co_before'+'\t'+movie+'\t'+word+'\n')
                    for word in list(co_set):
                        new_edges.add('co_occurance' + '\t' + movie + '\t' + word + '\n')
                    for word in line['movie_words']:
                        if word!=movie:
                            new_edges.add('co_occurance' + '\t' + movie + '\t' + word + '\n')
                    for word in list(after_set):
                        new_edges.add('co_after'+'\t'+word+'\t'+movie+'\n')
                        for word_a in list(co_set):
                            new_edges.add('co_after'+'\t'+word+'\t'+word_a+'\n')
        f=open('co_occurance.txt','w',encoding='utf-8')
        f.writelines(list(new_edges))
        f.close()
        json.dump(list(movie_wordset),open('movie_word.json','w',encoding='utf-8'),ensure_ascii=False)
        print(len(new_edges))
        print(len(movie_wordset))

    def entities2ids(self,entities):
        return [self.entity2entityId[word] for word in entities]

    def detect_movie(self,sentence,movies):
        token_text = word_tokenize(sentence)
        num=0
        token_text_com=[]
        while num<len(token_text):
            if token_text[num]=='@' and num+1<len(token_text):
                token_text_com.append(token_text[num]+token_text[num+1])
                num+=2
            else:
                token_text_com.append(token_text[num])
                num+=1
        movie_rec = []
        for word in token_text_com:
            if word[1:] in movies:
                movie_rec.append(word[1:])
        movie_rec_trans=[]
        for movie in movie_rec:
            entity = self.id2entity[int(movie)]
            try:
                movie_rec_trans.append(self.entity2entityId[entity])
            except:
                pass
        return token_text_com,movie_rec_trans

    #def _context_reformulate(self,context,movies,altitude,ini_altitude,s_id,re_id):

        #last_id=None
        ##perserve the list of dialogue
        #context_list=[]
        #for message in context:
        #    entities=[]
        #    try:
        #        for entity in self.text_dict[message['text']]:
        #            try:
        #                entities.append(self.entity2entityId[entity])
        #            except:
        #                pass
        #    except:
        #        pass
        #    token_text,movie_rec=self.detect_movie(message['text'],movies)
        #    if len(context_list)==0:
        #        context_dict={'text':token_text,'entity':entities+movie_rec,'user':message['senderWorkerId'],'movie':movie_rec}
        #        context_list.append(context_dict)
        #        last_id=message['senderWorkerId']
        #        continue
        #    if message['senderWorkerId']==last_id:
        #        context_list[-1]['text']+=token_text
        #        context_list[-1]['entity']+=entities+movie_rec
        #        context_list[-1]['movie']+=movie_rec
        #    else:
        #        context_dict = {'text': token_text, 'entity': entities+movie_rec,
        #                   'user': message['senderWorkerId'], 'movie':movie_rec}
        #        context_list.append(context_dict)
        #        last_id = message['senderWorkerId']

        #cases=[]
        #contexts=[]
        #entities_set=set()
        #entities=[]
        #for context_dict in context_list:
        #    self.corpus.append(context_dict['text'])
        #    if context_dict['user']==re_id and len(contexts)>0:
        #        response=context_dict['text']

        #        #entity_vec=np.zeros(self.entity_num)
        #        #for en in list(entities):
        #        #    entity_vec[en]=1
        #        #movie_vec=np.zeros(self.entity_num+1,dtype=np.float)
        #        if len(context_dict['movie'])!=0:
        #            for movie in context_dict['movie']:
        #                #if movie not in entities_set:
        #                cases.append({'contexts': deepcopy(contexts), 'response': response, 'entity': deepcopy(entities), 'movie': movie, 'rec':1})
        #        else:
        #            cases.append({'contexts': deepcopy(contexts), 'response': response, 'entity': deepcopy(entities), 'movie': 0, 'rec':0})

        #        contexts.append(context_dict['text'])
        #        for word in context_dict['entity']:
        #            if word not in entities_set:
        #                entities.append(word)
        #                entities_set.add(word)
        #    else:
        #        contexts.append(context_dict['text'])
        #        for word in context_dict['entity']:
        #            if word not in entities_set:
        #                entities.append(word)
        #                entities_set.add(word)
        #return cases

class CRSdataset(Dataset):
    def __init__(self, dataset, entity_num, concept_num):
        self.data=dataset
        self.entity_num = entity_num
        self.concept_num = concept_num+1

    def __getitem__(self, index):
        '''
        movie_vec = np.zeros(self.entity_num, dtype=np.float)
        context, c_lengths, response, r_length, entity, movie, concept_mask, dbpedia_mask, rec = self.data[index]
        for en in movie:
            movie_vec[en] = 1 / len(movie)
        return context, c_lengths, response, r_length, entity, movie_vec, concept_mask, dbpedia_mask, rec
        '''
        context, c_lengths, response, r_length, mask_response, mask_r_length, entity, movie, concept_mask, dbpedia_mask, rec= self.data[index]
        entity_vec = np.zeros(self.entity_num)
        entity_vector=np.zeros(50,dtype=np.int)
        point=0
        for en in entity:
            entity_vec[en]=1
            entity_vector[point]=en
            point+=1

        concept_vec=np.zeros(self.concept_num)
        for con in concept_mask:
            if con!=0:
                concept_vec[con]=1

        db_vec=np.zeros(self.entity_num)
        for db in dbpedia_mask:
            if db!=0:
                db_vec[db]=1

        #print(movie)

        return context, c_lengths, response, r_length, mask_response, mask_r_length, entity_vec, entity_vector, movie, np.array(concept_mask), np.array(dbpedia_mask), concept_vec, db_vec, rec

    def __len__(self):
        return len(self.data)

if __name__=='__main__':
    ds=dataset('train.jsonl')
    print()
