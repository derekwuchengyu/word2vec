import sys,os
import numpy as np
import pandas as pd
import datetime
import math
from gensim.models import word2vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.callbacks import PerplexityMetric
from gensim.models.word2vec import train_sg_pair
from pprint import pprint 
import time
import memcache,redis
from scipy.special import expit,softmax
from collections import defaultdict
from google.cloud import bigquery as bq
from utils.module import *
from p591.config import *
from clickhouse_driver import Client
from clickhouse_driver import connect
import multiprocessing as mp
from dotenv import load_dotenv
load_dotenv()

class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self,corpus,settings=None):
        self.tt = time()
        self.corpus = corpus
        self.epochs = 1 
        if settings:
            self.setting(**settings)
        # self.load_data()
    def pre_train(self,model):
        self.model = model
        self.w1 = self.model.wv.vectors 
        self.w2 = self.model.trainables.syn1neg 
        self.model.running_training_loss = 0
        self.size = min(self.w1.shape)
        if self.epochs == 1 :
            print("Epochs start")
            self.cum_table = self.model.vocabulary.cum_table
            self.word_counts = word_counts = {word:vocab.count for word,vocab in self.model.wv.vocab.items()}
            self.v_count = len(word_counts.keys()) # Unique總字數
            self.word_index = {word:vocab.index for word,vocab in self.model.wv.vocab.items()} # index-詞
            self.index_word = {vocab.index:word for word,vocab in self.model.wv.vocab.items()} # 詞-index 
            self.words_list = list(self.word_index.keys())
            # if self.negative>0:
            #     self.make_cum_table()
            # if self.local_negative>0:
            self.make_local_cum_table()
        # self.timerecord()      


    # local 負採樣映射表
    def make_local_cum_table(self, escape_list=[], domain=2**31 - 1):
        cum_table = {}
        cum_table_map = {}
        train_words_pow = {}
        cumulative = {}

        for word_index in range(self.v_count):
            word = self.index_word[word_index]
            group = word[:2]
            if group not in cum_table.keys():
                cum_table[group] = np.zeros(self.v_count)
                train_words_pow[group] = 0.0
                cumulative[group] = 0.0
                cum_table_map[group] = []
            if word[2:] not in escape_list:
                train_words_pow[group] += self.word_counts[word]**self.ns_exponent
                cum_table_map[group].append(self.word_index[word])
        for word_index in range(self.v_count):
            word = self.index_word[word_index]
            group = word[:2]
            if word[2:] not in escape_list:
                cumulative[group] += self.word_counts[word]**self.ns_exponent
                cum_table[group][word_index] = round(cumulative[group] / train_words_pow[group] * domain)  
        for group,value in cum_table.items():
            cum_table[group] = np.array([num for num in value.tolist() if num > 0])    
        self.local_cum_table = cum_table
        self.local_cum_table_map = cum_table_map
    def timerecord(self, string=""):
        print(str(string)+str(round(time()-self.tt,4)))
        self.tt = time()
 
    def getloss(self,center,context):
        w1=self.w1
        w2=self.w2
        pre = expit(np.dot(w2,w1[center].T))
        word_vec = [0 for i in range(0, self.v_count)]
        word_vec = np.array(word_vec)
        word_vec[context] = 1
        if math.isnan(np.sum(np.subtract(pre,word_vec))):
            print(center,context)
            print(w1)
            print(w2)
        return np.sum(np.subtract(pre,word_vec))
    
    def progesss_rate(self,index):
        total = len(self.corpus)
        if index % round(total/10) ==0 and index!=0 :
            print(round(index*100/total),"%")

    def setting(self,window_size=4,size=25,negative=5,learning_rate=0.005,ns_exponent=0.75,
                local_negative=0,neg_mtpl=1,glb_mtpl=1,local_neg_min=200):
        self.n = size
        self.lr = learning_rate
        self.window = window_size
        self.negative = negative
        self.ns_exponent = ns_exponent
        self.local_negative = local_negative
        self.neg_mtpl = neg_mtpl
        self.glb_mtpl = glb_mtpl 
        self.local_neg_min = local_neg_min
    def weight_update(self, center, word_indices, tj, lr=-1):
        if lr == -1:
            lr = self.lr
        if len(word_indices)==0:
            pass
        h = self.w1[center]
        w2_tmp = self.w2[word_indices]
        u = np.dot(w2_tmp, h.T)
        y_pred = expit(u)
        EI = np.subtract(y_pred,tj)

        #  W1 更新 
        dlt_w1 = np.dot(EI,w2_tmp)
        self.w1[center] -= (lr * dlt_w1)
        
        # W2 更新 
        dlt_w2 = np.outer(EI,h.T)
        self.w2[word_indices] -= (lr * dlt_w2)
        
        # if np.sum(dlt_w1)>10000 or np.sum(dlt_w1)<-10000: # debug用
        #     print(center,"-",(dlt_w1))
        #     print(center,"-",EI)
        
        # 計算loss
        if np.random.randint(30000) < 1:
            lss = self.getloss(center,word_indices)
            # print("loss",lss)
            
    def weight_update2(self, context, word_indices, lr=-1,compute_loss=True):
        if lr == -1:
            lr = self.lr
        if len(word_indices)==0:
            pass

        l1 = self.w1[context] #(h)
        l2b = self.w2[word_indices] #(w2_tmp)
        neu1e = np.zeros(l1.shape)
        tj  = np.zeros(len(word_indices))
        tj[0] = 1
        
        # forward
        prod_term = np.dot(l1, l2b.T)
        EI = expit(prod_term)
        gb = (tj - EI) * lr
        
        #  W1 更新 
        self.w1[context] += np.dot(gb, l2b)
        
        # W2 更新 
        self.w2[word_indices] += np.outer(gb, l1) 
        if compute_loss:
            if expit(-1 * prod_term[1:])==0 or expit(prod_term[0])==0:
                pass
            self.model.running_training_loss -= np.sum(np.log(expit(-1 * prod_term[1:])))  # for the sampled words
            self.model.running_training_loss -= np.log(expit(prod_term[0]))  # for the output word
            
    def local_neg_update(self,center_word,context):
        if center_word not in self.word_index.keys():
            return False
        group = center_word[:2]
        t_index = self.word_index[center_word]
        
        if len(self.local_cum_table[group]) <self.local_neg_min: # 群太小不採樣，避免推錯
            return False

        tj = []                            
        local_neg = []
        j = 0
        while len(local_neg) < self.local_negative:
            if j>100:
                print("group num:",self.local_cum_table_map[group])
                break
            # print("in local neg")
            w = self.local_cum_table[group].searchsorted(np.random.randint(self.local_cum_table[group][-1]))
            w = self.local_cum_table_map[group][w]
            if w != t_index and w not in context:
                local_neg.append(w)
                tj.append(0)
            j += 1
        # print("End local neg sampling")
        if len(local_neg)>0:
            self.weight_update(t_index,local_neg,tj,self.lr*self.neg_mtpl / self.size)
        
    def global_update(self,center,context):
        if center not in self.word_index.keys():
            return False
        t_index = self.word_index[center]
        for c in context: #排除重複瀏覽
            if c not in self.word_index.keys():
                return False
            context_index = self.word_index[c]

            # c_pos
            word_indices = [t_index]

            # c_neg 負採樣 
    #        while len(word_indices) < self.negative + 1:
            while len(word_indices) < 2:
                w = self.cum_table.searchsorted(np.random.randint(self.cum_table[-1]))
                if w != t_index and w != context_index:
                    word_indices.append(w)

            

            self.weight_update2(context_index, word_indices,self.lr * self.glb_mtpl / self.size)


            
    def on_epoch_begin(self, model):
        self.timerecord("on_epoch_begin start:")
        self.pre_train(model)   
        print("Epochs",str(self.epochs))
        for j,sentence in enumerate(self.corpus):  #每行資料
            self.progesss_rate(j) # 輸出進度
            sent_len = len(sentence) 
            global_list = []
            if sentence.count('-1')>0:
                g_index_start = sentence.index('-1')
                global_list = sentence[g_index_start+1:]
                browse_list = sentence[:g_index_start]
                sent_len = len(sentence[:g_index_start])

            

            # local負採樣
            if self.local_negative>0:
                for i, word in enumerate(sentence):
                    if word == '-1' or word not in self.word_index.keys():
                        continue

                    group = word[:2]
                    if len(self.local_cum_table[group]) >self.local_neg_min: # 不採樣，避免推離正相關的    
                        context = []
                        for j in range(i - self.window, i + self.window+1):
                            if j != i and j <= sent_len-1 and j >= 0:
                                if sentence[j] not in self.word_index.keys():
                                    continue
                                context.append(self.word_index[sentence[j]])

                        self.local_neg_update(word,context)
                 

            # Global context 主動提案
            if len(global_list)>0:
                browse_list = list(set(browse_list))
                for b in browse_list: 
                    self.global_update(b,global_list)
        print(self.model.running_training_loss)

                    
        self.timerecord("時間:")  
        self.epochs += 1

        
        
        
#宣告物件   


#obj=ClassGenerateTrain("/home/jupyer/derek/591")

#產生訓練txt
#sys.path.append('/home/jupyer/derek/591/')
#from data.config import city
#settings={}
#settings['city'] = city
#x = datetime.datetime.now()
#fileName='591Behavior_'+x.strftime("%Y%m%d")+'.txt'
#csvFileName='591Behavior_'+x.strftime("%Y%m%d")+'.csv'
#obj.generateTxt(pretrain_collection,fileName,otherInfo=settings)

##訓練模型
#gensim_config = {
#     'window_size': 5,    # context window +- center word
#     'size': 32,            # dimensions of word embeddings, also refer to size of hidden layer
#     'epochs': 20,       # number of training epochs
#     'learning_rate': 0.01,   # learning rate
#     'glb_mtpl': 0.05,   # 主動應徵learning rate 倍數(加權)
#     'neg_mtpl': 3,   # 負採樣learning rate 倍數(加權)
#     'local_negative': 5, #local負採樣數
#     'local_neg_min': 1000 #local負採樣 group數量門檻    
#}
#
#model = obj.train_model('basic',fileName,gensim_config)
#儲存模型
#obj.save_model(model,'demo.model')
