#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @Date    : 2019-03-14
  @Author  : Kaka
  @File    : BILSTM_CRF_ZH_NER.py
  @Software: PyCharm
  @Desc    : 基于bilstm和crf的中文命名实体识别
"""
import os, sys, logging
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.utils import np_utils, plot_model
from keras_contrib import metrics
from keras_contrib import losses
from keras.models import Sequential, load_model, save_model
from keras.layers import Embedding, Bidirectional, LSTM
from keras_contrib.layers import CRF
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import pickle

class data_prossor(object):
    '''数据处理'''
    path = 'data/'
    maxlen = 100

    @staticmethod
    def load_data():
        ptrain = os.path.join(data_prossor.path, 'train.txt')
        pval = os.path.join(data_prossor.path, 'test.txt')

        with open(ptrain, 'r', ) as fd, open(pval, 'r', ) as fdv:
            train, vocab, labels = [], [], []
            for ps in fd.read().strip().split('\n\n'):
                p = []
                for s in ps.strip().split('\n'):
                    wt = s.split()
                    p.append(wt)
                    vocab.append(wt[0])
                    labels.append(wt[1])
                train.append(p)
            val = [[s.split() for s in p.split('\n')] for p in fdv.read().split('\n\n')]
            vocab = dict([(w, i) for i, w in enumerate(list(set(vocab)))])
            if not os.path.exists(os.path.join(data_prossor.path,'vocab.pkl')):
                pickle.dump(vocab,open(os.path.join(data_prossor.path,'vocab.pkl'),'wb'))
            else:
                vocab = pickle.load(open(os.path.join(data_prossor.path,'vocab.pkl'),'rb'))
        labels = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG']
        train_x, train_y = data_prossor.econde_data(train, vocab, labels)
        val_x, val_y = data_prossor.econde_data(val, vocab, labels)
        return train_x, train_y, val_x, val_y, vocab, labels

    @staticmethod
    def econde_data(data, vocab, labels, onehot=False):
        x = [[vocab.get(it[0], 1) for it in item if len(it)==2 ] for item in data]
        y = [[labels.index(it[1]) for it in item if len(it)==2 ] for item in data]
        x = pad_sequences(x, data_prossor.maxlen)
        y = pad_sequences(y, data_prossor.maxlen, value=-1)

        if onehot:
            y = np_utils.to_categorical(y, len(labels))
        else:
            y = y.reshape((y.shape[0], y.shape[1], 1))

        print(x.shape, y.shape)
        return x, y

    @staticmethod
    def parse_text(text, vocab):
        ttext = [vocab.get(w, -1) for w in list(text)]
        ttext = pad_sequences([ttext], data_prossor.maxlen)
        print(ttext)
        return ttext

class kaka_play(object):
    def __init__(self, labels, vocab):
        self.max_len = 100
        self.max_wnum = len(vocab)
        self.embd_dim = 200
        self.drop_late = 0.5
        self.batch_size = 64
        self.epochs = 100
        self.lstmunit = 100
        self.label_num = len(labels)
        self.model_path = 'model/'

    def model_create(self):
        model = Sequential()
        model.add(Embedding(self.max_wnum, self.embd_dim, mask_zero=True))
        model.add(Bidirectional(LSTM(self.lstmunit, return_sequences=True,dropout=self.drop_late), merge_mode='concat',))
        crf = CRF(units=self.label_num, sparse_target=True)
        model.add(crf)
        model.compile(optimizer='adam', loss=losses.crf_loss, metrics=[metrics.crf_accuracy])
        plot_model(model, to_file="{}{}".format(self.model_path, 'bilstm-crf.png'))
        print(model.summary())
        return model

    def model_fit(self, model, train, val):
        early_stopping = EarlyStopping(monitor='crf_loss',patience=5, verbose=1, )
        model_checkpoint = ModelCheckpoint(filepath=self.model_path + 'bilstm_crf.h5', save_best_only=True,
                                           verbose=1)
        tensor_board = TensorBoard(log_dir='./logs', batch_size=self.batch_size, write_images=True)

        # 在这里使用校验数据会出现val_loss=nan的问题，小批量数据不会出现，采用softmax也不会出现
        # model.fit(train[0], train[1], validation_data=[val[0], val[1]],
        #           batch_size=self.batch_size, epochs=self.epochs,
        #           shuffle=True, callbacks=[early_stopping, model_checkpoint, tensor_board])
        model.fit(train[0], train[1], validation_split=0.2,
                  batch_size=self.batch_size, epochs=self.epochs,
                  shuffle=True, callbacks=[early_stopping, model_checkpoint, tensor_board])


    def model_predict(self, model, text, vocab):
        ttext = data_prossor.parse_text(text, vocab)
        result = model.predict_classes(ttext)
        result = [[labels[i] for i in p] for p in result]

        per, org, loc = '', '', ''
        for s, t in zip(list(text), result[0][-len(text):]):
            print(s,t)
            if t in ('B-PER', 'I-PER'):
                per += ' ' + s if (t == 'B-PER') else s
            if t in ('B-ORG', 'I-ORG'):
                org += ' ' + s if (t == 'B-ORG') else s
            if t in ('B-LOC', 'I-LOC'):
                loc += ' ' + s if (t == 'B-LOC') else s
        ner = 'PER:' + per + '\nORG:' + org + '\nLOC：' + loc
        return ner


if __name__ == '__main__':
    train_x, train_y, val_x, val_y, vocab, labels = data_prossor.load_data()
    kaka = kaka_play(labels, vocab)
    model = kaka.model_create()
    if os.path.exists(kaka.model_path + 'bilstm_crf.h5'):
        model.load_weights(kaka.model_path + 'bilstm_crf.h5')
        # if you want  fit the model more epochs
        # kaka.model_fit(model, (train_x, train_y), (val_x, val_y))
    else:
        kaka.model_fit(model, (train_x, train_y), (val_x, val_y))

    text = '中华人民共和国国务院总理周恩来在外交部长陈毅，副部长王东的陪同下，连续访问了埃塞俄比亚等非洲10国以及阿尔巴尼亚'
    ner = kaka.model_predict(model, text, vocab)
    print(ner)
    text = '中共中央批准，樊大志同志任中央纪委国家监委驻中国证券监督管理委员会纪检监察组组长，免去王会民同志的中央纪委国家监委驻中国证券监督管理委员会纪检监察组组长职务'
    ner = kaka.model_predict(model, text, vocab)
    print(ner)
    text = '樊大志同志1987年8月参加工作。先后在东北财经大学、北京国际信托投资公司、北京市境外融投资管理中心、北京市国有资产经营有限责任公司、北京证券有限责任公司、北京首都创业集团有限公司、华夏银行股份有限公司工作。'
    ner = kaka.model_predict(model, text, vocab)
    print(ner)