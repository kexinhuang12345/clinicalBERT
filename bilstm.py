import sys, os, re, csv, codecs, numpy as np, pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import initializers, regularizers, constraints, optimizers, layers

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
from sklearn.metrics import average_precision_score

%matplotlib inline

import gensim
m = gensim.models.KeyedVectors.load('word2vec.model')
weights = (m[m.wv.vocab])

max_words_count = 44082
embedding_size = 100
max_words_length = 318

def plot_auc(x,y):
    y_pred = model.predict(x).ravel()
    y_pred_s = [1 if i else 0 for i in (y_pred >= 0.5)]
    print (sum(y_pred_s == y) / len(y))
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y, y_pred)
    auc_keras = auc(fpr_keras, tpr_keras)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Test (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    return y_pred

def vote_score(df, score):
    df['pred_score'] = score
    df_sort = df.sort_values(by=['ID'])
    temp = (df_sort.groupby(['ID'])['pred_score'].agg(max)+df_sort.groupby(['ID'])['pred_score'].agg(sum)/2)/(1+df_sort.groupby(['ID'])['pred_score'].agg(len)/2)
    x = df_sort.groupby(['ID'])['Label'].agg(np.min).values
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(x, temp.values)
    auc_keras = auc(fpr_keras, tpr_keras)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Val (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    print (auc_keras)
    return fpr_keras, tpr_keras


def vote_pr_curve(df, score):
    df['pred_score'] = score
    df_sort = df.sort_values(by=['ID'])
    temp = (df_sort.groupby(['ID'])['pred_score'].agg(max)+df_sort.groupby(['ID'])['pred_score'].agg(sum)/2)/(1+df_sort.groupby(['ID'])['pred_score'].agg(len)/2)
    y = df_sort.groupby(['ID'])['Label'].agg(np.min).values
    
    precision, recall, thres = precision_recall_curve(y, temp)
    pr_thres = pd.DataFrame(data =  list(zip(precision, recall, thres)), columns = ['prec','recall','thres'])
    vote_df = pd.DataFrame(data =  list(zip(temp, y)), columns = ['score','label'])
    
    pr_curve_plot(y, temp)
    
    return pr_thres, vote_df, precision, recall


def pr_curve_plot(y, y_score):
    precision, recall, _ = precision_recall_curve(y, y_score)
    area = auc(recall,precision)
    print(area)
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    average_precision1 = average_precision_score(y, y_score)
    print(average_precision1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(
              area))

i=2

data_path = './good_datasets/fold'+str(i)+'/3days/'
train_path = data_path+'train_snippets.csv'
val_path = data_path+'val_snippets.csv'
test_path = data_path+'test_snippets.csv'
df_train=pd.read_csv(train_path)
df_val=pd.read_csv(val_path)
df_test=pd.read_csv(test_path)
df_test2 = pd.read_csv('./good_datasets/fold'+str(i)+'/2days/test_snippets.csv')

sent_train = df_train['TEXT']
y_train = df_train['Label']
sent_val = df_val['TEXT']
y_val = df_val['Label']
sent_test = df_test['TEXT']
y_test = df_test['Label']
sent_test2 = df_test2['TEXT']
y_test2 = df_test2['Label']

tokenizer=Tokenizer(num_words=max_words_count)
tokenizer.fit_on_texts(sent_train)
tokens_train = tokenizer.texts_to_sequences(sent_train)
tokens_val = tokenizer.texts_to_sequences(sent_val)
tokens_test = tokenizer.texts_to_sequences(sent_test)
tokens_test2 = tokenizer.texts_to_sequences(sent_test2)

x_train=pad_sequences(tokens_train,maxlen=max_words_length)
x_val=pad_sequences(tokens_val,maxlen = max_words_length)
x_test=pad_sequences(tokens_test,maxlen=max_words_length)
x_test2=pad_sequences(tokens_test2,maxlen=max_words_length)

word_idx=tokenizer.word_index
embed_dict = dict(zip(list(m.wv.vocab),list(m[m.wv.vocab])))
all_embs = np.stack(embed_dict.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embedding_matrix = np.random.normal(emb_mean, emb_std, (max_words_count, embedding_size))
for word,j in word_idx.items():
    if j < max_words_count:
        vec_temp=embed_dict.get(word)
        if vec_temp is not None:
            embedding_matrix[j]=vec_temp

inp=Input(shape=(max_words_length,))
x=Embedding(max_words_count,embedding_size,weights=[embedding_matrix])(inp)
x=Bidirectional(LSTM(embedding_size,return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)
x=GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mae','accuracy'])

save_path = './bilstm_models/fold'+str(i)+'_models/best_model_early_good.h5'
callbacks = [EarlyStopping(monitor='val_loss', patience=2),
         ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=True)]

history = model.fit(x_train, y_train, batch_size=64, epochs=3, callbacks=callbacks, verbose=1, validation_data=(x_val, y_val))
