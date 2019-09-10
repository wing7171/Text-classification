#coding=utf-8

import pandas as pd
import numpy as np
import datetime
import os

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.python.ops.rnn import static_rnn
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from tensorflow.python.ops.rnn_cell_impl import DropoutWrapper

from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

path = "../new_data/"
file_train = path + "{data}_set.csv".format(data="train")
file_test = path + "{data}_set.csv".format(data="test")

embedding_size = 16
num_classes = 19
max_document_length = 128 # time steps 55804
num_units = 16 # weights size
learning_rate = 0.05
num_iterators = 150

training_round = 5

def read_data(data_path, chunksize=10):
    chunks = []
    i = 0
    for chunk in pd.read_csv(data_path,iterator=True,sep=',',chunksize=chunksize,usecols=[0,2,3]):
        chunks.append(chunk)
        # if i > 10:
        #     break
        # i += 1
    df = pd.concat(chunks,ignore_index=True)

    return df

def get_vocabulary(data):
    print("---get vocabulary---")
    print(data.shape)
    print(type(data[0][0]), data[0][0])
    # array to list
    # data_list = data
    # 原始文本转词id序列
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    datas = np.array(list(vocab_processor.fit_transform(data)))
    vocab_size = len(vocab_processor.vocabulary_)

    return datas, vocab_size

def save_model(saver, sess):
    path = "../save/model-{}/my-model".format(training_round)
    if not os.path.exists(path):
        os.makedirs(path)
    try:
        saver.save(sess, path)
    except:
        return

if __name__ == '__main__':
    data = read_data(file_train)
    data_x = np.array(data["word_seg"])
    data_y = np.array(data["class"])
    train_x,test_x,train_y, test_y=train_test_split(data_x, data_y, stratify=data_y, test_size=0.2)

    # data_test = read_data(file_test)
    # data_all = pd.concat([data_train], axis=0)

    # 容器，存放输入输出
    datas_placeholder = tf.placeholder(tf.int64, [None, max_document_length],name="X")
    labels_placeholder = tf.placeholder(tf.int64, [None],name="y")
    keep_prob = tf.placeholder(tf.float32,name="keep_prob")

    # 词向量表-随机初始化
    train_x, vocab_size = get_vocabulary(train_x)
    test_x, vocab_size_test = get_vocabulary(test_x)
    print("datas shape:",train_x.shape)
    embeddings = tf.get_variable("embeddings", [vocab_size, embedding_size],
                                 initializer=tf.truncated_normal_initializer)

    # 将词索引号转换为词向量[None, max_document_length] => [None, max_document_length, embedding_size]
    embedded = tf.nn.embedding_lookup(embeddings, datas_placeholder)

    # 转换为LSTM的输入格式，要求是数组，数组的每个元素代表一个Batch下，一个时序的数据（即一个词）
    rnn_input = tf.unstack(embedded, max_document_length, axis=1)

    # 定义LSTM网络结构
    lstm_cell = BasicLSTMCell(num_units=num_units, forget_bias=1.0) # cell
    lstm_cell = DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    rnn_outputs, rnn_states = static_rnn(cell=lstm_cell, inputs=rnn_input, dtype=tf.float32) # network

    # 最后一层
    logits = tf.layers.dense(units=num_classes,inputs=rnn_outputs[-1]) # fully-connected
    pred_labels = tf.arg_max(input=logits,dimension=1) # 概率最大的类别为预测的类别

    # 定义损失函数, logists为网络最后一层输出, labels为真实标签
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                     labels=tf.one_hot(labels_placeholder, num_classes))
    mean_losses = tf.reduce_mean(losses)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_losses)

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        # 初始化变量
        print("---init all variables---")
        print(sess.run(tf.global_variables_initializer()))
        saver = tf.train.Saver()

        feed_dict = {
            "X:0": train_x,
            "y:0": train_y,
            "keep_prob:0": 0.5
        }

        print("---start train---")
        start_time = datetime.datetime.now()
        for step in range(num_iterators):
            print("step={}".format(step))
            what, mean_loss_val = sess.run([optimizer, mean_losses], feed_dict=feed_dict)
            print("mean_loss={} what={}".format(mean_loss_val, what))
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).seconds
        minutes = duration / 60
        seconds = duration % 60
        print("time taken for model training: ", minutes, " m ", seconds, " s")

        save_model(saver,sess)

        # 预测
        # 测试数据没有真实标签，feed-dict的labels？？？
        feed_dict_test = {
            "X:0": test_x,
            "y:0": test_y,
            "keep_prob:0":1.0
        }
        pred_result = sess.run(mean_losses, feed_dict=feed_dict_test)
        print("Cross entropy loss in test sets: ", pred_result)

        pred_labels_val = sess.run(fetches=pred_labels, feed_dict=feed_dict_test)

        try:
            test_y = test_y.reshape(-1,1)
            pred_labels_val = pred_labels_val.reshape(-1,1)
            result = pd.DataFrame([test_y,pred_labels_val],columns=['y_true','y_pred'])
            result.to_csv("result/result-{0}".format(training_round), index=False)
        except:
            print("save result error")

        print("Accuracy on test sets: ", precision_score(y_true=test_y, y_pred=pred_labels_val,average="micro"))
