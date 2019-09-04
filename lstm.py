#coding=utf-8

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.python.ops.rnn import static_rnn
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell

path = "../new_data/"
file_train = path + "{data}_set.csv".format(data="train")
file_test = path + "{data}_set.csv".format(data="test")

def read_data(data_path, chunksize=10):
    chunks = []
    i = 0
    for chunk in pd.read_csv(data_path,iterator=True,sep=',',chunksize=chunksize):
        chunks.append(chunk)
        # if i >= 2:
        #     break
        # i += 1

    df = pd.concat(chunks,ignore_index=True)

    return df

embedding_size = 32
num_classes = 19
max_document_length = 512 # time steps 55804
num_units = 32 # weights size
learning_rate = 0.1
num_iterators = 100

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
    print("<{0}>".format("transformed data"))
    print(datas)
    print("<{0}> {1}".format("vocab_size",vocab_size))

    return datas, vocab_size


if __name__ == '__main__':
    data_train = read_data(file_train)
    # data_test = read_data(file_test)
    data_all = pd.concat([data_train], axis=0)
    # data_arr = np.array(data_all)
    data_article = np.array(data_all["article"])
    data_word = np.array(data_all["word_seg"])

    print(data_train)

    # 容器，存放输入输出
    datas_placeholder = tf.placeholder(tf.int64, [None, max_document_length],name="X")
    labels_placeholder = tf.placeholder(tf.int64, [None],name="y")

    # 词向量表-随机初始化
    datas, vocab_size = get_vocabulary(data_article)
    labels = np.array(data_all["class"])
    print("datas shape:",datas.shape)
    embeddings = tf.get_variable("embeddings", [vocab_size, embedding_size],
                                 initializer=tf.truncated_normal_initializer)

    # 将词索引号转换为词向量[None, max_document_length] => [None, max_document_length, embedding_size]
    embedded = tf.nn.embedding_lookup(embeddings, datas_placeholder)
    print("<embedded>", embedded)

    # 转换为LSTM的输入格式，要求是数组，数组的每个元素代表一个Batch下，一个时序的数据（即一个词）
    rnn_input = tf.unstack(embedded, max_document_length, axis=1)
    print("<rnn input>")
    print(rnn_input)
    print("tensor size:{0}".format(len(rnn_input))) # 跟预期不一样???

    # 定义LSTM网络结构
    lstm_cell = BasicLSTMCell(num_units=num_units, forget_bias=1.0) # cell
    rnn_outputs, rnn_states = static_rnn(cell=lstm_cell, inputs=rnn_input, dtype=tf.float32) # network
    print("last outputs:", rnn_outputs[-1])
    print("last states:", rnn_states[-1])

    # 最后一层
    logits = tf.layers.dense(units=num_classes,inputs=rnn_outputs[-1]) # fully-connected
    print("logits(final output):",logits)
    pred_labels = tf.arg_max(input=logits,dimension=1) # 概率最大的类别为预测的类别
    print("predicted labels:",pred_labels)

    # 定义损失函数, logists为网络最后一层输出, labels为真实标签
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                     labels=tf.one_hot(labels_placeholder, num_classes))
    mean_losses = tf.reduce_mean(losses)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_losses)
    print("mean_loss:", mean_losses)
    print("optimizer:", optimizer)

    with tf.Session() as sess:
        # 初始化变量
        print("---init all variables---")
        sess.run(tf.global_variables_initializer())

        feed_dict = {
            "X:0": datas,
            "y:0": labels
        }

        print("---start train---")
        for step in range(num_iterators):
            print("step={}".format(step))
            what, mean_loss_val = sess.run([optimizer, mean_losses], feed_dict=feed_dict)
            # if step % 10 == 0:
            print("mean_loss={} what={}".format(mean_loss_val, what))

        # 预测
        # 测试数据没有真实标签，feed-dict的labels？？？
        # pred_labels_val = sess.run(fetches=pred_labels,feed_dict=feed_dict)
        #
        # for i, text in enumerate(data_test):
        #     label = pred_labels_val[i]
        #     print("{} => {}".format(text, label))