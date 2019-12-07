from tensorflow import keras
import tensorflow as tf
import numpy as np
import os
import copy
import re
import nltk
import pandas as pd


class Data():
    def __init__(self,info,label,shuffle=False):
        assert len(info) == len(label)
        self.info = np.array(info)
        self.label = np.array(label)
        self.batch_id = 0
        self.shuffle = shuffle

    def __len__(self):
        return len(self.label)

    def confuse(self):
        info = self.info
        label = self.label
        info = np.array(info)
        label = np.array(label)
        index = np.arange(len(info))
        np.random.shuffle(index)
        info,label = info[index],label[index]
        self.info,self.label = info,label

    def next(self,batch_size):
        if self.__len__() == self.batch_id:
            self.batch_id = 0
            if self.shuffle:
                self.confuse()
        batch_end = min(len(self.label),batch_size+self.batch_id)
        info = self.info[self.batch_id:batch_end]
        label = self.label[self.batch_id:batch_end]
        self.batch_id = batch_end
        return info,label


def load_wv(path:str):
    assert os.path.isfile(path)
    w2v = {}

    with open(path,'r') as file:

        for string in file.readlines():
            word = string.split()[0]
            vec = string.split()[1:]
            vec = np.asarray(vec, dtype='float32')
            w2v[word] = vec
    matrix = np.zeros([len(w2v)+1,len(w2v[word])])
    word_index = {}
    index_word = {}
    for i,word in enumerate(w2v.keys()):
        matrix[i+1] = w2v[word]
        word_index[word] = i+1
        index_word[i] = word
    return matrix,word_index,index_word


def padding(sentences:list,sen_max_len,word_max_len):
    '''
    the sentence shape is [the amount of samples,sentence size,word size] and the sentences is related with the index of word

    :param sentence:
    :param max_len:
    :return:
    '''
    ret_list = []
    for sentence in sentences:

        if len(sentence) > sen_max_len:
            temp = copy.deepcopy(sentence[:sen_max_len])
        elif len(sentence) < sen_max_len:
            temp = copy.deepcopy(sentence)
            temp.extend([[0]]*(sen_max_len - len(sentence)))
        else:
            temp = copy.deepcopy(sentence)
        word_list = []
        for words in temp:

            if len(words) > word_max_len:
                word = copy.deepcopy(words[:word_max_len])
            elif len(words) < word_max_len:
                word = copy.deepcopy(words)
                word.extend([0]*(word_max_len - len(words)))
            else:
                word = copy.deepcopy(words)
            word_list.append(word)
        ret_list.append(word_list)
    return ret_list


def token(doc,word_tokenizer,sent_tokenizer):
    """
    :param doc:the doc is just similar with [_,_,_,_]
    :param reg_spilt:
    :param tokenizer:
    :return:
    """
    sen_token = lambda x:list(sent_tokenizer(x))
    sentences = list(map(sen_token,doc))
    getWord = lambda sentence:word_tokenizer(sentence.lower())
    sentencesDir = lambda sentence:list(map(getWord,sentence))
    sentences = list(map(sentencesDir,sentences))
    return sentences


def getIndex(sentences,word_index):
    """
    :param sentences:
    :param word_index:
    :return:
    """
    getWordIndex = lambda word:word_index[word] if word in word_index.keys() else 0
    getSubSentencIndex = lambda sentence:list(map(getWordIndex,sentence))
    getSentence = lambda x:list(map(getSubSentencIndex,x))
    sentences_ = list(map(getSentence,sentences))
    return sentences_


class HierarchicalAttentionNetwork(tf.keras.Model):
    def __init__(self,attention_dim):
        super(HierarchicalAttentionNetwork,self).__init__()
        self.attention_dim = attention_dim
        self.dense = tf.keras.layers.Dense(attention_dim,activation=tf.nn.tanh,use_bias=True)
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, x):
        uit = self.dense(x)
        ait = tf.nn.softmax(tf.squeeze(self.dense2(uit),-1))
        weight_input = x * tf.expand_dims(ait,-1)
        output = tf.reduce_sum(weight_input,1)
        return output


class Model(keras.Model):

    def __init__(self,num_classes,units,matrix,max_word_len,max_seq_len):
        super(Model,self).__init__()
        self.sen_gru = tf.keras.layers.GRU(
            units,return_sequences=True,time_major=True,recurrent_dropout=0.2
        )
        self.num_class = num_classes
        self.embedding_layer = tf.keras.layers.Embedding(
            input_dim=matrix.shape[0],
            output_dim=matrix.shape[1],
            weights=[matrix],
            input_length=max_seq_len,
            trainable=True
        )

        self.gru_unit = units
        self.doc_gru = tf.keras.layers.GRU(
            units,return_sequences=True,time_major=True,recurrent_dropout=0.2
        )
        self.sen_han = HierarchicalAttentionNetwork(max_word_len)
        self.doc_han = HierarchicalAttentionNetwork(max_seq_len)
        self.dense = tf.keras.layers.Dense(units=units)
        self.classes = tf.keras.layers.Dense(units=num_classes,activation=tf.nn.softmax)

    def call(self,sentences):
        """

        :param sentences: shape = [batch_size,sen_size,words]

        :return:
        """
        x = tf.transpose(sentences,perm=[1,0,2])
        '''
        [sentences,sen_size,words]

        '''

        embedding_rep = self.embedding_layer(x)
        '''
        the shape of embedding_rep is [batch_size,sentences,words,24]
        '''
        doc_encoder = []
        for rep in tf.unstack(embedding_rep):
            '''
            rep:[sentences,words,24]
            '''
            zc = self.sen_gru(rep)
            sen_rep = self.sen_han(zc)
            doc_encoder.append(sen_rep)

        doc_encoder = tf.stack(doc_encoder)
        doc_encoder = tf.transpose(doc_encoder,[1,0,2])
        output = self.doc_gru(doc_encoder)
        output = self.doc_han(output)
        output = self.dense(output)
        output = self.classes(output)
        return output

    def predict(self,sentence):
        x = tf.expand_dims(sentence,0)
        y = self.call(x)
        index = tf.argmax(y,1)
        return index.numpy().tolist()[0]

    def predict_prob(self,sentence):
        x = tf.expand_dims(sentence, 0)
        y = self.call(x)
        return y.numpy()

    def predict_batch(self,sentences):
        y = self.call(sentences)

        return tf.argmax(y,1).numpy()

batch_size = 128


def save(model,path):
    model.save(path)


def load_model(path):
    model = tf.keras.models.load_model(path)
    return model


def evaluate(model,data:Data):
    info,target_label = data.info,data.label
    batch_id = 0
    length = len(data)
    predict_label = []
    while batch_id != length:
        batch_end = min(batch_id+64,length)
        info_ = info[batch_id:batch_end]
        out = model(info_)
        predict_label.extend(tf.argmax(out,1).numpy().tolist())
        batch_id = batch_end
    predict_label = np.array(predict_label)
    accuracy = sum(predict_label == target_label)/length
    return accuracy


def train(model,test_data,train_data):
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)

    # with tf.device("/gpu:0"):
    for epoch in range(100):

        for i in range(len(train_data)//batch_size):
            with tf.GradientTape() as tape:
                x, y = train_data.next(batch_size)
                x = tf.convert_to_tensor(x)
                y = tf.convert_to_tensor(y)
                y = tf.one_hot(y,model.num_class)
                output = model(x)
                '''
                这里都是one——hot表示
                '''
                loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, output, from_logits=True))
                correct = tf.equal(tf.argmax(output,1),tf.argmax(y,1))
                accuray = tf.reduce_mean(tf.cast(correct,tf.float32))

                print("epoch ",epoch," iter ",i,"loss is ",loss.numpy()," accuracy is ",accuray.numpy())

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        accuray = evaluate(model,train_data)
        print("after epoch ",epoch,"the accuracy of train data is ",accuray)
        accuray = evaluate(model, test_data)
        print("after epoch ",epoch,"the accuracy of test data is ",accuray)


if __name__ == '__main__':
    wv_path = 'glove.6B.100d.txt'
    matrix, word_index, index_word = load_wv(wv_path)
    sent_tokenizer = nltk.sent_tokenize
    word_tokenizer = nltk.word_tokenize
    train_path = r'labeledTrainData.tsv'
    train_data = pd.read_table(train_path,sep='\t')
    doc,label = train_data["review"],train_data["sentiment"]
    doc = doc.tolist()
    label = label.tolist()
    sentences = token(doc=doc,word_tokenizer=word_tokenizer,sent_tokenizer=sent_tokenizer)
    sentencesIndex = getIndex(sentences,word_index)
    sen_max_len, word_max_len = 8,30
    sentencesIndex = padding(sentencesIndex,sen_max_len,word_max_len)
    num_samples = len(sentencesIndex)
    index = np.arange(num_samples)
    np.random.shuffle(index)
    test_x = np.array(sentencesIndex)[index][:num_samples//5].tolist()
    test_y = np.array(label)[index][:num_samples//5].tolist()
    train_x = np.array(sentencesIndex)[index][num_samples//5:].tolist()
    train_y = np.array(label)[index][num_samples//5:].tolist()
    del sentencesIndex,sentences,label
    train_data = Data(train_x,train_y,shuffle=True)
    test_data = Data(test_x,test_y)
    model = Model(num_classes=2,units=108,matrix=matrix,max_word_len=word_max_len,max_seq_len=sen_max_len)
    train(model,test_data,train_data)
    save(model)