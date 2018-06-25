from underthesea import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import model_from_json
import numpy as np
import keras 

stemmer = LancasterStemmer()

file = open('vietnamese-stopwords.txt', 'r') 
stopwords = file.readlines()

import json
with open('data.json') as json_data:
    intents = json.load(json_data)

words = []
documents = []
classes = []
print(intents)
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = word_tokenize(pattern)
        words.extend(w)
        print(intent['tag'])
        documents.append((w,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
words = [stemmer.stem(w.lower()) for w in words if w not in stopwords]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

training = []
output = []
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])
training = np.array(training)

# create train and test lists
train_x = np.array(training[:,0])
train_y = np.array(training[:,1])
x_train = []
for train in train_x:
    train = np.array(train)
    x_train.append(train)
x_train = np.array(x_train)
print(x_train.shape)

y_train = []
for train in train_y:
    train = np.array(train)
    y_train.append(train)
y_train = np.array(y_train)
print(y_train[1])

x = Input(shape=(x_train.shape[1],))
h = Dense(8, activation='relu')(x)
h = Dense(8, activation='relu')(h)
h = Dense(y_train.shape[1],activation = 'sigmoid')(h)
model = Model(x,h)
model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam())
model.fit(x_train,y_train,epochs=1000,verbose=1)
score = model.evaluate(x_train,y_train,verbose=0)
model_json = model.to_json()

with open('model.json','w') as json_file:
    json_file.write(model_json)
model.save_weights('model.h5')

def clean_data(data):
   sentences = word_tokenize(data)
   sentences = [stemmer.stem(w.lower()) for w in sentences if w not in stopwords] 
   return sentences
def bow(sentences, words, show = False):
    split = clean_data(sentences)
    bag = [0]*len(words)  #tạo một mảng bag
    for s in split:
        for i,w in enumerate(words): 
            if w == s:
                bag[i] = 1
    bag = np.reshape(np.array(bag),(1,len(words)))
    return bag
def classify(sentence):
    print(bow(sentence, words).shape)
    results = model.predict( bow(sentence, words))
    print(classes[np.argmax(results)])
    return results
classify('Tớ thích cậu')
