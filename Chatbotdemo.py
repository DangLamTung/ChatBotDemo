from underthesea import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from keras.models import model_from_json
import numpy as np
import keras 
import random
stemmer = LancasterStemmer()

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
#load data

file = open('vietnamese-stopwords.txt', 'r') 
stopwords = file.readlines()

words = []
documents = []
classes = []

import json
with open('data.json') as json_data:
    intents = json.load(json_data)

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

def clean_data(data):
   sentences = word_tokenize(data)
   sentences = [stemmer.stem(w.lower()) for w in sentences if w not in stopwords] #pre process
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
    #print(classes[np.argmax(results)])
    return classes[np.argmax(results)]
#classify('tớ lên kstn rồi')
def response(sentence, show_details=False):
    results = classify(sentence)
    print(results)
    for intent in intents['intents']:
        if results == intent['tag']:
            print(random.choice(intent['responses']))

response('tớ thích cậu')
