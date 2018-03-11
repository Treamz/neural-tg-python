# -*- coding: utf-8 -*-
import sys
import telebot
import numpy as np
import pickle
import re
from Stemmer import Stemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


API_TOKEN = '503561866:AAGgyO922mNjLAC2j_wcqnltSuvo6K1rtpg'

bot = telebot.TeleBot(API_TOKEN)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# очистка текста с помощью regexp приведение слов в инфинитив и нижний регистр, замена цифр
def text_cleaner(text):
    text = text.lower() # приведение в lowercase 
    stemmer = Stemmer('russian')
    text = ' '.join( stemmer.stemWords( text.split() ) ) 
    text = re.sub( r'\b\d+\b', ' digit ', text ) # замена цифр 
    return  text 


# - - - - - - - - - - - - - - - - - - - - - - - - -
# загрузка данных из файла 1.txt
#

def load_data():   
    data = { 'text':[],'tag':[] }
    for line in open('1.txt', encoding='utf8'):
        if(not('#' in line)):
            row = line.split("@") 
            data['text'] += [row[0]]
            data['tag'] += [row[1]]
    return data



	
# - - - - - - - - - - - - - - - - - - - - - - - - -
# Обучение нейросети

def train_test_split( data, validation_split = 0.1):
    sz = len(data['text'])
    indices = np.arange(sz)
    np.random.shuffle(indices)

    X = [ data['text'][i] for i in indices ]
    Y = [ data['tag'][i] for i in indices ]
    nb_validation_samples = int( validation_split * sz )

    return { 
        'train': { 'x': X[:-nb_validation_samples], 'y': Y[:-nb_validation_samples]  },
        'test': { 'x': X[-nb_validation_samples:], 'y': Y[-nb_validation_samples:]  }
    }


# - - - - - - - - - - - - - - - - - - - - 
def openai(z,message):
    data = load_data()
    D = train_test_split( data )
    text_clf = Pipeline([
                    ('tfidf', TfidfVectorizer()),
                    ('clf', SGDClassifier(loss='hinge')),
                    ])
    text_clf.fit(D['train']['x'], D['train']['y'])
    predicted = text_clf.predict( D['train']['x'] )
	
	# Начало тестирования программы
    
    #z=input("Введите вопрос без знака вопроса на конце: ")
    #z="Кто полетел в космо"
    zz=[]
    zz.append(z)

    predicted = text_clf.predict( zz )
    
    print(predicted[0])
    bot.reply_to(message, (predicted[0]))

    # Handle '/start' and '/help'

@bot.message_handler(commands=['help', 'start'])
def send_welcome(message):
    bot.reply_to(message, "Привет")

@bot.message_handler(func=lambda message: True)
def send_message(message):
    q=message.text
    openai(q,message)

bot.polling()

