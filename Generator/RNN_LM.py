from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku
import numpy as np
from Generator import ner, text_processor
from Evaluation import evaluator
import random
import sys

# ********************************************************************
# Title: Language Modelling and Text Generation using LSTMs — Deep Learning for NLP
# Name: Shivam Bansal
# Date: 2018
# Availability: https://medium.com/@shivambansal36/language-modelling-text-generation-using-lstms-deep-learning-for-nlp-ed36b224b275
# ********************************************************************

tokenizer = Tokenizer()


def dataset_preparation(data):
    # basic cleanup
    corpus = data.lower().split("\n")

    # tokenization
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    # create input sequences using list of tokens
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)

    # pad sequences
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    # create predictors and label
    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = ku.to_categorical(label, num_classes=total_words)

    return predictors, label, max_sequence_len, total_words


def create_model(predictors, label, max_sequence_len, total_words):
    model = Sequential()
    model.add(Embedding(total_words, 10, input_length=max_sequence_len - 1))
    model.add(LSTM(150, return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    model.fit(predictors, label, epochs=5, verbose=1, callbacks=[earlystop])
    print(model.summary())
    return model


def generate_text(seed_text, next_words, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text


data = open('C:\\Users\\rebeca\\Desktop\\Spring 2019\\CAP 6640 Computer Understanding of Natural Language\\TermProject\\nlp-email-subjectline-generation\\Generator\\email_bodies.txt').read()


predictors, label, max_sequence_len, total_words = dataset_preparation(data)
model = create_model(predictors, label, max_sequence_len, total_words)

model.save('C:\\Users\\rebeca\\Desktop\\Spring 2019\\CAP 6640 Computer Understanding of Natural Language\\TermProject\\nlp-email-subjectline-generation\\rnn-lm-testing.h5')

#model = load_model('C:\\Users\\rebeca\\Desktop\\Spring 2019\\CAP 6640 Computer Understanding of Natural Language\\TermProject\\nlp-email-subjectline-generation\\rnn-lm-2.h5')


print(generate_text("california", 3, max_sequence_len))

data_path = sys.argv[1]
test_data = text_processor.process(data_path, 'none', False, False, False, 2)

email_ids = []
email_bodies = []

for email in sorted(test_data, key=lambda i:i['id']):
    if email['subject'] is not '':
        email_ids.append(email['id'])
        email_bodies.append(email['body'])

debug = False

if not debug:
    f = open("rnn_lm_results.txt", "w")

generated_subject_lines = []

for i, email in enumerate(email_bodies):
    entities = ner.generate_by_ner(email)
    subject_line = ""
    if len(entities) > 0:
        entity = random.choice(entities)
        index = email.split().index(entity.split()[-1])+1
        text_seed = entity + ' ' + ' '.join(email.split()[index:index+1])
        subject_line = generate_text(text_seed, 3, max_sequence_len)
    else:
        word = random.choice(email.split())
        index = email.split().index(word.split()[-1])+1
        text_seed = word + ' ' + ' '.join(email.split()[index:index+1])
        subject_line = generate_text(text_seed, 3, max_sequence_len)

    generated_subject_lines.append(subject_line)

    if debug:
        print('Email id: ' + email_ids[i])
        print('Email body:' + email + '\n')
        print('Subject line: ' + subject_line + '\n')
        print('\n')
    else:
        f.write('Email id: ' + email_ids[i])
        f.write('Email body:' + email + '\n')
        f.write('Subject line: ' + str(subject_line) + '\n')
        f.write('\n')

predicted_score = evaluator.subject_line_evaluator(generated_subject_lines)

if debug:
    print('Predicted (generated subject line) score: ' + str(predicted_score) + '\n')
else:
    f.write('Predicted (generated subject line) score: ' + str(predicted_score) + '\n')
    f.close()
