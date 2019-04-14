from Generator import text_processor
import sys
from nltk.corpus import reuters
from collections import Counter, defaultdict
from nltk import bigrams, trigrams
import random

data_path = sys.argv[1]

# Process the email dataset to clean, tokenize, and remove stop words (overly common words) from the email body
data = text_processor.process(data_path, 'sentence', False, False, True)

authors = list(data)

for author in authors[0:1]:
    print(author)

    model = defaultdict(lambda: defaultdict(lambda: 0))

    for email in data[author]:
        for sentence in email['body']:
            for w1, w2, w3 in trigrams(sentence.split(' '), pad_right=True, pad_left=True):
                model[(w1, w2)][w3] += 1

    for w1_w2 in model:
        total_count = float(sum(model[w1_w2].values()))
        for w3 in model[w1_w2]:
            model[w1_w2][w3] /= total_count

    text = [None, None]
    prob = 1.0  # <- Init probability

    sentence_finished = False

    while not sentence_finished:
        r = random.random()
        accumulator = .0

        for word in model[tuple(text[-2:])].keys():
            accumulator += model[tuple(text[-2:])][word]

            if accumulator >= r:
                prob *= model[tuple(text[-2:])][
                    word]  # <- Update the probability with the conditional probability of the new word
                text.append(word)
                break

        if text[-2:] == [None, None]:
            sentence_finished = True

    print("Probability of text=", prob) # <- Print the probability of the text
    print(' '.join([t for t in text if t]))