from Generator import text_processor
import sys
from collections import defaultdict
from nltk import trigrams
import random
from Generator import ner

DEBUG = True

data_path = sys.argv[1]

if DEBUG:
    print('Processing email dataset...')

# Process the email dataset to clean, tokenize, and remove stop words (overly common words) from the email body
data = text_processor.process(data_path, 'word', False, True, False)

# These have been pre-processed and cleaned
# Email bodies do not have stop words but subject lines do
email_bodies = []
subject_lines = []
email_ids = []

# Load the input emails and their original subject lines
# NOTE: test replacing tab with no space later
for email in sorted(data, key=lambda i: i['id']):
    if email['subject'] is not '':
        email_ids.append(email['id'])
        email_bodies.append(email['body'])
        subject_lines.append(email['subject'])

if DEBUG:
    print('Processed', len(email_bodies), 'emails')

for email in data:
    model = defaultdict(lambda: defaultdict(lambda: 0))

    for sentence in email['body']:
        for w1, w2, w3 in trigrams(sentence.split(' '), pad_right=True, pad_left=True):
            model[(w1, w2)][w3] += 1

    for w1_w2 in model:
        total_count = float(sum(model[w1_w2].values()))
        for w3 in model[w1_w2]:
            model[w1_w2][w3] /= total_count


test_data = text_processor.process(data_path, 'none', False, False, False, 2)


for email in sorted(test_data, key=lambda i: i['id'])[0:10]:
    if email['subject'] is not '':

        print('\n\n', email['body'])

        # Get top entity for this email
        entities = ner.generate_by_ner(email['body'])
        print(entities)

        # text = [None, None]
        # prob = 1.0  # <- Init probability
        #
        # sentence_finished = False
        #
        # while not sentence_finished:
        #     r = random.random()
        #     accumulator = .0
        #
        #     for word in model[tuple(text[-2:])].keys():
        #         accumulator += model[tuple(text[-2:])][word]
        #
        #         if accumulator >= r:
        #             prob *= model[tuple(text[-2:])][
        #                 word]  # <- Update the probability with the conditional probability of the new word
        #             text.append(word)
        #             break
        #
        #     if text[-2:] == [None, None]:
        #         sentence_finished = True

        # entities = []

        if not len(entities):
            text = [None, None]
        else:
            text = [None, 'email']#entities.pop()]
        prob = 1.0  # <- Init probability

        sentence_finished = False
        attempts = 0

        while not sentence_finished:
            attempts += 1

            if attempts >= 20:
                print(text, "no workie")
                if not len(entities):
                    print("bad email >:(, forfeit")
                    break
                text = [None, entities.pop()]

            r = random.random()
            accumulator = .0
            print(text)

            for word in model[tuple(text[-2:])].keys():
                accumulator += model[tuple(text[-2:])][word]

                if accumulator >= r:
                    prob *= model[tuple(text[-2:])][
                        word]  # <- Update the probability with the conditional probability of the new word
                    text.append(word)
                    break

            if len(text) > 4:
                # if prob < 0.001:
                #     text.pop()
                # else:
                sentence_finished = True

        print("Probability of text=", prob)  # <- Print the probability of the text
        print('Generated subject line: ', ' '.join([t for t in text if t]))
