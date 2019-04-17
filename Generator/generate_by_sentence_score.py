import heapq
import nltk
import sys
import re
from Generator import text_processor
import Evaluation.evaluator as evaluator
nltk.download('stopwords')

def generate_by_sentence_score(sentence_list):
    full_body = ' '.join(sentence_list)

    # Find weighted frequency of occurrence
    stopwords = nltk.corpus.stopwords.words('english')

    word_frequencies = {}
    for word in full_body.split(' '):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequency = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequency)

    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    summary_sentences = heapq.nlargest(1, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    summary = "%s%s" % (summary[:1].upper(), summary[1:])

    # Strip the sentence of special characters
    summary = re.sub(r"[^a-zA-Z0-9]+", ' ', summary)

    # Truncate sentence
    character_count = 40

    shorter_summary = ''
    if len(summary) > 40:
        for char in summary:
            if summary[character_count] is ' ':
                shorter_summary = summary[0:character_count]
                shorter_summary = shorter_summary + '...'
                break
            character_count = character_count - 1
    else:
        shorter_summary = summary
    return shorter_summary

# Testing Rebeca's code
f = open("sentence_score_results.txt", "w")
data_path = sys.argv[1]
email_text = ""
final_summary = ""
email_text = text_processor.process(data_path, "sentence", False, False)

email_bodies = []
subject_lines = []
email_ids = []
predicted_subject_lines = []

# Load the input emails and their original subject lines
# NOTE: test replacing tab with no space later
for email in sorted(email_text, key=lambda i: i['id']):
    if email['subject'] is not '':
        email_ids.append(email['id'])
        email_bodies.append(email['body'])
        subject_lines.append(email['subject'])

for i, email in enumerate(email_bodies):
    f.write("Email id: " + email_ids[i])
    f.write("Email body: " + ' '.join(email_bodies[i]) + '\n')
    final_summary = generate_by_sentence_score(email_bodies[i])
    f.write("Subject Line: " + final_summary + '\n\n')
    predicted_subject_lines.append(final_summary)

predicted_score = evaluator.subject_line_evaluator(predicted_subject_lines)
f.write("Predicted (generated subject line) score: " + str(predicted_score))