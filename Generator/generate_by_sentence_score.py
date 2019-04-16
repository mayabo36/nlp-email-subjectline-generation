import heapq
import nltk
import sys
from Generator import text_processor
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

    # Truncate sentence
    sentence_summary = len(summary)

    if sentence_summary > 35:
        shorter_summary = summary[0:35]
        i = 0
        while summary[35 + i] != ' ':
            shorter_summary = shorter_summary + summary[35 + i]
            i += 1
        shorter_summary = shorter_summary + '...'
    else:
        shorter_summary = summary
    return shorter_summary


# Testing Rebeca's code
data_path = sys.argv[1]
email_text = ""
final_summary = ""
email_text = text_processor.process(data_path, "sentence", False, False)

for j in range(0, 10):
    text = email_text[j]['body']
    final_summary = generate_by_sentence_score(text)
    print(final_summary)
