import heapq
import re
import nltk
nltk.download('stopwords')

def sentence_score(article_text):
    # Removing square brackets and extra spaces
    article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
    article_text = re.sub(r'\s+', ' ', article_text)

    # Removing special characters and digits
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

    # Converting text to sentences using sentence tokenization.
    sentence_list = nltk.sent_tokenize(article_text)

    # Find weighted frequency of occurence
    stopwords = nltk.corpus.stopwords.words('english')

    word_frequencies = {}
    for word in nltk.word_tokenize(formatted_article_text):
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

    shorter_summary = ''

    # Truncate sentence
    sentence_summary = len(summary)

    if sentence_summary > 35:
        shorter_summary = summary[0:35]
        i = 0
        while summary[35 + i] != ' ':
            shorter_summary = shorter_summary + summary[35 + i]
            i += 1

    shorter_summary = shorter_summary + '...'
    print(shorter_summary)
