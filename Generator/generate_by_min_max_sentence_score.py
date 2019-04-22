# ********************************************************************
# Title: The Glowing Python: 
# Name: Text summarization with NLTK 
# Date: Wednesday, September 24, 2014
# Availability: https://glowingpython.blogspot.com/2014/09/text-summarization-with-nltk.html
# ********************************************************************
import nltk
import text_processor
import evaluator 
import sys
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from string import punctuation
from heapq import nlargest

MAX_CHAR = 40

class FrequencySummarizer:
  def __init__(self, min_cut=0.0, max_cut=0.0):
    """
     Initilize the text summarizer.
     Words that have a frequency term lower than min_cut 
     or higer than max_cut will be ignored.
    """
    self._min_cut = min_cut
    self._max_cut = max_cut 
    self._stopwords = set(stopwords.words('english') + list(punctuation))

  def _compute_frequencies(self, word_sent):
    """ 
      Compute the frequency of each of word.
      Input: 
       word_sent, a list of sentences already tokenized.
      Output: 
       freq, a dictionary where freq[w] is the frequency of w.
    """
    freq = defaultdict(int)
    for s in word_sent:
      for word in s:
        if word not in self._stopwords:
          freq[word] += 1
    # frequencies normalization and fitering
    m = float(max(freq.values()))
    for w in freq.items():#.keys():
      freq[w] = freq[w]/m
      print freq[w]
      if freq[w] >= self._max_cut or freq[w] <= self._min_cut:
        del freq[w]
    return freq

  def summarize(self, text, n):
    """
      Return a list of n sentences 
      which represent the summary of text.
    """
    sents = sent_tokenize(text)
    #assert n <= len(sents)
    word_sent = [word_tokenize(s.lower()) for s in sents]
    self._freq = self._compute_frequencies(word_sent)
    ranking = defaultdict(int)
    for i,sent in enumerate(word_sent):
      for w in sent:
        if w in self._freq:
          ranking[i] += self._freq[w]
          print "ranking[i]: ", ranking[i]
          print "self.freq[w]: ", self._freq[w]
    sents_idx = self._rank(ranking, n)    
    return [sents[j] for j in sents_idx]

  def _rank(self, ranking, n):
    """ return the first n sentences with highest ranking """
    return nlargest(n, ranking, key=ranking.get)

#main
data_path = sys.argv[1]
data = text_processor.process(data_path, 'sentence', False, False)

# These have been pre-processed and cleaned
# Email bodies do not have stop words but subject lines do
email_bodies = []
subject_lines = []
email_ids = []

print(len(data))

# Load the input emails and their original subject lines
# NOTE: test replacing tab with no space later
for email in sorted(data, key=lambda i: i['id']) :
    if email['subject'] is not '':
        email_ids.append(email['id'])
        email_bodies.append(email['body'])
        subject_lines.append(email['subject'])
print("EMAIL ID's: ", len(email_ids))

# Inspecting some of the emails

f = open("generated_min_max_results.txt", "w+")
predicted_subject_lines = []
for i in range(300):
  #write to the file
  f.write("Email id: " + email_ids[i] )
  #grab by sentences to enter a period 
  body_i = ''
  for sentence in email_bodies[i]:
    body_i += sentence + '. '

  f.write("Email body: " + body_i )
  #remove al weird characters
  body_i = body_i.replace("*", "")
  body_i = body_i.replace("-", "")
  body_i = body_i.replace("_", "")
  body_i = body_i.replace("/", "")
  body_i = body_i.replace("$", "")
  body_i = body_i.replace("1", "one")
  body = [body_i, body_i]
  fs = FrequencySummarizer()
  title, text = body
  j = 0
  #Summarize for subject line
  for s in fs.summarize(text, 2):
    if j is 1: 
      if s[j] is ' ':
        body_i = body_i.replace("I", "")
        body_i = body_i.replace("i", "")
        body_i = body_i.replace("f", "")
        body_i = body_i.replace("A", "")
        body = [body_i, body_i]
        fs = FrequencySummarizer()
        title, text = body
        for s_jr in fs.summarize(text, 2):
            s = s_jr[0] #replace an empty subject line
      character_count = 40
      if len(s) > 40:
        #go backwards and erase until a space is reached
        for char in s:
          if s[character_count] is ' ':
            break
          character_count = character_count - 1 
      predicted_subject_lines.append(s[0:character_count])
      f.write("\nSubject Line: " + s[0:character_count] + "\n\n")
    j = j+1
predicted_score = evaluator.subject_line_evaluator(predicted_subject_lines)
f.write("Predicted (generated subject line) score:" + str(predicted_score))

f.close()
