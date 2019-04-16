import nltk
import text_processor
import sys
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from string import punctuation
from heapq import nlargest

class FrequencySummarizer:
  def __init__(self, min_cut=0.1, max_cut=0.9):
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
    sents_idx = self._rank(ranking, n)    
    return [sents[j] for j in sents_idx]

  def _rank(self, ranking, n):
    """ return the first n sentences with highest ranking """
    return nlargest(n, ranking, key=ranking.get)

#main
data_path = sys.argv[1]
data = text_processor.process(data_path, 'none', False, True)

# These have been pre-processed and cleaned
# Email bodies do not have stop words but subject lines do
email_bodies = []
subject_lines = []
# Load the input emails and their original subject lines
# NOTE: test replacing tab with no space later
for email in data:
    if email['subject'] is not '':
        email_bodies.append(email['body'])
        subject_lines.append(email['subject'])
        
# Inspecting some of the emails
for i in range(5):
    print("Email #", i + 1)
    print(email_bodies[i])
    print("Subject Line #", i + 1)
    print(subject_lines[i])
    print()
exit()
#sentences = ['mailtoIMCEANOTES +22Lafontaine+2C+20Steve+22+20+3Csteve+2Elafontaine+40ban kofamerica+2Ecom+3E+40ENRON @ ENRONcom Sent Thursday August 16 2001 402 PM To jarnold @ enroncom Subject rumours new rumour youre skillings replacement thats even better aga ********************************************************************** This e mail property Enron Corp relevant affiliate may contain confidential privileged material sole use intended recipient', 'Any review use distribution disclosure others strictly prohibited', 'If intended recipient authorized receive recipient please contact sender reply Enron Corp enronmessagingadministration @ enroncom delete copies message', 'This e mail attachments hereto intended offer acceptance create evidence binding enforceable contract Enron Corp affiliates intended recipient party may relied anyone basis contract estoppel otherwise', 'Thank', '**********************************************************************']
#for i, sentence in enumerate(sentences[0: -1]):
#  body = [sentences[i], sentences[i+1]]
#  fs = FrequencySummarizer()
#  #for article_url in to_summarize[:5]:
#  title, text = body
#  print('----------------------------------')
#  #print(text)
#  for s in fs.summarize(text, 2):
#    print('*',s)
#  print("stop.")
  #for s in fs.summarize(title, 2):
  #  print('*',s)
  