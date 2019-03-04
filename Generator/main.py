from Generator import text_processor
from sklearn.feature_extraction.text import CountVectorizer
import sys

data_path = sys.argv[1]

# Process the email dataset to clean, tokenize, and remove stop words (overly common words) from the email body
results = text_processor.process(data_path)
print("Number of emails:", len(results))

# Create the n-grams
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
ngram_vectorizer.fit([email['body'] for email in results])
print(ngram_vectorizer)

