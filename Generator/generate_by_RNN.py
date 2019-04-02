from Generator import text_processor
import sys

data_path = sys.argv[1]

# Process the email dataset to clean, tokenize, and remove stop words (overly common words) from the email body
data = text_processor.process(data_path, 'sentence')

print(data[42]['body'])

