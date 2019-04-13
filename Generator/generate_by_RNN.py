from Generator import text_processor
import sys

data_path = sys.argv[1]

# Process the email dataset to clean, tokenize, and remove stop words (overly common words) from the email body
data = text_processor.process(data_path, 'none', False, False)

print(data[40]['body'])

# Process the email dataset to clean, tokenize, and remove stop words (overly common words) from the email body
data = text_processor.process(data_path, 'none', False, True)

print(data[40]['body'])

# Process the email dataset to clean, tokenize, and remove stop words (overly common words) from the email body
data = text_processor.process(data_path, 'sentence', False, False)

print(data[40]['body'])

# Process the email dataset to clean, tokenize, and remove stop words (overly common words) from the email body
data = text_processor.process(data_path, 'sentence', False, True)

print(data[40]['body'])

# Process the email dataset to clean, tokenize, and remove stop words (overly common words) from the email body
data = text_processor.process(data_path, 'word', False, False)

print(data[40]['body'])

# Process the email dataset to clean, tokenize, and remove stop words (overly common words) from the email body
data = text_processor.process(data_path, 'word', False, True)

print(data[40]['body'])

