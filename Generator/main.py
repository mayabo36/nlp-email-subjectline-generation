from Generator import text_processor
import sys

data_path = sys.argv[1]

# Process the email dataset
results = text_processor.process(data_path)

print(results[0:10])
