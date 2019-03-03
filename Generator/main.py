from Generator import text_processor
import sys

data_path = sys.argv[1]

# Process the email dataset
results = text_processor.process(data_path)
print("Number of emails:", len(results))
#
# for result in results:
#     print(result["body"], "\n\n")
