from Generator import text_processor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import Evaluation.evaluator as evaluator
import sys

subject_line_labels = [
    'Please see the attached',
    '<Entity> Announcement/Update/Feedback',
    '<Entity> Report/Information/Summary',
    '<Entity> Request/Approval/Review',
    '<Entity> Confirmation',
    '<Entity> Question',
    'Favor to ask you',
    '<Entity> Suggestion/Recommendation',
    '<Entity> Meeting',
    '<Entity> Proposal',
    'Out of office',
    'Thank You',
    'Other/<Entity>']

data_path = sys.argv[1]

# Process the email dataset to clean, tokenize, and remove stop words (overly common words) from the email body
data = text_processor.process(data_path)

# Create the n-grams
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
emails = [email['body'] for email in data]
labels = [email['label'] for email in data]
ngram_vectorizer.fit(emails)
X = ngram_vectorizer.transform(emails)
Y = labels

X_train, X_val, y_train, y_val = train_test_split(
    X, Y, train_size=0.75
)

# Use a k-neighbors classifier to label emails and train it on 75% of the data
knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
print(accuracy_score(y_val, knn.predict(X_val)))

# var = 225
# print(labels[var], emails[var], knn.predict(ngram_vectorizer.transform([emails[var]])))

# Get the average evaluation score for the original subject lines
original_subject_lines = [email['subject'] for email in data]
baseline_score = evaluator.subject_line_evaluator(original_subject_lines)

# Get the average evaluation score for 25% of the data predicted subject lines
predicted_subject_lines = []
for index, email in enumerate(emails[::4]):
    subject_prediction = subject_line_labels[knn.predict(ngram_vectorizer.transform([email]))[0]]
    predicted_subject_lines.append(subject_prediction)
    # if index % 100 is 0:
    #     print('Subject: {}\nBody: {}'.format(subject_prediction, data[index]['original_body']))

predicted_score = evaluator.subject_line_evaluator(predicted_subject_lines)

print(baseline_score, predicted_score)
# 66.95886460230723 67.76563448694596
