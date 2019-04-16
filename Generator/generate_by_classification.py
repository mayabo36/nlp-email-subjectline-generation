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
data = text_processor.process(data_path, 'none', True, True)

# These have been pre-processed and cleaned
# Email bodies do not have stop words but subject lines do
email_bodies = []
subject_lines = []
email_ids = []
email_labels = []

# Load the input emails and their original subject lines
# NOTE: test replacing tab with no space later
for email in sorted(data, key=lambda i: i['id']):
    if email['subject'] is not '':
        email_ids.append(email['id'])
        email_bodies.append(email['body'])
        subject_lines.append(email['subject'])
        email_labels.append(email['label'])

# Inspecting some of the emails
for i in range(5):
    print("Email ", email_ids[i])
    print(email_bodies[i])
    print("Subject Line #", i + 1)
    print(subject_lines[i])
    print("Email label:", subject_line_labels[email_labels[i]])
    print()

# Create the n-grams
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
ngram_vectorizer.fit(email_bodies)
X = ngram_vectorizer.transform(email_bodies)
Y = email_labels

X_train, X_val, y_train, y_val = train_test_split(
    X, Y, train_size=0.75
)

# Use a k-neighbors classifier to label emails and train it on 75% of the data
knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
print('Accuracy Score:', accuracy_score(y_val, knn.predict(X_val)))


# Get the average evaluation score for the original subject lines
baseline_score = evaluator.subject_line_evaluator(subject_lines)

# Get the average evaluation score for 25% of the data predicted subject lines
predicted_subject_lines = []
for index, email in enumerate(email_bodies):
    subject_prediction = subject_line_labels[knn.predict(ngram_vectorizer.transform([email]))[0]]
    predicted_subject_lines.append(subject_prediction)
predicted_score = evaluator.subject_line_evaluator(predicted_subject_lines)

print('Baseline (original subject line) score:', baseline_score)
print('Predicted (generated subject line) score:', predicted_score)
# 66.95886460230723 67.76563448694596
