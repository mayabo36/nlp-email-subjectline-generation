import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re


def process(data_path):
    """
    Takes a data_path and assumes data is structured as:

        data_path
            employee_1
                sent_items
                    email_file_1
                    ...
            employee_2
                sent_items
                    email_file_1
                    ...
            ...

    Will return metadata for each email.

    :param str data_path: absolute path to the root folder for emails
    :return list<dict>: list of email metadata
    """

    email_metadata = []
    employees = os.listdir(data_path)

    for e in employees:
        folders = os.listdir(data_path + '/' + e)
        for f in folders:
            if f == 'sent_items':
                emails = os.listdir(data_path + '/' + e + '/' + f)
                for email in emails:
                    email_location = data_path + '/' + e + '/' + f + '/' + email
                    if os.path.isfile(email_location):
                        extracted = extract_metadata(email_location)
                        if 'body' in extracted and len(extracted['body']):
                            email_metadata.append(extracted)

    return email_metadata


def extract_metadata(file_name):
    """
    Will extract metadata from an email.

    :param str file_name: absolute file path to email
    :return dict: email metadata containing id, subject, body
    """
    metadata = {'file': file_name}

    with open(file_name, 'r') as file:
        rows = file.readlines()

        rules = [
            ['Message-ID:', 'id'],
            ['Subject:', 'subject'],
        ]

        for (index, row) in enumerate(rows):
            row = row.lstrip('> \t')
            for (pattern, prop) in rules:
                if row.startswith(pattern):
                    metadata[prop] = row.replace(pattern,'')

            if 'body' not in metadata:
                if row.startswith('\n'):
                    metadata['body'] = '\n'.join(rows[index:])

            elif '-----Original Message-----' in row:
                del metadata['body']

        if 'body' in metadata:
            metadata['body'] = clean_text(metadata['body'])
        return metadata


def clean_text(text_body):

    # Remove punctuation, quotation marks, etc.
    # From: https://towardsdatascience.com/sentiment-analysis-with-python-part-1-5ce197074184
    replace_no_space = re.compile('(\.)|(;)|(:)|(!)|(\')|(\?)|(,)|(\")|(\()|(\))|(\[)|(])|(>)|(<)')
    replace_with_space = re.compile('(-)|(/)')
    text_body = replace_no_space.sub("", text_body)
    text_body = replace_with_space.sub(" ", text_body)

    # Tokenize the text
    tokens = word_tokenize(text_body)

    # Remove stop words: a, the, is, etc.
    english_stop_words = stopwords.words('english')
    cleaned_tokens = [t for t in tokens if t not in english_stop_words]

    return " ".join(cleaned_tokens)
