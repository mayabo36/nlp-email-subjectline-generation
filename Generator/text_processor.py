import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import email

replace_no_space = re.compile('(\.)|(;)|(:)|(!)|(\')|(\?)|(,)|(\")|(\()|(\))|(\[)|(])|(>)|(<)')
replace_with_space = re.compile('(-)|(/)')
english_stop_words = stopwords.words('english')


# Token types: sentence, word, or none
# Create_labels and remove_stop_words are booleans
def process(data_path, token_type, create_labels, remove_stop_words):
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

    for e in employees[0:10]:
        folders = os.listdir(data_path + '/' + e)
        for f in folders:
            if f == 'sent_items':
                emails = os.listdir(data_path + '/' + e + '/' + f)
                for email in emails:
                    email_location = data_path + '/' + e + '/' + f + '/' + email
                    if os.path.isfile(email_location):
                        extracted = extract_metadata(email_location, token_type, create_labels, remove_stop_words)
                        if 'body' in extracted and len(extracted['body']):
                            email_metadata.append(extracted)

    return email_metadata


def extract_metadata(file_name, token_type, create_labels, remove_stop_words):
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
            metadata['original_body'] = metadata['body']
            metadata['body'] = clean_text(metadata['body'], token_type, remove_stop_words)
            if create_labels:
                metadata['label'] = create_label(metadata['body'])

        return metadata


def cleanse(text):
    return replace_with_space.sub(" ", replace_no_space.sub("", text)).replace('\n', ' ')


def strip_stop_words(text, remove_stop_words):
    if not remove_stop_words:
        return text
    return [t for t in text if t not in english_stop_words]


def clean_text(text_body, token_type, remove_stop_words):

    def func(text):
        return strip_stop_words(word_tokenize(cleanse(text)), remove_stop_words)
    # Remove email signature?

    if token_type == "sentence":
        return [' '.join(func(sentence)) for sentence in sent_tokenize(text_body)]
        # return [' '.join(strip_stop_words(word_tokenize(cleanse(sentence)), remove_stop_words)) for sentence in sent_tokenize(text_body)]

    elif token_type == 'word':
        return func(text_body)
        # return strip_stop_words(word_tokenize(cleanse(text_body)), remove_stop_words)

    elif token_type == 'none':
        return ' '.join(func(text_body))
        # return ' '.join(strip_stop_words(word_tokenize(cleanse(text_body)), remove_stop_words))

    return None


def create_label(text_body):
    labels = [
        ('Please see the attached', ['attachment', 'attached', 'attaching', 'please see', 'look at', 'take a look at',
                                     'enclosed', '.xls>>', '.xls >>', '.xls  >>', '.doc>>', '.doc >>', '.doc  >>',
                                     '.docx>>', '.docx >>', '.docx  >>', '.pdf>>', '.pdf >>', '.pdf  >>']),
        ('<Entity> Announcement/Update/Feedback', ['feedback', 'update', 'memo', 'announcement', 'by the way', 'btw',
                                                   'now', 'urgent', 'note']),
        ('<Entity> Report/Information/Summary', ['info', 'summary', 'report', 'summarize', ]),
        ('<Entity> Request/Approval/Review', ['review', 'approval', 'needs', 'need', 'request', 'would like',
                                              'let me know', 'lmk']),
        ('<Entity> Confirmation', ['confirm', 'i will', 'verify', 'can do']),
        ('<Entity> Question', ['what is', 'what are', 'what else', 'why', 'how', 'when', 'is there']),
        ('Favor to ask you', ['favor', 'can you', 'would you', 'please']),
        ('<Entity> Suggestion/Recommendation', ['suggestion', 'recommendation']),
        ('<Entity> Meeting', ['meeting', 'schedule', 'meet']),
        ('<Entity> Proposal', ['proposal']),
        ('Out of office',
         ['out of office', 'ooo', 'on holiday', 'on vacation', 'not be in the office', 'not be in office',
          'n\'t be in office', 'n\'t be in the office', 'not in office']),
        ('Thank You', ['thanks,', 'thank you']),
        ('Other/<Entity>', ['']),
    ]
    message = text_body.lower()
    for (index, (label, rules)) in enumerate(labels):
        for rule in rules:
            if rule in message:
                return index
