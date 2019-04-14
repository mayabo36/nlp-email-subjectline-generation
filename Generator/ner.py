# ********************************************************************
# Title: Named Entity Recognition with Regular Expression: NLTK
# Name: alvas
# Date: 2014
# Availability: https://stackoverflow.com/questions/24398536/named-entity-recognition-with-regular-expression-nltk
# ********************************************************************

from nltk import word_tokenize, pos_tag, ne_chunk
from nltk import Tree

def generate_by_ner(text_body):
    # The function nltk.ne_chunk() returns a nested tree object that can be traversed to find NEs.
    chunked = ne_chunk(pos_tag(word_tokenize(text_body)))
    # Chunk to keep NER that is more than one word such as a first and last name or a company name that is
    # more than one word together.
    continuous_chunk = []
    # Holds only one NE.
    current_chunk = []

    # Traversing tree to find NEs.
    for subtree in chunked:
        if type(subtree) == Tree:
            current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue
            
    return continuous_chunk