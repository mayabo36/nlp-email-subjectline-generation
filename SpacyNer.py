#!/usr/bin/env python
# coding: utf-8

# In[35]:


import spacy
nlp = spacy.load('en')

def find_ne(string):
    
    # Input passed from main function
    mainInput = nlp(string)

    # Array of Named Entities to be returned
    namedEntities = []

    for chunk in mainInput.noun_chunks:
        if chunk.text not in namedEntities:
            namedEntities.append(chunk.text)

    return namedEntities


# Main 
string1 = "I am waiting to get info. on two more properties in San Antonio. The broker will be faxing info on Monday One is 74 units for $1,900,000, and the other is 24 units for $550,000 Also, I have mailed you info. on three other properties in addition to the 100 unit property that I faxed this AM. Let me know if you have any questions."
string2 = "For clarity, wherever you cite Section 6.2 in your response I assume you mean Section 6.3. Your response does not address the key points upon which I based my interpretation of the plan.  The beginning of Section 6.3 states Notwithstanding any other provision of the plan?   This implies that Section 6.3 contains the complete rules for an accelerated distribution.  I disagree with your translation of a single sum distribution as meaning all at once instead of a single amount.  I would like to meet to discuss or appeal to Greg W. or John L.   In addition, your response does not address Section 7.1.  This appears to be the relevant section if the PSA is to be paid in shares.  Additionally the Q & A section in the plan brochure (pg. 8) works through an example of adjusting the number of shares to retain a certain dollar value.  This is the methodology described in Section 7.1. Tom Martin and Scott Neal share the above concerns.  Please advise of a convenient time we can all meet to discuss further."
string3 = "Sempra Energy has called and asked to cover outstanding positions with Enron. They have done some deals with me this morning via EOL, as well as a sleeve with EPNG in order to minimize their exposure with us. Indications were that they have longer dated power and other gas positions that they want to minimize as well, I advised them to call the respected traders first, as apposed to calling the voice brokers, to see what they can faciliate."

ne = []
ne = find_ne(string3)

# Testing spacy's NER
for i in ne:
    print(i)
    


# In[ ]:




