'''
This Evaluator will take an array of subject lines (input) and take the average of all the emails
'''
#Keeps the average score of all subject lines

def evaluator(subjectLine):
    score = 100
    print("!@!@!## " , subjectLine)
     #No words detected (default to 0)
    if subjectLine is "":
        score = 0
        print("returning score of: ", score)
        return score
    #No capitalization detected (-10 points)
    if not subjectLine.isupper():
        score -= 10
        print("No Capitalization was detected")
    #First letter not capi-talized (-10 points)
    if not subjectLine[0].isupper():
        score -= 10
        print("First letter is not capitialized")

    #Contains 4 words or less (-10 points)
    if len(subjectLine.split(" ")) <= 4:
        score -= 10
        print("contained 4 words or less")

    #Mobile Subject Line is over 35 characters (-10 points)
    if len(subjectLine) > 35:
        score -= 10
        print("subject line contained over 35 characters")

    #Contains any word that has more than 8 characters (-10 points)
    contains_8_chars = False
    for words in subjectLine.split(" "):
        if len(words) > 8:
            contains_8_chars = True
            print(words, "is more than 8 characters")
    if contains_8_chars:
        score -= 10

    #Contains number (-10 points)
    if subjectLine.isdigit():
        score -= 10
        print("Sentence contained number")

     #Does contain sense of urgency (+10 points)
    contained_key_word = False
    keywords = ["important", "urgent", "please", "asap", "today", "shortly", "quickly", "now", "soon"] #evaluates urgency
    for word in subjectLine.split(" "):
        if word.lower() is keywords:
            contained_key_word = True
            print("contained key word: ", word)
    if contained_key_word:
        if score is not 100:
            score += 10

    print("returning score of: ", score)
    return score

def subject_line_evaluator(lines, avg_score=0):
    for eachLine in lines:
        #Evaluator:
        avg_score = avg_score + evaluator(eachLine)
    return avg_score

from input import x
avg_score = subject_line_evaluator(x)
print("avg ", avg_score, " / ", len(x))
print( avg_score/len(x))