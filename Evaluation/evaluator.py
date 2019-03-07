'''
This Evaluator will take an array of subject lines (input) and take the average of all the emails
'''


# Keeps the average score of all subject lines

def evaluator(subject_line):
    score = 100
    # print("!@!@!## ", subject_line)
    # No words detected (default to 0)
    if subject_line is "":
        score = 0
        # print("returning score of: ", score)
        return score
    # No capitalization detected (-10 points)
    if not subject_line.isupper():
        score -= 10
        # print("No Capitalization was detected")
    # First letter not capi-talized (-10 points)
    if not subject_line[0].isupper():
        score -= 10
        # print("First letter is not capitialized")

    # Contains 4 words or less (-10 points)
    if len(subject_line.split(" ")) <= 4:
        score -= 10
        # print("contained 4 words or less")

    # Mobile Subject Line is over 35 characters (-10 points)
    if len(subject_line) > 35:
        score -= 10
        # print("subject line contained over 35 characters")

    # Contains any word that has more than 8 characters (-10 points)
    contains_8_chars = False
    for words in subject_line.split(" "):
        if len(words) > 8:
            contains_8_chars = True
            # print(words, "is more than 8 characters")
    if contains_8_chars:
        score -= 10

    # Contains number (-10 points)
    if subject_line.isdigit():
        score -= 10
        # print("Sentence contained number")

    # Does contain sense of urgency (+10 points)
    contained_key_word = False
    keywords = ["important", "urgent", "please", "asap", "today", "shortly", "quickly", "now",
                "soon"]  # evaluates urgency
    for word in subject_line.split(" "):
        if word.lower() is keywords:
            contained_key_word = True
            # print("contained key word: ", word)
    if contained_key_word:
        if score is not 100:
            score += 10

    # print("returning score of: ", score)
    return score


def subject_line_evaluator(lines, avg=0):
    for eachLine in lines:
        # Evaluator:
        avg = avg + evaluator(eachLine)
    return avg / len(lines)


# from input import x
#
# avg_score = subject_line_evaluator(x)
# print("avg ", avg_score, " / ", len(x))
# print(avg_score / len(x))
