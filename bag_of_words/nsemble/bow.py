#The Bag-of-Words model is a simple method for extracting features from text data.

def calculate_bag_of_words(text, sentence):
    # create a dictionary for frequency check
    freqDict = dict.fromkeys(text, 0)
    # loop over the words in sentences
    for word in sentence:
        # update word frequency
        freqDict[word]=sentence.count(word)
    # return dictionary 
    return freqDict