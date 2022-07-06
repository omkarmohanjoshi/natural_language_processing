# import the necessary packages
import re

def preprocess(sentDf, stopWords, key="sentence"):
    # loop over all the sentences
    for num in range(len(sentDf[key])):
        # lowercase the string and remove punctuation
        sentence = sentDf[key][num]
        print(sentence)
        sentence = re.sub(
            r"[^a-zA-Z0-9]", " ", sentence.lower()
        ).split()
        # define a list for processed words
        newWords = list()
        # loop over the words in each sentence and filter out the
        # stopwords
        for word in sentence:
            if word not in stopWords:
                # append word if not a stopword    
                newWords.append(word)
        # replace sentence with the list of new words   
        sentDf[key][num] = newWords
    
    # return the preprocessed data
    return sentDf


def prepare_tokenizer(df, sentKey="sentence", outputKey="sentiment"):
    # counters for tokenizer indices
    wordCounter = 0
    labelCounter = 0
    # create placeholder dictionaries for tokenizer
    textDict = dict()
    labelDict = dict()
    # loop over the sentences
    for entry in df[sentKey]:
        # loop over each word and
        # check if encountered before
        for word in entry:
            if word not in textDict.keys():
                textDict[word] = wordCounter
                # update word counter if new
                # word is encountered
                wordCounter += 1
    
    # repeat same process for labels  
    for label in df[outputKey]:
        if label not in labelDict.keys():
            labelDict[label] = labelCounter
            labelCounter += 1
    
    # return the dictionaries 
    return (textDict, labelDict)