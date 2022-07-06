# import the necessary packages
from tensorflow.keras.preprocessing.text import Tokenizer 

def tensorflow_wrap(df):
    # create the tokenizer for sentences
    tokenizerSentence = Tokenizer()
    # create the tokenizer for labels
    tokenizerLabel = Tokenizer()
    # fit the tokenizer on the documents
    tokenizerSentence.fit_on_texts(df["sentence"])
    # fit the tokenizer on the labels
    tokenizerLabel.fit_on_texts(df["sentiment"])
    # create vectors using tensorflow
    encodedData = tokenizerSentence.texts_to_matrix(
        texts=df["sentence"], mode="count")
    # add label column
    labels = df["sentiment"]
    # correct label vectors
    for i in range(len(labels)):
        labels[i] = tokenizerLabel.word_index[labels[i]] - 1
    # return data and labels
    return (encodedData[:, 1:], labels.astype("float32"))


