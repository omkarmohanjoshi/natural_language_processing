# USAGE
# python train.py
# import the necessary packages

from nsemble import config
from nsemble.model import build_shallow_net
from nsemble.bow import calculate_bag_of_words
from nsemble.data_preprocessing import preprocess
from nsemble.data_preprocessing import prepare_tokenizer
from nsemble.tensorflow_wrapper import tensorflow_wrap
import pandas as pd
# convert the input data dictionary to a pandas data frame
df = pd.DataFrame.from_dict(config.dataDict)
# preprocess the data frame and create data dictionaries
preprocessedDf = preprocess(sentDf=df, stopWords=config.stopWrds)
(textDict, labelDict) = prepare_tokenizer(df)
# create an empty list for vectors
freqList = list()
# build vectors from the sentences
for sentence in df["sentence"]:
    # create entries for each sentence and update the vector list   
    entryFreq = calculate_bag_of_words(text=textDict,
        sentence=sentence)
    freqList.append(entryFreq)
# create an empty data frame for the vectors
finalDf = pd.DataFrame() 
# loop over the vectors and concat them
for vector in freqList:
    vector = pd.DataFrame([vector])
    finalDf = pd.concat([finalDf, vector], ignore_index=True)
# add label column to the final data frame
finalDf["label"] = df["sentiment"]
# convert label into corresponding vector
for i in range(len(finalDf["label"])):
    finalDf["label"][i] = labelDict[finalDf["label"][i]]
# initialize the vanilla model
shallowModel = build_shallow_net()
print("[Info] Compiling model...")
# fit the Keras model on the dataset
shallowModel.fit(
    finalDf.iloc[:,0:10],
    finalDf.iloc[:,10].astype("float32"),
    epochs=config.epochs,
    batch_size=config.batchSize
)
# create dataset using TensorFlow
trainX, trainY = tensorflow_wrap(df)
# initialize the new model for tf wrapped data
tensorflowModel = build_shallow_net()
print("[Info] Compiling model with tensorflow wrapped data...")
# fit the keras model on the tensorflow dataset
tensorflowModel.fit(
    trainX,
    trainY,
    epochs=config.epochs,
    batch_size=config.batchSize
)