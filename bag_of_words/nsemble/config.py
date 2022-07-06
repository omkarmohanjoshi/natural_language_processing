# define the data to be used
dataDict = {
    "sentence":[
        "Avengers is a great movie.",
        "I love Avengers it is great.",
        "Avengers is a bad movie.",
        "I hate Avengers.",
        "I didnt like the Avengers movie.",
        "I think Avengers is a bad movie.",
        "I love the movie.",
        "I think it is great."
    ],
    "sentiment":[
        "good",
        "good",
        "bad",
        "bad",
        "bad",
        "bad",
        "good",
        "good"
    ]
}
# define a list of stopwords
stopWrds = ["is", "a", "i", "it"] 
# define model training parameters
epochs = 30
batchSize = 10
# define number of dense units
denseUnits = 50

