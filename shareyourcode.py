# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.
# %pip install NLTK
# %pip install pyldavis

from google.colab import drive
drive.mount("/content/gdrive")

# importing the libraries

import gensim
import gensim.corpora as corpora
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from pprint import pprint
import spacy
import pickle
import re
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import pandas as pd



# the latest version of the tweets
#tweets = pd.read_csv('/content/gdrive/MyDrive/LDA/dp-export-77711e6c-e900-4724-9d88-06795ddbcd9a.csv')
# the tweets that are used in the tutorial
tweets = pd.read_csv('/content/gdrive/MyDrive/LDA/dp-export-original.csv')

tweets = tweets.Tweets.values.tolist()

# Turn the list of string into a list of tokens
tweets = [t.split(',') for t in tweets]

"""## Removing Symbols and Building the corpus
Topic modeling involves counting words and grouping similar word patterns to describe topics within the data. If the model understands how often words are used and which words tend to appear together, it will find patterns to group different words.

To begin, we convert a set of words into a "bag of words," which is a list of pairs (word_id, word_frequency).





"""

from gensim.corpora import Dictionary
import re

# Ensure all elements in the tweets list are strings
tweets = [str(tweet) for tweet in tweets]

# Remove symbols from each tweet
cleaned_tweets = [re.sub(r'[^\w\s]', '', tweet) for tweet in tweets]

# Tokenize and remove stop words for each cleaned tweet
tokenized_tweets = [word_tokenize(text) for text in cleaned_tweets]
filtered_tweets = [[word for word in tokens if word.lower() not in extended_stop_words] for tokens in tokenized_tweets]

# Create a Dictionary and Term Document Frequency
id2word = Dictionary(filtered_tweets)
corpus = [id2word.doc2bow(text) for text in filtered_tweets]

# corpus[:1] extracts the bag-of-words representation of the first text.
print(corpus[:1])

"""What do these tuples mean? Let’s convert them into human readable format to understand:

* Let's create a list of word-frequency pairs for the words in the first document's bag-of-words representation.

"""

[[(id2word[i], freq) for i, freq in doc] for doc in corpus[:1]]


# Build LDA model
lda_model = LdaModel(corpus=corpus,
                   id2word=id2word,
                   num_topics=5,
                   random_state=0,
                   chunksize=100,
                   alpha='auto',
                   per_word_topics=True,
                   iterations=100)

pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


pyLDAvis.enable_notebook()
p = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
p



from gensim.models import CoherenceModel

# Calculate Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=filtered_tweets, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()

print('Coherence Score:', coherence_lda)



