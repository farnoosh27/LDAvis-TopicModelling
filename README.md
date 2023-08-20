# LDAvis-TopicModelling
This code is an example of using the Gensim library to perform topic modeling and visualize them for a given set of tweets. It uses the Latent Dirichlet Allocation (LDA) algorithm to discover topics within the tweet data. The coherence score is used to measure the interpretability and quality of the discovered topics.
In this notebook, I will walk through the tutorial provided at this link (https://neptune.ai/blog/pyldavis-topic-modelling-exploration-tool-that-every-nlp-data-scientist-should-know) and try to run a part of it in GoogleColab.

## What is Topic Modelling?
## What is LDA?
### What are alpha and eta parameters?
### Benefits of having an LDA?
### What is pyLDAvis?
### What is the λ (Lambda) parameter in the visualization?

In ldavis, λ controls the relevance of terms within a topic. It is a parameter that you can adjust to influence which terms are shown for each topic in the visualization. Higher values of λ will prioritize terms that are more strongly associated with the topic, potentially leading to a more focused and coherent representation of the topic. Lower values of λ will result in a broader range of terms being displayed for the topic.
