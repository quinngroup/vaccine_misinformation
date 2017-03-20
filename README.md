# Vaccine Misinformation

This is the repo for Jonathan Waring's CURO 2017 project. 

## Abstract 

Vaccination provides the most effective method of preventing infectious diseases. While the effectiveness and safety of vaccines has been widely studied and verified, there is still opposition from the anti-vaccine movement. It has led to vaccine hesitancy, which is defined as a delay in acceptance or a refusal of vaccine services. It is an ever-growing and constantly changing problem that needs constant surveillance. The Internet plays a large role in disseminating vaccine misinformation to a large number of people, which contributes to the vaccine hesitancy problem. In order to combat the spread of misinformation online, it is important to first recognize true facts from false ones. We attempt to develop a machine learning strategy using natural language processing (NLP) that allows one to identify misinformation in vaccine-related webpages. This will be accomplished through the use of the low-dimensional document embedding algorithm, Doc2Vec. Through the use of semi-supervised learning, we take a small sample of manually labeled vaccine webpages and a large amount of unlabeled vaccine webpages, and attempt to classify misinformed webpages from accurate ones. Doc2Vec also provides methods for determining how semantically similar two documents may be, which can be used to determine what makes a vaccine webpage misinformed. The results of this study could enable both public health practitioners and the general public to monitor vaccine misinformation online in order to reduce vaccine hesitancy and identify strategies to improve vaccine education.  

## Resources

Using Google Custom Search API to extract URLs about vaccine information. Instructions can be found here: https://stackoverflow.com/questions/37083058/programmatically-searching-google-in-python-using-custom-search 

Using the third-party library, Goose, to extract main text of each HTML article from the scrapped URLs. The aim of the software is to take any news article or article-type web page and not only extract what is the main body of the article but also all meta data and most probable image candidate. However, for the purposes of this project, I currently only use it to extract main body of the article in order to do some NLP on the text. The repository for the library is hosted here: https://github.com/grangier/python-goose

Using the gensim libary to build our doc2vec models. Information can be found here: https://radimrehurek.com/gensim/models/doc2vec.html. 
A good explanation of the model is found here: https://rare-technologies.com/doc2vec-tutorial/. 
This tutorial was used to help me build my model: https://medium.com/@klintcho/doc2vec-tutorial-using-gensim-ab3ac03d3a1#.q8qtp8n81. 

## CSV Files

The 'Vaccine Dataset.csv' file contains 20 URL links, along with the title of the article, text of the article, and my classification of MISINFORMED or TRUE. These serve as our labeled documents in the model, and all classification of the 'ground truth' was done by me via fact checking. 

The 'Custom Search.csv' file contains 1500 URL links, along with the title of the article associated with that link. All of these links were extracted from the Google Custom Search Engine API using common vaccine queries into Google. Each query has 100 links associated with it (limiations of the API), and the queries used are as follows: "vaccine safety", "vaccine information", "vaccine ingredients", "vaccine and autism", "vaccine schedule", "vaccines", "vaccine injury", "vaccine side effects", "vaccines for children", "vaccine benefits", "vaccine dangers", "vaccine herd immunity", "vaccine mercury", "vaccine research", and "vacine yes or no". 

The 'Text.csv" file contains 1096 of the custom search url links and their corresponding main article text. The reason some links got eliminated are either that the Goose extractor could not find their text within the HTML, or Goose timed out in trying to reach that URL. 

After removing duplicate webpage links from the original 1520, we are left with 1411 links for analysis.  

After running text collection algorithm, we are left with the original 20 links, and 1096 of the custom search links. 

## Hyperparameter Optimization

Given that the Doc2Vec model is sensitve to the dimensionality, window size, and min count word parameters, I decided to run the doc2vec model on 216 different combinations of these parameters to see what would result in the best classification inference accuracy when using only the labeled data. The results of the hyperparameter optimization suggests that a vector dimensionality of 1000, window size of 6, and min count of 4 results in the highest classification accuracy (0.95). These paramters will be used when building the doc2vec model on all data. 

## Models

This directory contains the 2 models, one with labeled data only and one withboth labeled and unlabeled data, built on the optimized hyperparameters described above. 
 
