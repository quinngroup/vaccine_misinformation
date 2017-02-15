# Vaccine Misinformation

This is the repo for Jonathan Waring's CURO 2017 project. 

##Abstract 

Vaccination provides the most effective method of preventing infectious diseases. While the effectiveness and safety of vaccines has been widely studied and verified, there is still opposition from the anti-vaccine movement. It has led to vaccine hesitancy, which is defined as a delay in acceptance or a refusal of vaccine services. It is an ever-growing and constantly changing problem that needs constant surveillance. The Internet plays a large role in disseminating vaccine misinformation to a large number of people, which contributes to the vaccine hesitancy problem. In order to combat the spread of misinformation online, it is important to first recognize true facts from false ones. We attempt to develop a machine learning strategy using natural language processing (NLP) that allows one to identify misinformation in vaccine-related webpages. This will be accomplished through the use of the low-dimensional document embedding algorithm, Doc2Vec. Through the use of semi-supervised learning, we take a small sample of manually labeled vaccine webpages and a large amount of unlabeled vaccine webpages, and attempt to classify misinformed webpages from accurate ones. Doc2Vec also provides methods for determining how semantically similar two documents may be, which can be used to determine what makes a vaccine webpage misinformed. The results of this study could enable both public health practitioners and the general public to monitor vaccine misinformation online in order to reduce vaccine hesitancy and identify strategies to improve vaccine education.  

##Notes 

Using Google Custom Search API to extract URLs about vaccine information. Instructions can be found here: https://stackoverflow.com/questions/37083058/programmatically-searching-google-in-python-using-custom-search 

##url.txt file

The first 20 links in this file are from my manual search for ground truth vaccine articles.
 
The next 100 links are from a Google CSE search using the term "vaccine safety".

The next 100 links are from a Google CSE search using the term "vaccine information".

The next 63 links are from a Google CSE search using the term "vaccine ingredients".  

The next 100 links are from a Google CSE search using the term "vaccine and autism". 

The next 100 links are from a Google CSE search using the term "vaccine schedule". 

The next 100 links are from a Google CSE search using the term "vaccines". 

The next 100 links are from a Google CSE search using the term "vaccine injury". 

The next 100 links are from a Google CSE search using the term "vaccine side effects". 

The next 100 links are from a Google CSE search using the term "vaccines for children".

The next 100 links are from a Google CSE search using the term "vaccine benefits". 

The next 100 links are from a Google CSE search using the term "vaccine dangers". 

The next 100 links are from a Google CSE search using the term "vaccine herd immunity".

The next 100 links are from a Google CSE search using the term "vaccine mercury".

The next 100 links are from a Google CSE search using the term "vaccine research".

The next 100 links are from a Google CSE search using the term "vaccine yes or no".

After removing duplicate webpage links, we are left with 1334 links for analysis. 