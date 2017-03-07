from __future__ import with_statement 
import signal
from contextlib import contextmanager
import csv
from goose import Goose
import pandas as pd

#List that stores url and text of vaccine webpages
webpage_data = {'Title': [],
                'Site URL': [],
                'Text': []}

urls = [] #iterable for text extraction

#open Vaccine Dataset.csv to get urls 
with open('CSVFiles/Vaccine Dataset.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, dialect='excel')
    for row in reader:
        if(row[1] != 'Site URL'):
            urls.append(row[1]) #url is in second cell of csv

#open Custom Search.csv to get urls 
with open('CSVFiles/Custom Search.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, dialect='excel')
    for row in reader:
        if(row[1] != 'Site URL'):
            urls.append(row[1]) #url is in second cell of csv

#remove duplicate links without having to use set
def uniq(input):
    output = []
    for x in input:
        if x not in output:
            output.append(x)
    return output

urls = uniq(urls)

#function to trim article to certain amount of words
def word_trimmer(s, n):
    return ' '.join(s.split()[:n])

urls = urls[20:] #remove inital 20 which we already have text on

#function used to skip iteration if goose takes too long to extract
class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException, "Timed out!"
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

for url in urls:
    try:
        with time_limit(10): #skip if url takes longer than 10 seconds
            g = Goose()
            article = g.extract(url=url)
            unicode_text = article.cleaned_text
            text = unicode_text.encode('ascii', 'ignore').replace('\n', '')
            limited_text = word_trimmer(text, 5000)
            title = article.title.encode('ascii', 'ignore')
            if(limited_text != ''):
                webpage_data['Title'].append(title)
                webpage_data['Site URL'].append(url)
                webpage_data['Text'].append(limited_text)
    except Exception as e:
        print repr(e)

# build a DataFrame with the extracted information
df = pd.DataFrame(webpage_data, 
                  columns=['Title', 'Site URL', 'Text', 'Classifaction'])
df.to_csv('CSVFiles/Text.csv', mode='w', index= False, 
                          encoding='utf-8')
