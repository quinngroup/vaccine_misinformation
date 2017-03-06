import csv
from goose import Goose
import pandas as pd

#List that stores url and text of vaccine webpages
webpage_data = {'Title': [],
                'Site URL': [],
                'Text': []}

urls = []

#open Vaccine Dataset.csv to get urls 
with open('Vaccine Dataset.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, dialect='excel')
    for row in reader:
        if(row[1] != 'Site URL'):
            urls.append(row[1]) #url is in second cell of csv

#open Custom Search.csv to get urls 
with open('Custom Search.csv', 'r') as csvfile:
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

print(len(urls))
urls = uniq(urls)
print(len(urls))

#function to trim article to certain amount of words
def word_trimmer(s, n):
    return ' '.join(s.split()[:n])

urls = urls[20:] #remove inital 20 which we already have text on

i = 0 #track how many urls have been processed in cycle
j = 0 #track total number of urls already processed

for url in urls:
    if 'naturalnews' in url: #goose can't open this domain for some reason
        pass
    else:
        try:
            g = Goose()
            article = g.extract(url=url)
            unicode_text = article.cleaned_text
            text = unicode_text.encode('ascii', 'ignore').replace('\n', '')
            limited_text = word_trimmer(text, 5000)
            title = article.title.encode('ascii', 'ignore')
            webpage_data['Title'].append(title)
            webpage_data['Site URL'].append(url)
            webpage_data['Text'].append(limited_text)
            i = i + 1
            j = j + 1
            if(i == 10 or j == len(urls)):
                # build a DataFrame with the extracted information
                df = pd.DataFrame(webpage_data, 
                                  columns=['Title', 'Site URL', 'Text', 'Classifaction'])
                df.to_csv('Text.csv', mode='a', index= False, 
                          encoding='utf-8', header = False)
                #reset in order to avoid running out of memory
                webpage_data.clear()
                webpage_data = {'Title': [],
                                'Site URL': [],
                                'Text': []}
                i = 0
        except Exception as e:
            print repr(e)
            j = j + 1
            #pass
