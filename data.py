import csv
from xgoogle.search import GoogleSearch, SearchError #google search tool

#List that stores both url and text of vaccine webpages
webpage_data = {'url': [],
                'text': []}

#Collect urls of labeled data from saved dataset.csv file
with open('Vaccine Dataset.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, dialect='excel')
    for row in reader:
        webpage_data['url'].append(row[1]) #url is in second cell of csv

#Collect urls of unlabeled data using xgoogle tool
try:
    vaccine_search = GoogleSearch("vaccine safety") #term to serach
    vaccine_search.results_per_page = 10 
    results = vaccine_search.get_results() #iterable object
    print len(results)
    for res in results:
        print res.title.encode("utf8")
        print res.desc.encode("utf8")
        print res.url.encode("utf8")
        print
except SearchError, e:
  print "Search failed: %s" % e
