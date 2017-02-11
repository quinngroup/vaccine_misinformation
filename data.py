import csv
#import requests 
#from bs4 import BeautifulSoup 

#List that stores both url and text of vaccine webpages
webpage_data = {'url': [],
                'text': []}

#Collect urls of labeled data from saved dataset.csv file
with open('Vaccine Dataset.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, dialect='excel')
    for row in reader:
        webpage_data['url'].append(row[1]) #url is in second cell of csv

#Collect urls of unlabeled data
from googleapiclient.discovery import build
import pprint

my_api_key = "AIzaSyB9Y2Al6mhddN61ry1uouSDlvs2QLDEZdQ"
my_cse_key = "014695627772573494021:o3ywzrbv8eg"

def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return res['items']

for i in range(1, 100, 10): #range gets the next 10 results 
    results = google_search(
        'vaccine safety', my_api_key, my_cse_key, start=i, num=10)
    for result in results:
        webpage_data['url'].append(result['formattedUrl'])
    
print(len(webpage_data['url']))
print(len(set(webpage_data['url'])))
