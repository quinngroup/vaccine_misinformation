import csv
 
#List that stores url of vaccine webpages
webpage_data = {'url': []}

#Collect urls of labeled data from saved dataset.csv file
#with open('Vaccine Dataset.csv', 'r') as csvfile:
 #   reader = csv.reader(csvfile, dialect='excel')
  #  for row in reader:
   #     webpage_data['url'].append(row[1]) #url is in second cell of csv

#Collect urls of unlabeled data
from googleapiclient.discovery import build

my_api_key = "AIzaSyB9Y2Al6mhddN61ry1uouSDlvs2QLDEZdQ"
my_cse_id = "014695627772573494021:o3ywzrbv8eg"

def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return res['items']

for i in range(1, 100, 10): #range gets the next 10 results 
    results = google_search(
        'vaccine yes or no', my_api_key, my_cse_id, start=i, num=10)
    for result in results:
        webpage_data['url'].append(result['formattedUrl'])
    
print(len(webpage_data['url']))
print(len(set(webpage_data['url'])))

#write url results to txt file
file = open("url.txt", "a")
for url in webpage_data['url']:
    file.write(url)
    file.write('\n')
