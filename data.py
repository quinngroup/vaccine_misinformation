import csv

#List that stores both url and text of vaccine webpages
webpage_data = {'url': [],
                'text': []}

#Collect urls of labeled data from saved dataset.csv file
with open('Vaccine Dataset.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, dialect='excel')
    for row in reader:
        webpage_data['url'].append(row[1]) #url is in second cell of csv

#Collect urls of unlabeled data
