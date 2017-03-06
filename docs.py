import csv

trueCounter = 1 #used to help label docs
misinformedCounter = 1 #used to help label docs
documentCounter = 1 #used to help label docs

#open Vaccine Dataset.csv to get text
with open('CSVFiles/Vaccine Dataset.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, dialect='excel')
    for row in reader:
        if(row[2] != 'Text'):
            text = row[2] #text is in third cell of csv
            if(row[3] == 'TRUE'):
                #write text results to txt file
                directory = 'Documents/ClassifiedDocuments/'
                label = 'True' + str(trueCounter) + '.txt'
                filename = directory + label
                file = open(filename, "w")
                file.write(text)
                file.close()
                trueCounter = trueCounter + 1
            elif(row[3] == 'MISINFORMED'):
                #write text results to txt file
                directory = 'Documents/ClassifiedDocuments/'
                label = 'Misinformed' + str(misinformedCounter) + '.txt'
                filename = directory + label
                file = open(filename, "w")
                file.write(text)
                file.close()
                misinformedCounter = misinformedCounter + 1

'''
#open Text.csv to get custom search text
with open('CSVFiles/Text.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, dialect='excel')
    for row in reader:
        if(row[2] != 'Text'):
            text = row[2] #text is in third cell of csv
            #write text results to txt file
            directory = 'Documents/UnlabeledDocuments/'
            label = 'Doc' + str(documentCounter) + '.txt'
            filename = directory + label
            file = open(filename, "w")
            file.write(text)
            file.close()
            documentCounter = documentCounter + 1
'''
