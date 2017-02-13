import urllib 
from bs4 import BeautifulSoup

#List that stores url and text of vaccine webpages
webpage_data = {'url': [],
                'text': []}

#open url.txt file to get all urls
with open('url.txt') as file:
    urls = file.readlines()

#remove duplicate links without having to use set
def uniq(input):
    output = []
    for x in input:
        if x not in output:
            output.append(x)
    return output

urls = uniq(urls)

for url in urls:
    try:
        html = urllib.urlopen(url).read()
        webpage_data['url'].append(url)
        soup = BeautifulSoup(html, 'lxml')

        #get rid of any script or stylisitc crap
        for script in soup(["scrip", "style"]):
            script.extract() #remove script

        #get text
        text = soup.get_text()

        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)

        #convert unicode to string
        text = text.encode("utf-8")

        #Get rid of non-body text
        raw_strings = str.splitlines(text)
        long_text = []
        for string in raw_strings:
            if len(string) > 200:
                long_text.append(string)

        #Get rid of excess text that don't appear to be sentences
        sentences = []
        for text in long_text:
            if '[' in text or '|' in text or '=' in text or '&' in text or '\\' in text:
                pass
            else:
                sentences.append(text)

        #join list of sentences into one string
        page_text = '\n'.join(sentences)
    
        #add text to bigger list of webpage text
        webpage_data['text'].append(page_text)
    except Exception as e:
        print repr(e)

print(len(webpage_data['url']))
print(len(webpage_data['text']))

