from __future__ import with_statement # Required in 2.5
import signal
from contextlib import contextmanager
from goose import Goose

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

url = 'http://www.who.int/vaccine_safety/en/'
url2 = 'http://www.naturalnews.com/056161_fake_news_mainstream_media_vaccines.html'

g = Goose()

try:
    with time_limit(10):
        article = g.extract(url=url)
except TimeoutException, msg:
    print "Timed out!"

print(article.cleaned_text)

'''
article2 = g.extract(url=url2)
print(type(article2))
#print(article2.cleaned_text)
'''
