import requests
from bs4 import BeautifulSoup
import regex

html=requests.get("https://en.wikipedia.org/wiki/Deep_learning")
wiki=BeautifulSoup(html.content,"html.parser")
print(wiki.title)
print(wiki.find_all('a'))
for link in wiki.find_all('a',attrs={'href':regex.compile("htt")}):
    print(link.get('href'))
