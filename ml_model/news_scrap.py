import requests
from bs4 import BeautifulSoup
import urllib.request, json 
import re
import pandas as pd
import csv
import time
#import numpy as np
# import predict_ner as ner
from predict_ner import *

API_KEY = ''

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

def get_all_text_element(url_link):
	page = requests.get(url_link)
	soup = BeautifulSoup(page.content, "html.parser")
	return cleanhtml(str(soup.find_all("p")))

def replace_str_index(text,index=0,replacement=''):
    return '%s%s%s'%(text[:index],replacement,text[index+1:])

def scrap_news(keyword):
    keyword =keyword.replace(" ", "+")
    url_scrap = []
    words = []
    url_api = 'https://newsapi.org/v2/everything?q='+keyword+'&sortBy=publishedAt&apiKey=' + API_KEY 
    with urllib.request.urlopen(url_api) as url:
        data = json.loads(url.read().decode())
    docanno = ''
    for i in data["articles"]:
        try:
            text_scrap = get_all_text_element(i["url"])
        except:
            text_scrap = "<EOS>"
        text_scrap = text_scrap[1:]
        text_scrap = replace_str_index(text_scrap, len(text_scrap)-1, " <EOS> ")
        text_scrap = text_scrap.replace("\n", "")
        text_scrap = text_scrap.replace("\t", "")
        # text_scrap = text_scrap.encode('ascii', errors='ignore').strip().decode('ascii')
        url_scrap.append({"text":text_scrap})
        docanno = run(text_scrap)

    output = {"scrapping":url_scrap}
    output = docanno
    return output

if __name__ == "__main__":
    article = 'LANGSA'
    ner.run(article)
