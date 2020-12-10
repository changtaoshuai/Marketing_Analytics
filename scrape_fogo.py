#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 19:15:12 2020

@author: sctlivelife
"""
import os
import pandas as pd
import re
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from datetime import datetime
from collections import Counter

os.chdir("/Users/sctlivelife/Downloads/brandeis_class/Marketing Analytics/final_project")
path_to_driver = '/Users/sctlivelife/Downloads/Web_Scraping/chromedriver'
driver = webdriver.Chrome(executable_path = path_to_driver)
url = 'https://www.opentable.com/r/fogo-de-chao-brazilian-steakhouse-boston?originId=2&corrid=fc909582-033b-4e45-9527-5d4fe0e9e781&avt=eyJ2IjoyLCJtIjowLCJwIjowLCJzIjowLCJuIjowfQ&p=2020-12-07T00%3A00%3A00'
driver.get(url)
               
Review = []
Rating = []
dine_time = []         
condition = True
while (condition):
    reviews = driver.find_elements_by_xpath("//div[@class='reviewListItem oc-reviews-91417a38']")
    for i in range(len(reviews)):
        soup = BeautifulSoup(reviews[i].get_attribute('innerHTML'))
        Review.append(re.findall('style="">(.*?)</p>',str(soup))[0])
        Rating.append(re.findall('div aria-label="(.*?) out of 5 ',str(soup))[0])
        dine_time.append(re.findall('Dined on(.*?)</span>',str(soup))[0])
    try:
        driver.find_element_by_xpath("//button[@aria-label = 'next-page']").click()
        time.sleep(2)
    except:
        condition = False
        
df = pd.concat([pd.DataFrame(Review),pd.DataFrame(Rating),pd.DataFrame(dine_time)],axis = 1)
df.columns = ['reviews','ratings','dine_in_time']
df['ratings'] = df['ratings'].astype('int')
df.to_csv('Fogo_de_Chao_review.csv')
