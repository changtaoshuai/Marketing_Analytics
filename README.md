In this section, I perform web scraping on OpenTable for restaurant reviews. Here I chose Fogo de Chao in Boston.

Here is an example of reviews:
![](/images/website_review.jpg)

First, I used **selenium** in Python to open Fogo de Chao's web page.

```python
from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd
import re
path_to_driver = '/Users/sctlivelife/Downloads/Web_Scraping/chromedriver'
driver = webdriver.Chrome(executable_path = path_to_driver)
url = 'https://www.opentable.com/r/fogo-de-chao-brazilian-steakhouse-boston?originId=2&corrid=fc909582-033b-4e45-9527-5d4fe0e9e781&avt=eyJ2IjoyLCJtIjowLCJwIjowLCJzIjowLCJuIjowfQ&p=2020-12-07T00%3A00%3A00'
driver.get(url)
```

Next, I use x path to get each individual review and **Beautifulsoup** to transform the data structure. After I look into the data structure and find common patterns, I use regular expression to extract review text, ratings (stars), and dine in time.

```python
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
```

Now that I have review text, ratings (stars), and dine in time in 3 separate lists, it is easy to transform them into a single data set.

```python
df = pd.concat([pd.DataFrame(Review),pd.DataFrame(Rating),pd.DataFrame(dine_time)],axis = 1)
df.columns = ['reviews','ratings','dine_in_time']
df['ratings'] = df['ratings'].astype('int')
df.to_csv('Fogo_de_Chao_review.csv')
```

The final data set looks like this (2389 rows in total):

![](/images/data.jpg)


Now, web scraping is completed!

