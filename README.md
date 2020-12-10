# Sentiment Analysis                                              

In wanting to demonstrate my abilities in using Data Analytic techniques, I chose one of my favoraite restaurants, **Fogo de Chao (Botson)**, to perform Web Scraping, create data visualizations, and build prediction models. As such, discussed in Part 1, I used Web Scraping to draw from reviews of the restaurant that were posted on Opentable. Based on these reviews, I then created visualization and prediction models (Part 2). After comparing the utility of various model performances, I concluded that `Lasso` is the best model for this case because it has the highest accuracy score.

## Part 1: Web Scraping

In Part 1, I conducted Web Scraping using posted reviews of Fogo de Chao (Boston) found on the website OpenTable.

Here are two examples of reviews posted on OpenTable:
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

Next, I used x path to get reviews and used **Beautifulsoup** to transform the data structure. After finding common patterns in Beautifulsoup, I used regular expression to extract review text, ratings (stars), and dine in time. For example, review text follows *style=""* and is followed by *</p>*. So I extracted everything in the middle using ((.*?)). 

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

The code above scraped the reviews in text, ratings (stars), and dine-in time, placing these into 3 separate lists. In ordre to prepare for data analysis, I then combined the lists into a single data set. 

```python
df = pd.concat([pd.DataFrame(Review),pd.DataFrame(Rating),pd.DataFrame(dine_time)],axis = 1)
df.columns = ['reviews','ratings','dine_in_time']
df['ratings'] = df['ratings'].astype('int')
df.to_csv('Fogo_de_Chao_review.csv')
```

Here is a quick look of the final data set:

![](/images/data.jpg)


There are 2389 rows and 3 columns. The first column is reviews in text (e.g., 'I would not recommend'), the second column is ratings (e.g., 4 stars) and the last column is the customers' dine-in time (i.e., reservation time) at Fogo de Chao. 

Web Scraping is now complete.

## Part 2: Data Analysis

In Part 2 I used the data set (discussed above) for data analysis. Prior to performing deeper analysis like prediction models, it was important to create visualizations in order to understand what the data "looks like." Therefore, I started by importing the following packages for data visualization:

### Import data

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from wordcloud import WordCloud, STOPWORDS
```

After importing the packages, I deleted the column "Unnamed: 0" and then created 3 separate variables: Year, Month, Day.

```python
df = pd.read_csv('Fogo_de_Chao_review.csv')
df.dtypes
df.head()
df = df.drop('Unnamed: 0', axis = 1)
df['dine_in_time'] = df['dine_in_time'].astype('datetime64')
df['year'] = pd.DatetimeIndex(df['dine_in_time']).year
df['month'] = pd.DatetimeIndex(df['dine_in_time']).month
df['day'] = pd.DatetimeIndex(df['dine_in_time']).day
```
Here is a summary of what I've done so far:

![](/images/data_final.jpg)

### Exploratory data analysis

To get a better understanding of how Fogo de Chao has performed, I looked at the number of reviews by each year.

![](/images/year.png)

Generally speaking, a higher number of reviews means a higher number of customers or related service activities. The restaurant seemed to do pretty well from 2016 to 2019. However, there has been a noticeable drop in reviews throughout 2020, most likely due to COVID-19.

Next, I looked at the number of reviews by month.

![](/images/month.png)

From this graph, we can see people are more likely to leave reviews in August, which suggests that there might be higher number of customers during this month.

Next, I was interested in examining each year individually by month.

![](/images/year_month.png)

Between 2016 and 2019, there is a slight trend in the number of reviews submnitted, which start low in January and increase through August (with a considerable spike in reviews during the month of August). After August, reviews begin to decrease over the next few months with a final spike in December. This may be a suggestion as to the number of customer visits per month at Fogo de Chao.  

In addition, I wanted to investigate whether or not there were particular days of the month (1st-31st) that costumers preferred to visit.

![](/images/day.png)

This graph showed that people are more likely to eat at Fogo de Chao during the middle of the month.

Lastly, I split reviews into two parts: negative reviews(ratings <= 3) and positive reviews(ratings >= 4) and created a word cloud.

```python
## Word Cloud
fig, ax = plt.subplots(figsize = [16,12], nrows = 1, ncols = 2)
df_low = df.loc[df['ratings'] <= 3]
df_high = df.loc[df['ratings'] > 3]
stopwords_1 = set(STOPWORDS)
stopwords_1.update(['excited','w','ma','m','th','place','lot','brazilian','menu','ste','good','reviews','food','great','better','recommend','google','doesn','despite','except','object','try','maybe','t','saw','pan','dtype','hung'])
wordcloud1 = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords_1,
                min_font_size = 10).generate(str(df_low['reviews']))

# plot the WordCloud image                        
ax[0].imshow(wordcloud1)
ax[0].axis("off")
ax[0].set_title('Negative', size = 30)
#ax[0].tight_layout(pad = 0)
stopwords_2 = set(STOPWORDS)
stopwords_2.update(['fogo','chao','boston','de','restaurant','group','go','meal','seems','covid','place','li','name','dtype','object'])
wordcloud2 = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords_2,
                min_font_size = 10).generate(str(df_high['reviews']))

# plot the WordCloud image                        
ax[1].imshow(wordcloud2)
ax[1].axis("off")
ax[1].set_title('Positive', size = 30)
plt.show()
```

![](/images/wordcloud.png)

From the negative word cloud, we can formulate the main complaints: The restaurant was too busy, too expensive, and/or the meat did not taste delicious.

From the positive word cloud, we can gather the recurring compliments: The restaurant had a good atmosphere, good service (described as attentive), and had great food.

### Predictive Modeling
The visualizations offered some basic insights into Fogo de Chao's performance and consumer behavior. However, further questions rise: how are reviews and ratings related? Can reviews be used to predict ratings accurately? In order to address these questions, I cleaned the review text and built models that predict ratings based on the reviews.

First, I removed punctuations, and switched characters from upper case to lower case.
#### Text cleaning
```python
## Remove punctuation
def remove_punctuation(input_text):
        punct = string.punctuation
        trantab = str.maketrans(punct, len(punct)*' ')  
        return input_text.translate(trantab)

df['reviews'] = [i.lower() for i in df['reviews']]
df['reviews'] = df['reviews'].apply(remove_punctuation)
```

After cleaning the text, I imported machine learning packages and defined a function that calculated accuracy of confusion matrices. This allowed the model's prediction results to be within one star difference. In other words, if a review has 4 stars, a prediction of 3 stars or 5 stars is considered accurate for this model. 
#### Modeling
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.linear_model            import LinearRegression
from sklearn.linear_model            import Lasso
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

# confusion matrix accuracy
def right_pred_off1(cm):
    accuracy = (cm.diagonal(offset=-1).sum()+cm.diagonal(offset=0).sum()+cm.diagonal(offset=1).sum())/cm.sum()
    return accuracy
```
#### Create train-validating-test split

The next step was to create a train-validating-test split data on a 80-10-10 basis. I trained the models using train set, selected hyperparameters on validating the set, and compared models on test set. 

```python
df['ML_group']   = np.random.randint(100,size = df.shape[0])
df              = df.sort_values(by='ML_group').reset_index()

inx_train         = df.ML_group<80                     
inx_valid         = (df.ML_group>=80)&(df.ML_group<90)
inx_test          = (df.ML_group>=90)
```
The last step prior to modeling in sentiment analysis was to transform the text into vectors. I used **TfidfVectorizer** to transform the text and ignored **stop words** as they appear commonly and usually do not provide useful information.

```python
corpus          = df['reviews'].to_list()
ngram_range     = (1,1)
max_df          = 0.85
min_df          = 0.01
vectorizer      = TfidfVectorizer(lowercase   = True,
                                  ngram_range = ngram_range,
                                  max_df      = max_df     ,
                                  min_df      = min_df     ,
                                  stop_words = stopwords.words('english'))

X               = vectorizer.fit_transform(corpus)
Y         = df['ratings']
Y_train   = df['ratings'][inx_train].to_list()
Y_valid   = df['ratings'][inx_valid].to_list()
Y_test    = df['ratings'][inx_test].to_list()
X_train   = X[np.where(inx_train)[0],:].toarray()
X_valid   = X[np.where(inx_valid)[0],:].toarray()
X_test    = X[np.where(inx_test) [0],:].toarray()

test_score_model_off1 = {}
```
After these necessary preparations for modeling, I applied 8 different models. Below I offer 3 examples:

#### Linear Regression

```python
clf  = LinearRegression()
clf.fit(X_train, Y_train)
df['clf_hat'] = np.concatenate(
        [
                clf.predict(X_train),
                clf.predict(X_valid),
                clf.predict(X_test)
        ]
        ).round().astype(int)


df.loc[df['clf_hat'] > 5,'clf_hat'] = 5
df.loc[df['clf_hat'] < 1,'clf_hat'] = 1
cm_clf = confusion_matrix(Y_test, df['clf_hat'][inx_test])
right_pred_off1(cm_clf)
test_score_model_off1['Linear Model'] = round(right_pred_off1(cm_clf),3)

```

#### Lasso

Here, I performed hyperparameter tuning to find the best penalty (alpha) for Lasso.

```python
alpha_list = [0.001, 0.01, 0.1]
valid_score_off1 = []
for alpha in alpha_list:
    la_list = []
    la  = Lasso(alpha = alpha)
    la.fit(X_train, Y_train)
    la_list = np.concatenate(
        [
                la.predict(X_train),
                la.predict(X_valid),
                la.predict(X_test)
        ]
        ).round().astype(int)
    for i in range(len(la_list)):
        if la_list[i] > 5:
            la_list[i] = 5
        elif la_list[i] < 1:
            la_list[i] = 1

    cm_la = confusion_matrix(Y_valid, la_list[inx_valid])
    valid_score_off1.append(right_pred_off1(cm_la))

#print(np.argmax(valid_score_off1))

la  = Lasso(alpha = alpha_list[np.argmax(valid_score_off1)])
la.fit(X_train, Y_train)
la_final = np.concatenate(
        [
                la.predict(X_train),
                la.predict(X_valid),
                la.predict(X_test)
        ]
        ).round().astype(int)
for i in range(len(la_final)):
    if la_final[i] > 5:
        la_final[i] = 5
    elif la_final[i] < 1:
        la_final[i] = 1

cm_la = confusion_matrix(Y_test, la_final[inx_test])

test_score_model_off1['Lasso'] = round(right_pred_off1(cm_la),3)
```

#### Decision tree

```python
accuracy_tree         = []
criterion_list     = ['entropy','gini']
max_depth_list = list(range(2,11))
for i in criterion_list:
    for depth in max_depth_list:
        dtree    = tree.DecisionTreeClassifier(criterion    = i, 
                                               max_depth    = depth).fit(X_train, Y_train)
        cm_tree = confusion_matrix(dtree.predict(X_valid),Y_valid)
        accuracy_tree.append(right_pred_off1(cm_tree))

tree_best = np.argmax(accuracy_tree)
dtree    = tree.DecisionTreeClassifier(criterion= criterion_list[tree_best // len(max_depth_list)], 
                                     max_depth = max_depth_list[tree_best % len(max_depth_list)]).fit(X_train, Y_train)
cm_tree = confusion_matrix(dtree.predict(X_test),Y_test)

test_score_model_off1['Decision Tree'] = round(right_pred_off1(cm_tree),3)
```

#### Model results

Besides these above models, I also ran additional models, including Ridge regression, KNN, Naive Bayes, Random forest and SVC. I placed all of these models, along with their corresponding accuracy scores, into a dictionary **test_score_model_off1**. In the end, by comparing scores among models, I found that the best model in this case was Lasso, with an accuracy of 0.921. However, other models such as Ridge regression, Linear regression, and SVC worked well too.

![](/images/results.jpeg)



My full code can be found on [Github](https://github.com/changtaoshuai/Marketing_Analytics)
