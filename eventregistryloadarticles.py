
# coding: utf-8

# In[1]:

from eventregistry import *
er = EventRegistry()
er.login(EMAIL,PASSWORD)


# In[25]:

#to query articles

'''q = QueryArticles()
# set the date limit of interest
q.setDateLimit(datetime.date(2016, 12, 1), datetime.date(2016, 12, 8))
# find articles mentioning the Academy Awards
q.addNewsSource(er.getNewsSourceUri("Los Angeles Times"))
q.addConcept(er.getConceptUri("Donald Trump"))
# return the list of top 200 articles
q.addRequestedResult(RequestArticlesInfo(page = 1, count = 20, 
    returnInfo = ReturnInfo(articleInfo = ArticleInfoFlags(eventUri = False))))
res = er.execQuery(q)
print(er.format(res))
with open('LA-Times.json', 'w') as f:
    f.write(er.format(res))'''


# In[1]:

#create pandas dataframe from article data

import pandas as pd
import numpy as np

columns = ['headline', 'source']
df = pd.DataFrame(data=np.zeros((0,len(columns))), columns=columns)

from json import loads

sources = ["Los Angeles Times","The New York Times", "Breitbart", "The Huffington Post", "ThinkProgress", "National Review Online","Fox News","CNN International","NBC News","The Wall Street Journal","Entertainment Weekly","New York Post","Daily News"]

for source in sources:  
    data = open('{}.json'.format(source),'r').read()
    results = loads(data)["articles"]["results"]
    for result in results:
        df = df.append({'headline':result["title"], 'source':result["source"]["title"]},ignore_index=True)
    print "Loaded {}".format(source)


# In[4]:

#split headlines into feature vectors
df.headline = df.headline.str.lower().str.split()


# In[6]:

#randomize order
df = df.sample(frac=1)
D = df.headline
c = df.source


# In[7]:

#split into train, development and test

D_train, D_validate, D_test = np.split(D, [int(.7*len(D)), int(.85*len(D))])
c_train, c_validate, c_test = np.split(c, [int(.7*len(c)), int(.85*len(c))])


# In[ ]:



