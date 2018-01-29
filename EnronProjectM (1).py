
# coding: utf-8

# In[ ]:

## Identify fraud from Enron Email

### Enron became a symbol for fraud 
# The spectacular collapse of a giant american company in electric field was not only an end to the company
# but brought also a massive change for the american and global economy. 
# In 2001 Enron announced results for 3rd quarter and at the same time to a big surprise of shareholders a banckruptacy.
# In this project I'd like to focus on the most influencial workers of Enron which are obviously involved in the fraud.
# The most famous is CEO Jeffrey Keith "Jeff" Skilling and a chairman  Kenneth Lay of Enron 
# during most of the time when the crime occured.
# We will see their salaries, bonuses and stocks which are quite interesting.
# 

# Going through given dataset with e-mails within Enron co-workers we will discover a POI - Person of Interest,
# basically a person suspected to participate in the fraud.
# 


# In[52]:

# Firstly, let's load the necessary data and packages. 
# I am going to create a dataframe in pandas and then with the use of numpy arrays and matplotlib visualize it.
# To further analysis in classyfing will need Sklearn, GaussianNB,... i co jeszcze?

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.grid_search import GridSearchCV
from time import time

import pandas as pd
get_ipython().magic(u'matplotlib inline')
from matplotlib import pyplot as plt


### Let's load my dictionary providede by Udacity
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# In[53]:

# Let's check the type of my dataset, as we see it's a dictionary
type(data_dict)


# In[211]:

# By converting the dictionary to a dataframe with pandas it will be easier and faster to work with it:
enron_dataf = pd.DataFrame.from_records(list(data_dict.values()))


# In[212]:

# Previosly the index were numbers, but it's easier to set them as names of employees series:
employees = pd.Series(list(data_dict.keys()))
enron_dataf.set_index(employees, inplace=True)
enron_dataf.head()


# In[204]:

# Let's load the data and answer a few simple questions:


# In[207]:

# Here we see the whole data_set


# In[208]:

print enron_dataf


# In[218]:

# wydaje mi się że to nie ma sensu
d_enron_dict = pickle.load(open("final_project_dataset.pkl", "r") )


# In[219]:

# Let's see what features we have about the Enron's CEO:

enron_dataf.loc['SKILLING JEFFREY K']


# In[220]:

# The dataset contains information about 146 people


# In[222]:

print len(d_enron_dict.keys())


# In[196]:

print enron_dataf.keys()


# In[ ]:




# In[67]:

poi_names = open("poi_names.txt").read().split('\n')
poi_y = [name for name in poi_names if "(y)" in name]
print("poi_names_count:", len(poi_y))


# In[68]:

print poi_names + poi_y


# In[69]:

print len('poi_names') + len('poi_y')


# In[223]:

# to powinnam a  może z tym chairman?
print d_enron_dict['SKILLING JEFFREY K']
#print data_dict['SKILLING JEFFREY K']


# In[71]:

# to jest raczej bez sensu
#to powinnam zrobic z tym gościem głównym czyli prezesem
#print enron_dataf['SKILLING JEFFREY K']


# In[224]:

print d_enron_dict["SKILLING JEFFREY K"]["from_this_person_to_poi"]
#print enron_data["SKILLING JEFFREY K"]["from_this_person_to_poi"]


# In[225]:

print d_enron_dict["SKILLING JEFFREY K"]["exercised_stock_options"]
#print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]


# In[226]:

print d_enron_dict["SKILLING JEFFREY K"]["total_payments"]
#print enron_data["SKILLING JEFFREY K"]["total_payments"]


# In[ ]:




# In[ ]:




# In[75]:

## OUTLIERS
##


# In[240]:

import random
import numpy
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import pickle

from outlier_cleaner import outlierCleaner


# In[ ]:




# In[235]:

#chyba dam sobie z tym spokój
#enron_dataf.drop['TOTAL']


# In[ ]:




# In[228]:

# Let's focus on earning of employees, by making  a plot of salaries and bonus:

data_features = ["salary", "bonus"]

data = featureFormat(data_dict, data_features)

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")


# In[244]:

# As we can see there are some outliers that must be removed for further analysis.
# let's get rid off the total, which could cause a chaos. 
# After that we should remove all NANs and see the 6 top salaries as a list in Enron.

data_features = ["salary", "bonus"]

d_enron_dict.pop('TOTAL', 0)

data = featureFormat(d_enron_dict, features)

outliers = []
for key in data_dict:
    val = d_enron_dict[key]['salary']
    if val == 'NaN':
        continue
    outliers.append((key, int(val)))

outliers_final = (sorted(outliers,key=lambda x:x[1],reverse=True)[:6])

print outliers_final


# In[ ]:




# In[159]:

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[279]:

# Now we can see the graph without outliers. 
# 
d_enron_dict.pop('TOTAL',0)

data_features = ["salary", "bonus", "poi"]
data = featureFormat(d_enron_dict, data_features)


for point in data:
    salary = point[0]
    bonus = point[1]
    if point[2] == 1:
        matplotlib.pyplot.scatter( salary, bonus, color = 'orange' )
    else:
        matplotlib.pyplot.scatter( salary, bonus, color = 'grey' )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


# In[ ]:




# In[162]:




# In[163]:

#### FEATURES


# In[ ]:




# In[232]:

# Created two new features in ratios
enron_dataf[new_feature_from_meassages_to_poi_ratio] = enron_dataf['from_messages']/enron_dataf['from_poi_to_this_person']
enron_dataf[new_feature_to_messages_from_this_person_to_poi_ratio] = enron_dataf['to_messages']/enron_dataf['from_this_person_to_poi']


# In[ ]:

# Now let's focus on choosing the features to indentify a POI
# I will use only the provided features, POI, financial and email as per below:

# POI

# Features with email: 'from_messages', 'shared_receipt_with_poi',['fraction_mail_from_poi', 'fraction_mail_to_poi', 'from_poi_to_this_person', 'from_this_person_to_poi', 'to_messages', 'from_messages']

# Financial features: ['poi', 'salary', 'bonus','deferral_payments', 'expenses', 
#                 'restricted_stock_deferred', 'restricted_stock', 'deferred_income','total_payments',
#                 'exercised_stock_options', 'total_stock_value', 'restricted_stock']


# In[ ]:

features_list = ['poi', 'salary', 'bonus', 'deferral_payments', 'expenses', 
                 'restricted_stock_deferred', 'restricted_stock', 'deferred_income','total_payments',
                 'fraction_mail_to_poi', 'from_poi_to_this_person', 'from_this_person_to_poi', 'to_messages', 
                 'from_messages', 'shared_receipt_with_poi', 'exercised_stock_options',
                'total_stock_value', 'restricted_stock']


# In[254]:


enron_dataf = pickle.load(open("final_project_dataset.pkl", "r") )


# In[ ]:

# teraz ma być to ale nie bardzo wiem jak
#2. Feature processing of the Enron dataset


# In[257]:

# zmien to tez prosze
### split data into training and testing datasets
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, 
                                                                                             random_state=42)


# In[ ]:




# In[268]:

from sklearn.naive_bayes import GaussianNB

features_list = ["poi", "salary", "bonus", 'deferral_payments', 'expenses', 
                 'restricted_stock_deferred', 'restricted_stock', 'deferred_income','total_payments',
                 'fraction_mail_to_poi', 'from_poi_to_this_person', 'from_this_person_to_poi', 'to_messages', 
                 'from_messages', 'shared_receipt_with_poi', 'exercised_stock_options',
                'total_stock_value', 'restricted_stock']
t0 = time()
#zobacz czy mozna zmienic nazwe tego accuracy score i skad sie to wzielo
clf = GaussianNB()
clf.fit(features_train, labels_train)
accuracy = accuracy_score(labels_test, prediction)


print "Accuracy for GaussianNB:", accuracy

print "GaussianNB time of running algorithm:", round(time()-t0, 3), "s"


# In[ ]:




# In[269]:

## Boże to dziala!!
# Another classifer is a Decision Tree,
# it gives certainly bigger accuracy


from sklearn.tree import DecisionTreeClassifier

t0 = time()

clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)
pred= clf.predict(features_test)
print 'accuracy', score

print "Decision tree algorithm time:", round(time()-t0, 3), "s"


# In[ ]:




# In[ ]:




# In[270]:

from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(features_train, labels_train)


# In[ ]:




# In[276]:

from sklearn.neighbors import KNeighborsClassifier


clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)
print "testing time: ", round(time()-t0, 3), "s"
print "Accuracy of DT classifer is  : ",accuracy_score(labels_test, prediction)


# In[ ]:




# In[ ]:




# In[274]:


clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)
print "testing time: ", round(time()-t0, 3), "s"


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



