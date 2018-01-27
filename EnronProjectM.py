
# coding: utf-8

# In[ ]:

### Enron became a symbol for fraud 
# The spectacular collapse of a giant american company in electrin field was not only an end to the company
# but brought also a massive change for the american and global economy. 
# In 2001 Enron announced results for 3rd quarter and at the same time to a big surprise of shareholders a banckruptacy.
# In this project I'd like to focues on the most influence ppl in the company which are obviously the ones involved in the fraud.
# The most famous is CEO Jeffrey Keith "Jeff" Skilling, we will find out about his secretes.
# We will see their salaries, bonuses and stocks. 
# And Kenneth Lay who was a chairman of Enron during most of the time when the crime occured.
# 

#


# In[ ]:

# Let's load the data and answer a few simple questions:


# In[1]:

import pickle

enron_data = pickle.load(open("final_project_dataset.pkl", "r"))


# In[2]:

import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


# In[3]:

# Here we see the whole data_set


# In[4]:

print enron_data


# In[5]:

d_enron_dict = pickle.load(open("final_project_dataset.pkl", "r") )


# In[6]:

# The dataset contains information about 146 people


# In[7]:

print len(d_enron_dict.keys())


# In[8]:

print enron_data.keys()


# In[ ]:




# In[9]:

# to chyba nie jest właściwie bo ten gosc ma 21

print len("features_list")


# In[10]:

## to na pewno muszę zmienić bo to czysta kopia
# There is 18 POI in the dataset

pois = 0
nantppoi = 0.0
for e in enron_data:
	if enron_data[e]['poi'] == True:
		pois += 1
		if enron_data[e]['total_payments'] == 'NaN':
			nantppoi += 1
print 'POIs: ' + str(pois)
print 'NaN Total Payments POIs: ' + str(nantppoi/pois*100.0)


# In[11]:

poi_names = open("poi_names.txt").read().split('\n')
poi_y = [name for name in poi_names if "(y)" in name]
print("poi_names_count:", len(poi_y))


# In[12]:

print poi_names + poi_y


# In[13]:

print len('poi_names') + len('poi_y')


# In[15]:

# to powinnam zrobic z tym gościem głównym czyli prezesem
print enron_data['SKILLING JEFFREY K']


# In[16]:

print enron_data["SKILLING JEFFREY K"]["from_this_person_to_poi"]


# In[17]:

print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]


# In[18]:

print enron_data["SKILLING JEFFREY K"]["total_payments"]


# In[ ]:




# In[ ]:




# In[19]:

## OUTLIERS
##to jest tez kopia wiec uwazaj


# In[20]:

import random
import numpy
import matplotlib.pyplot as plt
import pickle

from outlier_cleaner import outlierCleaner


# In[21]:


### load up some practice data with outliers in it
ages = pickle.load( open("practice_outliers_ages.pkl", "r") )
net_worths = pickle.load( open("practice_outliers_net_worths.pkl", "r") )


# In[22]:

### ages and net_worths need to be reshaped into 2D numpy arrays
### second argument of reshape command is a tuple of integers: (n_rows, n_columns)
### by convention, n_rows is the number of data points
### and n_columns is the number of features
ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))
from sklearn.cross_validation import train_test_split
ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths, test_size=0.1, random_state=42)

### fill in a regression here!  Name the regression object reg so that
### the plotting code below works, and you can see what your regression looks like


# In[23]:

## to ejst chyba to validation czy cos to zobacz w tym video czy to to
from sklearn.cross_validation import train_test_split
ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths, test_size=0.1, random_state=42)


# In[ ]:

from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(ages_train, net_worths_train)


# In[25]:

### to jest chyba tylko do lekcji a nie do projektu
### fill in a regression here!  Name the regression object reg so that
### the plotting code below works, and you can see what your regression looks like

from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(ages_train, net_worths_train)


# In[26]:

print 'The slope for this regression is : ', reg.coef_


# In[48]:

print "The score when using regression to make predictions with the test data : ", reg.score(ages_test,net_worths_test)


# In[49]:

features = ["salary", "bonus"]
#data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)
### plot features
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
#plt.show()


# In[50]:

features = ["salary", "bonus"]
data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)

### remove NAN's from dataset
outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    outliers.append((key, int(val)))

outliers_final = (sorted(outliers,key=lambda x:x[1],reverse=True)[:4])
### print top 4 salaries
print outliers_final


# In[51]:

data_dict.pop('TOTAL',0)
features = ["salary", "bonus", "poi"]
data = featureFormat(data_dict, features)


for point in data:
    salary = point[0]
    bonus = point[1]
    if point[2] == 1:
        matplotlib.pyplot.scatter( salary, bonus, color = 'red' )
    else:
        matplotlib.pyplot.scatter( salary, bonus, color = 'blue' )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


# In[55]:

import pandas as pd
d_enron = pd.read_csv('final_project_dataset.pkl')


# In[56]:

#z tym sobie chyba dam spokój 
d_enron.head()


# In[ ]:

#### FEATURES


# In[41]:

def Feature_new_list(key,normalizer):
    new_list=[]

    for i in enron_data:
        if enron_data[i][key]=="NaN" or enron_data[i][normalizer]=="NaN":
            new_list.append(0.)
        elif enron_data[i][key]>=0:
            new_list.append(float(enron_data[i][key])/float(enron_data[i][normalizer]))
    return new_list


new_feature_from_meassages_to_poi_ratio = ['from_messages']/['from_poi_to_this_person']
new_feature_to_messages_from_this_person_to_poi_ratio= ['to_messages']/['from_this_person_to_poi']


count=0
for i in enron_data:
    enron_data["new_feature_from_meassages_to_poi_ratio"][i] = new_feature_from_meassages_to_poi_rario[count]
    enron_data["new_feature_to_messages_from_this_person_to_poi_ratio"][i] = new_feature_to_messages_from_this_person_to_poi_ratio[count]
    count +=1
    
##zmien to please   
        
features_new_list = ["poi", "new_feature_from_meassages_from_poi_ratio", "new_feature_to_messages_from_this_person_to_poi_ratio"]    
    ### store to my_dataset for easy export below
me_data = enron_data


### these two lines extract the features specified in features_list
### and extract them from data_dict, returning a numpy array
data = featureFormat(my_dataset, features_list)


# In[ ]:




# In[ ]:




# In[ ]:



