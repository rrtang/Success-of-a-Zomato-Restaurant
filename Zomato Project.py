#!/usr/bin/env python
# coding: utf-8

# In[1]:


# standard libraries
import pandas as pd
import numpy as np
import re
from warnings import filterwarnings
filterwarnings('ignore')
# visulization libraries
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import iplot
from geopy.geocoders import Nominatim
import folium
from folium.plugins import HeatMap
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# In[2]:


# import dataset
df = pd.read_csv("/Users/RRT/Desktop/自学/zomato project/zomato.csv")


# In[3]:


### 1. understand dataset


# In[4]:


# dataset summary
df.info()


# In[5]:


df.head()


# In[208]:


# get features having null values
feature_na = [feature for feature in df.columns if df[feature].isnull().sum() > 1]
# get null value percentage for feature in feature_na
for feature in feature_na:
    print("{} has {}% missing values".format(feature, np.round(df[feature].isnull().sum()/len(df)*100,4)))


# In[209]:


### 2. data cleaning


# In[210]:


# 2.1 perform data cleaning on "approx_cost" feature


# In[211]:


df["approx_cost(for two people)"].dtype


# In[212]:


### check unique values in "approx_cost"
df["approx_cost(for two people)"].unique()


# In[213]:


# using lambda function to replace "," with ""(first convert datatype to str)
df["approx_cost(for two people)"] = df["approx_cost(for two people)"].astype(str).apply(lambda x:x.replace(",", ""))


# In[214]:


# convert datatype to float
df["approx_cost(for two people)"] = df["approx_cost(for two people)"].astype(float)
df["approx_cost(for two people)"].unique()


# In[215]:


### 2.2 perform data cleaning on "rate" feature 


# In[216]:


# check unique values in "rate"
df["rate"].unique()


# In[217]:


# count null values in "rate"
df["rate"].isnull().sum()


# In[218]:


# define function to split rate and keep only the first number, drop"/5"
def split(x):
    return x.split("/")[0]


# In[219]:


# first convert dtype to string, then apply split func
df["rate"] = df["rate"].astype(str).apply(split)
df["rate"].unique()


# In[220]:


# replace dirty value "-" and "NEW" with 0
df["rate"] = df["rate"].replace("-", 0)
df["rate"] = df["rate"].replace("NEW", 0)


# In[221]:


# convert datatype to float
df["rate"] = df["rate"].astype(float)
df["rate"].unique()


# In[222]:


df["rate"].isnull().sum()


# In[33]:


### 3. analyze categories of restaurants in the dataset


# In[19]:


# how many type of retaurants do we have?
plt.figure(figsize=(20,12))
# count the 20 most popular retaurant types
df["rest_type"].value_counts().nlargest(20).plot.bar()


# In[65]:


# define function to assign restaurant type to "Quick Bites + Casual Dining" and "others"
def mark(x):
    if x in ("Quick Bites", "Casual Dining"):
        return "Quick Bites + Casual Dining"
    else:
        return "others"


# In[66]:


# apply mark func to "rest_type"
df["Top types"] = df["rest_type"].apply(mark)


# In[67]:


# pie chart plot to visualize how many restaurants belong to "Quick Bites + Casual Dining"
values = df["Top types"].value_counts().values
labels = df["Top types"].value_counts().index
fig = px.pie(df, names = labels, values = values)
fig.show()


# In[76]:


### 4. find 5 most and least popular restaurants


# In[69]:


# create rest dataframe containing four attributes
rest = df.groupby("name").agg({"votes":"sum", "url":"count", "approx_cost(for two people)":"mean", "rate":"mean"}).reset_index()


# In[70]:


# rename columns of rest
rest.columns = ["name", "total_votes", "total_unities", "avg_approx_cost", "mean_rating"]
rest.head()


# In[71]:


# create new feature votes_per_unity
rest["votes_per_unity"] = rest["total_votes"]/rest["total_unities"]
rest.head()


# In[74]:


# sort rest by "total_unities"
popular = rest.sort_values(by="total_unities", ascending=False)
popular.head(10)


# In[75]:


# visualize the combination of:
# 1. avg votes received by restaurant
# 2. Top 5 most votes restaurant
# 3. Top 5 less votes restaurant


# In[29]:


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 30))
# avg votes received by restaurant
ax1.text(0.5, 0.3, int(popular["total_votes"].mean()), fontsize = 45, ha = "center")
ax1.text(0.5, 0.12, "is the average votes", fontsize = 12, ha = "center")
ax1.text(0.5, 0, "received by restaurants", fontsize = 12, ha = "center")
ax1.axis("off")
# Top 5 most votes restaurant
sns.barplot(x = "total_votes", y="name", 
            data = popular.sort_values("total_votes", ascending=False).query("total_votes > 0").head(5), ax = ax2)
ax2.set_title("Top 5 most votes rest")
# Top 5 less votes restaurant
sns.barplot(x = "total_votes", y="name", 
            data = popular.sort_values("total_votes", ascending=False).query("total_votes > 0").tail(5), ax = ax3)
ax3.set_title("Top 5 least votes rest")


# In[30]:


### 5. perform in-depth analysis of restaurant


# In[31]:


# 5.1 find most expensive & cheapest restaurant


# In[32]:


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 30))
# avg_approx_cost for Bangaluru restaurants
ax1.text(0.5, 0.3, int(popular["avg_approx_cost"].mean()), fontsize = 45, ha = "center")
ax1.text(0.5, 0.12, "is the mean approx_cost", fontsize = 12, ha = "center")
ax1.text(0.5, 0, "for Bangaluru restaurants", fontsize = 12, ha = "center")
ax1.axis("off")
# Top 5 most expensive rest
sns.barplot(x = "avg_approx_cost", y="name", 
            data = popular.sort_values("avg_approx_cost", ascending=False).query("avg_approx_cost > 0").head(5), ax = ax2)
ax2.set_title("Top 5 most expensive rest")
# Top 5 cheapest rest
sns.barplot(x = "avg_approx_cost", y="name", 
            data = popular.sort_values("avg_approx_cost", ascending=False).query("avg_approx_cost > 0").tail(5), ax = ax3)
ax3.set_title("Top 5 cheapest rest")


# In[33]:


# 5.2 How many rest offer booke table service? And what about online order service?


# In[77]:


# pie chart to visualize "book" or "no book"
values = df["book_table"].value_counts()
labels = ["not book", "book"]
trace = go.Pie(labels = labels, values = values, hoverinfo="label+percent", textinfo = "percent")
iplot([trace])


# In[78]:


# pie chart to visualize "accept online_order" or "not accepted"
values = df["online_order"].value_counts()
labels = ["accepted", "not accepted"]
fig = px.pie(df, values = values, names = labels, title="pie chart")
fig.show()


# In[36]:


### 6. Find best budget restaurant in any location with rating > 4


# In[124]:


def return_budget(location, restaurant_type):
    filter = (df["approx_cost(for two people)"] <= 400) & (df["location"] == location) & (df["rate"] >= 4) & (df["rest_type"] == restaurant_type)
    best_budget = df[filter]
    return (best_budget["name"].unique())


# In[125]:


return_budget("BTM", "Quick Bites")


# In[126]:


### 7. perform geographical analysis on dataset


# In[142]:


# create basemap for bangaluru city
basemap = folium.Map(location=[12.97, 77.59])
basemap 


# In[162]:


# create Rest_location df with counts for each location
Rest_location = df["location"].value_counts(dropna=False).reset_index()
Rest_location.columns = ["name", "count"]


# In[163]:


# extract Latitude & Longitude for any place
geolocator = Nominatim(user_agent="app")
lat=[]
lon= []
for location in Rest_location["name"]:
    location = geolocator.geocode(location)
    if location is None:
        lat.append(np.nan)
        lon.append(np.nan)
    else:
        lat.append(location.latitude)
        lon.append(location.longitude)
Rest_location["latitude"] = lat
Rest_location["longitude"] = lon


# In[164]:


# drop nan values
Rest_location = Rest_location.dropna()


# In[165]:


# add each location counts to basemap
HeatMap(data=Rest_location[["latitude", "longitude", "count"]]).add_to(basemap)
basemap


# In[166]:


# 8. Analyse customer behaviour using Wordcloud of a particular type restaurant: "Quick Bites"


# In[168]:


# store all Quick Bites data into quickbites df
quickbites = df[df["rest_type"]=="Quick Bites"]
quickbites.head()


# In[169]:


# create dishes str to store all liked dishes
dishes = ""
for word in quickbites["dish_liked"].dropna():
    words = word.split()
    for item in words:
        item = item.lower()
    dishes = dishes + " ".join(words) + " "


# In[170]:


# set stopwords
stopwords = set(STOPWORDS)
# apply wordcloud
wordcloud = WordCloud(stopwords = stopwords, width = 1500, height = 1500).generate(dishes)
# show wordcloud
plt.imshow(wordcloud)
# remove axis
plt.axis("off")


# In[48]:


### 9. Analyze review of a particular type restaurant: "Quick Bites"


# In[171]:


# use "Quick Bites" dataset above
# data cleaning on "reviews_list"
total_review = ""
for review in quickbites["reviews_list"]:
    review = review.lower()
    # remove any character that IS NOT a-z OR A-Z
    review = re.sub("[^a-zA-z]", " ", review)
    # remove all "rated" character
    review = re.sub("rated", " ", review)
    # remove all "x"
    review = re.sub("x", " ", review)
    # remove multiple space
    review = re.sub(" +", " ", review)
    total_review = total_review + str(review)


# In[172]:


# apply wordcloud
wordcloud2 = WordCloud(stopwords = stopwords, width = 1500, height = 1500).generate(total_review)
# show wordcloud
plt.imshow(wordcloud2)
# remove axis
plt.axis("off")


# In[ ]:





# In[51]:


### 10. prepare data for machine learning 


# In[52]:


# divide dataset into new and old restaurant
# define function to divide dataset
def assign(x):
    if x > 0:
        return 1
    else:
        return 0
# apply assign func to 'rate' column
df["rated"] = df["rate"].apply(assign)


# In[53]:


# create two datasets based on "rated" value
new_res = df[df["rated"]==0]
train_res = df.query("rated == 1")
train_res.info()


# In[54]:


# create target value based on "rate" value: rate<3.75 is bad--0, rate>=3.75 is bad--1
threshold = 3.75
train_res["target"] = train_res["rate"].apply(lambda x:1 if x >= threshold else 0)


# In[55]:


# check if dataset is imbalanced
count = train_res["target"].value_counts()
labels = count.index
# create pie chart to visulize data composition
plt.pie(count, labels = labels)


# In[56]:


# perform feature selection


# In[57]:


# define function count
def count(x):
    return len(x.split(","))


# In[58]:


# apply count func to column "cuisines" and "rest_type"
train_res["total_cuisions"] = train_res["cuisines"].astype(str).apply(count)
train_res["multiple_types"] = train_res["rest_type"].astype(str).apply(count)
train_res.head()


# In[59]:


# select import features based on 
train_res.columns
imp_features = ['online_order', 'book_table',
       'location', 'rest_type',
       'approx_cost(for two people)',
       'listed_in(type)', 'listed_in(city)', 
       'target','total_cuisions', 'multiple_types']
train_new = train_res[imp_features]
train_new.head()


# In[60]:


# deal with missing values


# In[61]:


# check number of missing values
train_new.isnull().sum()


# In[66]:


# since missing values is few, just delete those rows
train_new = train_new.dropna(how="any")
train_new.head()


# In[67]:


# extract all "object" columns
catogorical_col = [col for col in train_new.columns if train_new[col].dtype == "O"]
# extract all numerical columns
numerical_col = [col for col in train_new.columns if train_new[col].dtype != "O"]


# In[69]:


# check how many unique values in each column
for feature in catogorical_col:
    print("{} has in total {} unique values".format(feature, train_new[feature].nunique()))


# In[70]:


for feature in numerical_col:
    print("{} has in total {} unique values".format(feature, train_new[feature].nunique()))


# In[71]:


# perform feature encoding on data


# In[72]:


# perform one hot encoding to convert string to numbers
# first consider 'location' column
ratio = train_new["location"].value_counts()/len(train_new)*100
# for ratio < 0.4, set to "other"
threshold = 0.4
imp = ratio[ratio > threshold]
# using np.where to recatogrize location
train_new["location"] = np.where(train_new["location"].isin(imp.index), train_new["location"], "other")
train_new["location"].nunique()


# In[73]:


# second consider "rest_type" column
ratio2 = train_new["rest_type"].value_counts()/len(train_new)*100
# for ratio < 1.5, set to "other"
threshold2 = 1.5
imp2 = ratio2[ratio2 > threshold2]
# using np.where to recariogrize location
train_new["rest_type"] = np.where(train_new["rest_type"].isin(imp2.index), train_new["rest_type"], "other")
train_new["rest_type"].nunique()


# In[74]:


# check angain how many unique values in each column
for feature in catogorical_col:
    print("{} has in total {} unique values".format(feature, train_new[feature].nunique()))


# In[75]:


# perform one hot encoding
train_new_cat = train_new[catogorical_col]
for col in catogorical_col:
    col_encoded = pd.get_dummies(train_new_cat[col], prefix = col, drop_first = True)
    train_new_cat = pd.concat([train_new_cat, col_encoded], axis = 1)
    train_new_cat = train_new_cat.drop(col, axis = 1)


# In[76]:


train_new_cat.shape


# In[77]:


train_new_cat.head()


# In[78]:


# concatenate all numeric and categorical data
train_final = pd.concat([train_new.loc[:, ['approx_cost(for two people)', 'target', 'total_cuisions', 'multiple_types']], train_new_cat], axis=1)


# In[79]:


train_final.head()
train_final.shape


# In[82]:


# split train_final into x and y
x = train_final.drop("target", axis = 1)
y = train_final["target"]
# split data into train and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
# apply randomforest algorithm
model_randomf = RandomForestClassifier()
model_randomf.fit(x_train, y_train)
# fit test data after training
predict_randomf = model_randomf.predict(x_test)


# In[83]:


# check accuracy
confusion_matrix(predict_randomf, y_test)
accuracy_score(predict_randomf, y_test)


# In[86]:


# apply 
models = []
models.append(["LogisticRegression", LogisticRegression()])
models.append(["Random Forest", RandomForestClassifier()])
models.append(["Decision Tree", DecisionTreeClassifier()])
models.append(["KNN", KNeighborsClassifier()])


# In[87]:


for name, model in models:
    print(name)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    cm = confusion_matrix(predictions, y_test)
    print(cm)
    acc = accuracy_score(predictions, y_test)
    print(acc)
    print("\n")

