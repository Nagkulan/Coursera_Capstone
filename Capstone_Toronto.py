#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# ### Installing LXML since Kernel shoots 'LXML' Error while reading the Wikipedia Page

# In[3]:


pip install lxml


# In[4]:


import lxml


# In[5]:


url='https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'


# In[6]:


dfs = pd.read_html(url)


# In[7]:


frame=dfs[0:1]
frame


# In[8]:


data=frame[0]


# In[9]:


data


# In[10]:


data.shape


# In[11]:


data.head()


# #### Setting the Borough as Index and deleting Indexes which are equal to 'Not assigned'

# In[12]:


data = data.set_index("Borough")


# In[13]:


data=data.drop('Not assigned',axis=0)


# In[14]:


data.reset_index(inplace=True)


# #### Checking for 'Not assigned' values in column 'Neighbourhood'

# In[15]:


(data['Neighbourhood']=='Not assigned').value_counts()


# In[16]:


print(data.shape)
data.head()


# In[17]:


df=data


# In[18]:


df


# In[19]:


print(df['Postcode'].value_counts())


# In[20]:


table=df.groupby(['Postcode','Borough'],as_index=False, sort=False).agg(','.join)


# ### Having a look at the values corresponding to 'M9C' in the Original table df

# In[21]:


for i in df.index:
    if df.iloc[i,1] == 'M9V':
        print(df.iloc[i,2])


# ### Checking number of 'M9V' in TABLE that is grouped-by Postcode

# In[22]:


for i in table.index:
    if table.iloc[i,0] == 'M9V':
        print(table.iloc[i,0])


# ### Checking if all the values corresponding to 'M9V' are present in it's row

# In[23]:


print(table[table['Postcode']=='M9V'])
table.loc[89,'Neighbourhood']


# In[24]:


table


# ## Another Algorithm I tried to group the values as per 'Postcode', which did not work

# ### for i in range(0,211):
#         for j in range(i+1,211):
#             if df['Postcode'][i]==df['Postcode'][j]:
#                 #print(df['Neighbourhood'][i],df['Neighbourhood'][j])
#                 df['Neighbourhood'][i]=df['Neighbourhood'][i]+', '+df['Neighbourhood'][j]
#                 df.drop(j,inplace=True)
#             else:
#                 print('Invalid')

# ### Cross-Checking if there are any duplicate values in columns 'Postcode'

# In[25]:


print(table['Postcode'].value_counts())


# In[26]:


(table['Neighbourhood']=='Not Assigned').value_counts()


# ### Getting Latitude Longitude Coordinates using Geocoder, which did not work as expected by the Faculty

# In[28]:


pip install geocoder


# In[29]:


import geocoder


# In[30]:


g = geocoder.google('Mountain View, CA')
print(g.latlng)


# In[31]:


latlng=None


# In[71]:


table.columns


# In[48]:


print(table.iloc[0,0],table.iloc[0,1])


# In[70]:


latlng


# In[ ]:


#for i in table.index:
while(latlng is None):
    g = geocoder.google('{}, {}, {}'.format(table.iloc[0,0],table.iloc[0,1],table.iloc[0,2]))
    latlng=g


# In[348]:


for i in range(0,10):
    print('{}, {}, {}'.format(table.iloc[i,0],table.iloc[i,1],table.iloc[i,2]))


# ## Making use of the file provided to get Latitudes and Longitudes

# In[27]:


file = pd.read_csv('http://cocl.us/Geospatial_data')


# In[28]:


file


# In[29]:


file.iloc[0,1]


# ### Inserting Lat and Lng Columns in the Table

# In[30]:


Latitude=[]
Longitude=[]


# In[31]:


for j in table.index:
    for i in file.index:
        if table.iloc[j,0]==file.iloc[i,0]:
            Latitude.append(file.iloc[i,1])
            Longitude.append(file.iloc[i,2])


# In[32]:


table['Latitude']=Latitude


# In[33]:


table['Longitude']=Longitude


# In[37]:


table


# In[34]:


table['Borough'].values


# In[35]:


tor=table[table['Borough'].str.contains("Toronto")]


# In[36]:


tor.reset_index(drop=True,inplace=True)


# In[37]:


tor.head()


# In[38]:


from sklearn.cluster import KMeans
import folium


# In[39]:


map = folium.Map(location=[43.6532,-79.3832])


# In[41]:


for lat, lng, br, nh in zip(tor['Latitude'],tor['Longitude'],tor['Borough'],tor['Neighbourhood']):
    label='{}. {}'.format(nh,br)
    label=folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat,lng],
        radius=3,
        popup=label, color='red', fill=True, parse_htmal=False).add_to(map)


# In[42]:


map


# In[43]:


import matplotlib.pyplot as plt


# ## Obtaining Optimal k

# In[44]:


k=range(1,10)
kmeans=[KMeans(n_clusters=i) for i in k]

yaxis=tor[['Latitude']]
score=[kmeans[i].fit(yaxis).score(yaxis) for i in range(len(k))]

plt.plot(k,score)


# In[45]:


kmeans=KMeans(n_clusters=3,init='k-means++')
kmeans.fit(tor[['Latitude','Longitude']])


# In[46]:


tor['Label']=kmeans.fit_predict(tor[['Latitude','Longitude']])
centers=kmeans.cluster_centers_


# In[47]:


tor.head()


# In[48]:


tor.shape


# In[49]:


centers


# In[50]:


labels = kmeans.predict(tor[['Latitude','Longitude']])


# In[54]:


kcolor=3
labels


# ## Visualising the Clusters

# In[58]:


import matplotlib.cm as cm
import matplotlib.colors as colors


# In[63]:


x = np.arange(kcolor)
ys = [i + x + (i*x)**2 for i in range(kcolor)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

markercolor=[]
for lat, lng, nh, cluster in zip(tor['Latitude'],tor['Longitude'],tor['Neighbourhood'],tor['Label']):
    label=folium.Popup(nh + ' Cluster' + str(cluster),parse_html=True)
    folium.CircleMarker(
        [lat,lng], radius=3, popu=label, color=rainbow[cluster-1]).add_to(map)
    
map


# ## Visualising Map clusters

# In[64]:


from folium import plugins


# In[69]:


clusters=plugins.MarkerCluster().add_to(map)

for lat, lng, label, in zip(tor.Latitude, tor.Longitude, tor.Label):
    folium.Marker(
        location=[lat, lng],
        icon=None,
        
    ).add_to(clusters)


# In[70]:


map


# In[ ]:




