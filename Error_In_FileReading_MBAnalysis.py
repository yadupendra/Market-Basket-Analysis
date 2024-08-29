#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[53]:


import pandas as pd
from collections import defaultdict

import matplotlib.pyplot as plt

import seaborn as sns
import datetime as dt


# # Why do we need to add "r" before the filepath and "on_bad_lines="skip"

# In[16]:


df = pd.read_csv(r"D:\KaggleDataset\attribution data.csv", on_bad_lines='skip')


# In[12]:


#Reading the Excel file
#Ecxcelfile doesn't require 'on_bad_line='skip'
df2 = pd.read_excel(r"D:\KaggleDataset\archive4MarketBasketAnalysis\Assignment-1_Data.xlsx")


# In[14]:


print(df2.head(10))


# In[18]:


print(df.head())


# In[25]:


#Getting the total number of rows and columns in the dataframe.
rows, columns = df.shape
print(f"Total number of rows: {rows}")
print(f"Total number of columns: {columns}")


# In[26]:


# OR additional way to display the shape of the dataframe
df.shape


# In[35]:


#Sorting the rows in the DF based on values in two columns- cookie and time.
df = df.sort_values(['cookie', 'time'],
                    ascending=[False, True])
print(df.head())


# In[36]:


# New column indicating the order of the touch-points for each user
df['visit_order'] = df.groupby('cookie').cumcount() + 1
df.head()


# # Pivoting the data frame from long-form to wide-form, so we ultimately get a single row per user
# 

# In[37]:


# grouping the chronological touch-points into a list
df_paths = df.groupby('cookie')['channel'].aggregate(
    lambda x: x.unique().tolist()).reset_index()
    
# merging the list of final conversion/non-conversion events onto that data    
df_last_interaction = df.drop_duplicates('cookie', keep='last')[['cookie', 'conversion']]

df_paths = pd.merge(df_paths, df_last_interaction, how='left', on='cookie')

# adding a “Null” or “Conversion” event to the end of our user-journey lists
df_paths['path'] = np.where(df_paths['conversion'] == 0,
                            ['Start, '] + df_paths['channel'].apply(', '.join) + [', Null'],
                            ['Start, '] + df_paths['channel'].apply(', '.join) + [', Conversion'])

df_paths['path'] = df_paths['path'].str.split(', ')

df_paths = df_paths[['cookie', 'path']]


# In[39]:


df_paths.head()


# # Markov Chains

# # defining a list of all user journeys, the number of total conversion and the base level conversion rate.

# In[41]:


list_of_paths = df_paths['path']
total_conversions = sum(path.count('Conversion') for path in df_paths['path'].tolist())
base_conversion_rate = total_conversions / len(list_of_paths)

total_conversions


# In[42]:


base_conversion_rate


# # define a function that identifies all potential state transitions and outputs a dictionary containing these. We’ll use this as an input when calculating transition probabilities:

# In[43]:


def transition_states(list_of_paths):
    list_of_unique_channels = set(x for element in list_of_paths for x in element)
    transition_states = {x + '>' + y: 0 for x in list_of_unique_channels for y in list_of_unique_channels}

    for possible_state in list_of_unique_channels:
        if possible_state not in ['Conversion', 'Null']:
            for user_path in list_of_paths:
                if possible_state in user_path:
                    indices = [i for i, s in enumerate(user_path) if possible_state in s]
                    for col in indices:
                        transition_states[user_path[col] + '>' + user_path[col + 1]] += 1

    return transition_states


trans_states = transition_states(list_of_paths)


# In[47]:


## function to calculate all transition probabilities 
def transition_prob(trans_dict):
    list_of_unique_channels = set(x for element in list_of_paths for x in element)
    trans_prob = defaultdict(dict)
    for state in list_of_unique_channels:
        if state not in ['Conversion', 'Null']:
            counter = 0
            index = [i for i, s in enumerate(trans_dict) if state + '>' in s]
            for col in index:
                if trans_dict[list(trans_dict)[col]] > 0:
                    counter += trans_dict[list(trans_dict)[col]]
            for col in index:
                if trans_dict[list(trans_dict)[col]] > 0:
                    state_prob = float((trans_dict[list(trans_dict)[col]])) / float(counter)
                    trans_prob[list(trans_dict)[col]] = state_prob

    return trans_prob


trans_prob = transition_prob(trans_states)
trans_prob


# In[56]:


def transition_matrix(list_of_paths, transition_probabilities):
    trans_matrix = pd.DataFrame()
    list_of_unique_channels = set(x for element in list_of_paths for x in element)

    for channel in list_of_unique_channels:
        trans_matrix[channel] = 0.00
        trans_matrix.loc[channel] = 0.00
        trans_matrix.loc[channel][channel] = 1.0 if channel in ['Conversion', 'Null'] else 0.0

    for key, value in transition_probabilities.items():
        origin, destination = key.split('>')
        trans_matrix.at[origin, destination] = value

    return trans_matrix


trans_matrix = transition_matrix(list_of_paths, trans_prob)


# In[57]:


fig, ax = plt.subplots(figsize=(11, 9))
# plot heatmap
sns.heatmap(trans_matrix, cmap="Blues",annot=True, linewidths=.5)
plt.show()


# In[ ]:




