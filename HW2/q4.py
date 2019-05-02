#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# In[ ]:


user_show_matrix = []
with open("user-shows.txt") as f:
  lines = f.readlines()
  for line in lines:
    ratings = line.split(" ")
    user_row = []
    for rating in ratings:
      user_row.append(int(rating))
    user_show_matrix.append(user_row)


# In[ ]:


R = np.array(user_show_matrix)


# In[ ]:


P = np.diag(np.apply_along_axis(sum,1,R))
Q = np.diag(np.apply_along_axis(sum,0,R))


# In[ ]:


T_user = R.dot(R.T)
T_item = R.T.dot(R)


# In[ ]:


Q_half = np.sqrt(Q)
P_half = np.sqrt(P)
Q_minus_half = np.zeros(Q_half.shape)
P_minus_half = np.zeros(P_half.shape)
for i in range(len(Q_half)):
  for j in range(len(Q_half[i])):
    if(Q_half[i][j]!=0):
      Q_minus_half[i][j] = 1/Q_half[i][j]
for i in range(len(P_half)):
  for j in range(len(P_half[i])):
    if(P_half[i][j]!=0):
      P_minus_half[i][j] = 1/P_half[i][j]


# In[ ]:


S_show = Q_minus_half.dot(T_item).dot(Q_minus_half)


# In[ ]:


S_user = P_minus_half.dot(T_user).dot(P_minus_half)


# In[ ]:


Tu_show = R.dot(S_show)
Tu_user = S_user.dot(R)


# In[ ]:


user_similarities_alex = Tu_user[499][:100]
item_similarities_alex = Tu_show[499][:100]


# In[ ]:


user_similarities_dict = {}
item_similarities_dict = {}
for i,sim in enumerate(user_similarities_alex):
  user_similarities_dict[i] = sim
for i,sim in enumerate(item_similarities_alex):
  item_similarities_dict[i] = sim


# In[ ]:


user_similarities_sorted = sorted(user_similarities_dict.items(), key = lambda x:(x[1],-x[0]), reverse=True)
item_similarities_sorted = sorted(item_similarities_dict.items(), key = lambda x:(x[1],-x[0]), reverse=True)


# In[ ]:


user_similarities_indices_scores = user_similarities_sorted[:5]
item_similarities_indices_scores = item_similarities_sorted[:5]


# In[102]:


print(user_similarities_indices_scores)


# In[103]:


print(item_similarities_indices_scores)


# In[ ]:


listOfUserIndices = [i for i,j in user_similarities_indices_scores]
listOfItemIndices = [i for i,j in item_similarities_indices_scores]


# In[ ]:


shows_list = []
with open("shows.txt") as f:
  lines = f.readlines()
  for line in lines:
    shows_list.append(line)


# In[ ]:


userRecommendedShows = list(shows_list[i] for i in listOfUserIndices)
itemRecommendedShows = list(shows_list[i] for i in listOfItemIndices)


# In[109]:


print("User-user collaborative filtering recommendations are as follows:\n",userRecommendedShows)


# In[110]:


print("Item-item collaborative filtering recommendations are as follows:\n",itemRecommendedShows)


# In[ ]:




