#!/usr/bin/env python
# coding: utf-8

# # Start

# In[29]:


# !source /fs/project/PAS1066/zhang_anaconda/anaconda3/bin/activate


# In[30]:


import cudf 
import cupy as cp
import cuml
from cuml.svm import SVR
from cuml.model_selection import train_test_split
from cuml.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# In[31]:


df = cudf.read_csv('ricky_data.csv')


# In[32]:


df.describe()


# In[33]:


df.head()


# In[34]:


df2 = df.copy()


# # Creating the data splits

# In[35]:


X = df2.iloc[0:len(df2), [1, 2, 4, 5, 6, 8]]
y = df2.iloc[0:len(df2), 3]


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.67)


# # Standardizing X_train

# In[37]:


stand_scaler = StandardScaler()


# In[38]:


stand_scaler.fit(X_train)


# In[39]:


X_train_norm= stand_scaler.transform(X_train)


# In[40]:


X_test_norm = stand_scaler.transform(X_test)


# # Log transform of y_train
# 

# In[41]:


y_train_log = cp.log(y_train)


# # Creating the model

# In[42]:


svr = SVR(
    kernel='poly',
    degree= 3,
    gamma='auto',
    C=15,
    epsilon=0.001,
    coef0= 1.2
)


# In[43]:


start_time = time.time()


# In[44]:


svr.fit(X_train_norm, y_train_log)


# # Exponentiating the predictions

# In[45]:


y_predict_train = cp.asnumpy(cp.exp(svr.predict(X_train_norm)))
y_predict_test = cp.asnumpy(cp.exp(svr.predict(X_test_norm)))


# In[46]:


end_time = time.time()


# # Transfering cupy arrays and cudf to numpy arrays

# In[47]:


myar = np.zeros(len(y_predict_test))


# In[48]:


for i in range(len(myar)):
    myar[i] = y_predict_test[i]


# In[49]:


y_predict_test = myar


# In[50]:


myar2 = np.zeros(len(y_test))


# In[51]:


y_test = y_test.to_numpy()


# # Summary

# In[52]:


print(f'Size of training set| {len(X_train)}\nSize of testing set | {len(X_test)}')


# In[53]:


print(f'MSE: {mean_squared_error(y_test, y_predict_test)}')


# In[54]:


print(f'r^2 score :{r2_score(y_test, y_predict_test)}')


# In[55]:


total_time = end_time - start_time
print(f'Wall Clock\n\nHours  | {total_time // 60**2}\nMinutes| {total_time // 60}\nSeconds| {np.abs(total_time - 60 * (total_time // 60 ))}')


# # Testing the time needed for varying amounts of testing sizes

# In[ ]:


xSize = np.linspace(start = .25, stop= .95, num= 50)
yTime = np.zeros(len(xSize))
for i in range(len(xSize)):
    print(f'job: {i+1}')
    X_trainTest, X_testTest, y_trainTest, y_testTest = train_test_split(X, y, train_size = xSize[i])
    # 
    stand_scalerTest = StandardScaler()
    #
    stand_scalerTest.fit(X_trainTest)
    #
    X_train_normTest = stand_scalerTest.transform(X_trainTest)
    #
    X_test_normTest = stand_scalerTest.transform(X_testTest)
    #
    y_train_logTest = cp.log(y_trainTest)
    #
    svrTest = SVR(
        kernel='poly',
        degree= 3,
        gamma='auto',
        C=15,
        epsilon=0.001,
        coef0= 1.2
    )
    #
    start_timeTest = time.time()
    #
    svrTest.fit(X_train_normTest, y_train_logTest)
    #
    y_predict_trainTest = cp.asnumpy(cp.exp(svr.predict(X_train_normTest)))
    y_predict_testTest = cp.asnumpy(cp.exp(svr.predict(X_test_normTest)))
    #
    end_timeTest = time.time()
    #
    total_timeTest = end_timeTest - start_timeTest
    #
    yTime[i] = total_timeTest
    #
    print(f'{xSize[i], yTime[i]}')
    print()



# In[ ]:


plt.scatter(xSize, yTime)


# In[ ]:


dfTime = pd.DataFrame({'xSize':xSize, 'yTime':yTime})


# In[ ]:


dfTime.to_csv('time_needed2.csv')


# In[ ]:




