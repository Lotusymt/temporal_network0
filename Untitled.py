#!/usr/bin/env python
# coding: utf-8

# In[56]:


import numpy as np
from scipy import io
import h5py
import math
import matplotlib.pyplot as plt


# In[57]:


#draw entropy_rate_graph for 1102_down_DC_BJ

matrix_1min = np.load('agg_final/agg_final/1min/20211102/down/DC_BJ_agg.npy')
matrix_5min = np.load('agg_final/agg_final/5min/20211102/down/DC_BJ_agg.npy')
matrix_10min = np.load('agg_final/agg_final/10min/20211102/down/DC_BJ_agg.npy')
matrix_30min = np.load('agg_final/agg_final/30min/20211102/down/DC_BJ_agg.npy')
matrix_60min = np.load('agg_final/agg_final/60min/20211102/down/DC_BJ_agg.npy')

#put the arrays into a list to reference them in iteration
list=[  matrix_1min ,
        matrix_5min ,
        matrix_10min,
        matrix_30min,
        matrix_60min]


# In[58]:


print(list)


# In[59]:


#d = 0.5
#c1 = np.where(matrix3 >= (matrix3[1] - d))
#c2 = np.where(matrix3 <= (matrix3[1] + d))
#c3 = np.intersect1d(c1,c2)
#Number = matrix3.shape[0]


# In[60]:


#print(c1)


# In[61]:


#print(c2)


# In[62]:


#print(np.intersect1d(c1,c2))


# In[63]:


#Whether p is a subarray of q(q>2*p) within an error of d
def my_key(q, p, d):
    numQ = q.shape[0]
    numP = p.shape[0]
    if numQ < 2 * numP:
        return False
    else:
        for i in range(numQ):
            flag = 0
            for j in range(numP):
                if (i + j >= numQ) or ((q[i + j] < p[j] - d) or (q[i + j] > p[j] + d)):
                    flag = 1
                    break;
            if (j == numP - 1) and (flag == 0):
                return True
        return False  


# In[64]:


#q = np.array([2,1,2,3,3,4,3,5,3])
#p = np.array([1,2,3])


# In[65]:


#my_key(q,p,0)


# In[66]:


def LambdaHelper(q, start, end, d):
    ## Must Under Condition STEP-BY-STEP
    ## Must Set end at 0. Must Set start at 0.
    if my_key(q[(start + 1):], q[start : end + 1], d):
        #print(q[(start + 1):])
        #print(q[start : end + 1])
        return LambdaHelper(q, start, end + 1, d)
    else:
        #print(start)
        #print(end)
        return (end - start + 1)


# In[67]:


#q = np.array([3,4,3,3,3,3,3])
#start = 0
#end = 0
#print(q[start+1:])
#print(q[start : (end + 1)])
#print(d)
#print(my_key(q[start+1:], q[start : (end + 1)], 0))
#print(LambdaHelper(q,0,0,0.05))


# In[68]:


def Lambda(q,i,d):
    ## Calculate the i-th element's lambda in time series q.
    ## q--given time series
    ## i--target element in the sequence
    ## d--devariance
    numLength = q.shape[0]
    Lambda = 0
    if (i == numLength - 1):
        Lambda = 1
        return Lambda
    else:
        return LambdaHelper(q,0,0,d)


# In[69]:


#q = np.array([3,4,3,3,4,3,3])
#print(Lambda(q,0,0.05))


# In[70]:


d = 0.05
estEntropy_list=[]
x_axis=[1,5,10,30,60]
for i in range(len(list)):
    sumLambda = 0
    data = list[i]
    Number = data.shape[0]

    #c1 = np.where(data >= (data[1] - d))
    #c2 = np.where(data <= (data[1] + d))
    #c3 = np.intersect1d(c1,c2)

    for i in range(Number):
        sumLambda += Lambda(data, i, d)
    #print(sumLambda)
    estEntropy_list.append(math.log(Number)/(sumLambda/Number))
    #print(estEntropy_list)

# Create a figure and a subplot
fig, ax = plt.subplots()

# Plot the data
ax.plot(x_axis,estEntropy_list)

# draw entropy_rate_graph for 1102_down_DC_BJ
ax.set(title='1102_down_DC_BJ', xlabel='x', ylabel='y')

# Show the plot
plt.show()


# In[ ]:




