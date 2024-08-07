#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# In[2]:


df=pd.read_csv("Reliance Stock Price.csv")


# In[3]:


df


# In[4]:


df.info()


# In[5]:


df.describe().T


# In[6]:


df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)


# In[7]:


df.shape


# In[8]:


df.size


# In[9]:


df.isnull().sum()


# In[14]:


df.dropna(inplace=True)


# In[15]:


df.isnull().sum()


# In[17]:


# Visualizing the Stock Data using matplotlib
#weighted average price (WAP) 
df.plot(figsize=(18,6),subplots = True,grid = True)
df.WAP.plot()
plt.show()


# In[36]:


plt.figure(figsize=(14, 7))
plt.plot(df['Close Price'], label='Close Price',linewidth=3)
plt.title('Stock Close Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


# In[35]:


plt.figure(figsize=(14, 7))
plt.plot(df['Open Price'], label='Open Price', color='red',linewidth=3)
plt.title('Stock Open Price')
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.legend()
plt.show()


# In[37]:


plt.figure(figsize=(14,7))
plt.plot(df['High Price'],label='High Price',color='pink',linewidth=3)
plt.title('High Price Over Time')
plt.xlabel('Date')
plt.ylabel('High Price')
plt.show()


# In[38]:


plt.figure(figsize=(14,7))
plt.plot(df['Low Price'], label='Low Price',color='green',linewidth=3)
plt.title('Low Price over Time')
plt.xlabel('Date')
plt.ylabel('Low Price')
plt.show()


# In[39]:


plt.figure(figsize=(14,7))
plt.plot(df['No.of Shares'], label='No.of Shares',color='yellow',linewidth=3)
plt.title('No.of Sharesover Time')
plt.xlabel('Date')
plt.ylabel('No.of Shares')
plt.show()


# In[40]:


plt.figure(figsize=(14,7))
plt.plot(df['No. of Trades'], label='No. of Trades',color='purple',linewidth=3)
plt.title('No. of Trades over Time')
plt.xlabel('Date')
plt.ylabel('No. of Trades')
plt.show()


# In[25]:


plt.figure(figsize=(14,7))
plt.plot(df['Total Turnover (Rs.)'], label='Total Turnover (Rs.)')
plt.title('Total Turnover (Rs.)')
plt.xlabel('Date')
plt.ylabel('Total Turnover (Rs.)')
plt.show()


# In[18]:


df.info()


# In[26]:


#### Candlestick Chart

fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

candlestick = go.Candlestick(x=df.index,
                             open=df['Open Price'],
                             high=df['High Price'],
                             low=df['Low Price'],
                             close=df['Close Price'],
                             name='Candlestick')
fig.add_trace(candlestick)

fig.update_layout(title='Candlestick Chart',
                  xaxis_title='Date',
                  yaxis_title='Price')
fig.show()


# In[27]:


#### Combined Price and Volume Chart
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.1, subplot_titles=('Combined Price', 'Deliverable Quantity'),
                    row_width=[0.2, 0.7])

fig.add_trace(go.Candlestick(x=df.index,
                             open=df['Open Price'],
                             high=df['High Price'],
                             low=df['Low Price'],
                             close=df['Close Price'],
                             name='Candlestick'), row=1, col=1)

fig.add_trace(go.Bar(x=df.index, y=df['Deliverable Quantity'], name='Deliverable Quantity'), row=2, col=1)

fig.update_layout(title='Stock Price and Deliverable Quantity',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  yaxis2_title='Deliverable Quantity')

fig.show()


# In[28]:


### Step 5: Analyzing Trading Volumes

#### Volume Chart

plt.figure(figsize=(14, 7))
plt.bar(df.index, df['Deliverable Quantity'], label='Deliverable Quantity')
plt.title('Trading Deliverable Quantity Over Time')
plt.xlabel('Date')
plt.ylabel('Deliverable Quantity')
plt.legend()
plt.show()


# In[29]:


### Step 6: Advanced Analysis with Seaborn

#### Distribution of Daily Returns
df['Daily Return'] = df['Close Price'].pct_change()

plt.figure(figsize=(10, 6))
sns.distplot(df['Daily Return'].dropna(), bins=100, kde=True)
plt.title('Distribution of Daily Returns')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.show()


# In[30]:


###  Adding Moving Averages
df['MA20'] = df['Close Price'].rolling(window=20).mean()
df['MA50'] = df['Close Price'].rolling(window=50).mean()

plt.figure(figsize=(14, 7))
plt.plot(df['Close Price'], label='Close Price')
plt.plot(df['MA20'], label='20-Day Moving Average')
plt.plot(df['MA50'], label='50-Day Moving Average')
plt.title('Stock Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[31]:


#### Correlation Matrix

# Assuming you have a DataFrame with returns of multiple stocks
returns_df = df[['Daily Return','MA20', 'MA50']].pct_change()
corr = returns_df.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Daily Returns')
plt.show()


# In[32]:


#### Correlation Matrix
corr = df.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Reliance Stocks')
plt.show()


# In[33]:


sns.pairplot(df)


# In[ ]:




