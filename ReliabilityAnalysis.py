#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import random
import math


# In[ ]:





# In[ ]:





# In[3]:


delta_t=1
lambda11=0.005
lambda12=0.017
lambda21=0.0049
lambda22=0.0169
lambda31=0.00503
lambda32=0.0161


# In[4]:


pu11= 1 - math.exp(-lambda11 * delta_t)
pu12= 1 - math.exp(-lambda12 * delta_t)
pu21= 1 - math.exp(-lambda21 * delta_t)
pu22= 1 - math.exp(-lambda22 * delta_t)
pu31= 1 - math.exp(-lambda31 * delta_t)
pu32= 1 - math.exp(-lambda32 * delta_t)


# In[ ]:





# In[5]:


# Stworzenie list wieku i listy wypełnionej zerami
no= list(range(0,250))
empty=['NaN']*250


# In[6]:


# Stworzenie pustego DataFramu z odpowiednimi kolumnami
d = {'age': [],  'stan': [] , 'stan1':[],'stan11': [], 'Pu_11': [],'random11':[], 'stan12':[], 
    'Pu_12': [],'random12':[], 
    'stan2':[],'stan21':[],'Pu_21': [],'random21':[],'stan22':[],'Pu_22': [],'random12':[],
    'stan3':[],'stan31':[],'Pu_31': [],'random31':[],'stan32':[],'Pu_32': [],'random12':[]}
df = pd.DataFrame(data=d)
df


# In[7]:


# Iteracja i wyznaczenie stanu po elementach tak aby wyznaczyć stan każdego elementu 
n=1000
dic={}
stan_11 =[]
stan_12 =[]
stan_21 =[]
stan_22 =[]
stan_31=[]
stan_32 =[]

r_11 = [random.random()for x in range(0, 250)]
r_12 = [random.random()for x in range(0, 250)]
r_21 = [random.random()for x in range(0, 250)]
r_22 = [random.random()for x in range(0, 250)]
r_31 = [random.random()for x in range(0, 250)]
r_32 = [random.random()for x in range(0, 250)]

randomowe = [r_11,r_12,r_21,r_22,r_31,r_32]
pu = [pu11,pu12,pu21,pu22,pu31,pu32]
listt = [stan_11, stan_12, stan_21, stan_22, stan_31, stan_32]

lt = ['stan11', 'stan12', 'stan21', 'stan22', 'stan31', 'stan32']
for i in range(0,n):
    t = {'age': no,  'stan': empty , 'stan1':empty,'stan11': empty, 'Pu_11': pu11,'random11':r_11, 'stan12':empty, 
        'Pu_12': pu12,'random12':r_12, 
        'stan2':empty,'stan21':empty,'Pu_21': pu21,'random21':r_21,'stan22':empty,'Pu_22': pu22,'random22':r_22,
        'stan3':empty,'stan31':empty,'Pu_31': pu31,'random31':r_31,'stan32':empty,'Pu_32': pu32,'random32':r_32}
    temp = pd.DataFrame(data=t)
    stan=[]
    
    
    stan_11 =[]
    stan_12 =[]
    stan_21 =[]
    stan_22 =[]
    stan_31=[]
    stan_32 =[]

    r_11 = [random.random()for x in range(0, 250)]
    r_12 = [random.random()for x in range(0, 250)]
    r_21 = [random.random()for x in range(0, 250)]
    r_22 = [random.random()for x in range(0, 250)]
    r_31 = [random.random()for x in range(0, 250)]
    r_32 = [random.random()for x in range(0, 250)]

    randomowe = [r_11,r_12,r_21,r_22,r_31,r_32]
    pu = [pu11,pu12,pu21,pu22,pu31,pu32]
    listt = [stan_11, stan_12, stan_21, stan_22, stan_31, stan_32]


    
    for i in range(0,6):
        for r in randomowe[i]:
            if r > pu[i]:
                 listt[i].append(1)
            else:
                listt[i].append(0)
                break
        new_df = pd.DataFrame({lt[i]: listt[i]})
        temp.update(new_df)
    #Warunek ustawiający stay i połącznie z poprzenimi df


    temp['stan1'] = np.where((temp['stan11'] == 'NaN')|(temp['stan12'] == 'NaN'), 0, 1)
    temp['stan2'] = np.where((temp['stan21'] == 'NaN')|(temp['stan22'] == 'NaN'), 0, 1)
    temp['stan3'] = np.where((temp['stan31'] == 'NaN')|(temp['stan32'] == 'NaN'), 0, 1)
    
    temp['Count'] = temp['stan1'] +temp['stan2']+temp['stan3']
    
    l = len(temp)
    
    for i in range(0,l):
        if temp.iloc[i]['Count'] >= 2.0:
            stan.append(1)
        else:
            stan.append(0)
            break
        
        
        

    new_df2 = pd.DataFrame({'stan': stan})
    temp.update(new_df2)  
    
#     if
#     df['stan'] = np.where((df['Count'] < 2 ), 0, 1)
    
    df = pd.concat([df,temp])    


# In[9]:


df=df.reset_index()


# In[ ]:





# In[89]:


df.to_csv(r'/home/magda/Dokumenty/projects/Analiza_niezawodności.csv')


# # zliczanie wartości z kolejnych iteracji
# 

# In[8]:


df
df.columns


# In[10]:


import plotly.express as px

fig = px.bar(df, x='age', y='stan')
fig.show()


# # Liczba uszkodzeń układu w czasie t

# In[11]:


# DataFrame z uszkodzeniami
#stan1
uszkodzenia = df.loc[df.stan == 0]
uszkodzenia_stan = uszkodzenia[['age', 'stan']]
uszkodzenia_stan


# In[13]:


uszkodzenia_stan.sort_index(by ='age')


# In[14]:


#Zamiana wartości aby zliczyć zapsucia
uszkodzenia_stan =uszkodzenia_stan.replace({0:1})


# In[17]:


uszkodzenia_stan = uszkodzenia_stan.rename(columns={'stan': 'liczba uszkodzeń'})


# In[18]:


fig = px.bar(uszkodzenia_stan, x='age', y='liczba uszkodzeń', title='Liczba uszkodzeń układu w czasie t')
fig.show()


# # Empiryczna funkcji niezawodnośc

# In[ ]:





# In[58]:



f_niezawodności =uszkodzenia_stan.groupby(['age']).count()
f_niezawodności = f_niezawodności.reset_index()
f_niezawodności = f_niezawodności.sort_values(by='age')
f_niezawodności


# In[59]:


l = list(range(0,150))
zero= [0]*150
data = {'age':l, 'stan':zero } 
  
fill = pd.DataFrame(data) 

# wypełnienie zerowymi warto sciami pustych kroków
f_niezawodności = pd.merge(f_niezawodności, fill, left_on='age', right_on='age', how='right')
f_niezawodności = f_niezawodności.replace(r'^\s*$', np.nan, regex = True) 
f_niezawodności = f_niezawodności.replace(r'nan', np.nan, regex = True)
f_niezawodności = f_niezawodności.replace({np.nan: 0})
f_niezawodności = f_niezawodności.drop(columns = 'stan')
# f_niezawodności = f_niezawodności.rename(columns={'stan_x':'stan'})
f_niezawodności


# In[61]:


f_niezawodności=f_niezawodności.sort_values(by='age')
f_niezawodności


# In[62]:


f_niezawodności['running_sum'] = f_niezawodności['liczba uszkodzeń'].cumsum()


# In[63]:


f_niezawodności['emp_f_niezawodnosci'] = (n - f_niezawodności['running_sum']) / n
f_niezawodności


# In[64]:


f_niezawodności = f_niezawodności.rename(columns={'emp_f_niezawodnosci':'R(t)'})


# In[ ]:





# In[84]:


fig = px.scatter(f_niezawodności, x="age", y="R(t)", title='Empiryczna funkcjia niezawodności')
fig.show()


# In[68]:


#Plotowanie empirycznej funkcji niezawodności  
fig = px.bar(f_niezawodności, x='age', y='R(t)', title='Empiryczna funkcja niezawodności ')
fig.show()


# In[86]:


fig = px.line(f_niezawodności, x="age", y="R(t)", title='Empiryczna funkcja niezawodności ')
fig.show()


# In[70]:


import numpy.polynomial.polynomial as poly

x=f_niezawodności['age'].tolist()
y=f_niezawodności['R(t)'].tolist()

coefs = poly.polyfit(x, y, 4)
z = np.polyfit(x, y, 4)
f = np.poly1d(z)


# calculate new x's and y's
x_new = np.linspace(x[0], x[-1], 150)
y_new = f(x_new)


ffit = np.polyval(coefs[::-1], x_new)
plt.plot(x_new, ffit)


ffit = poly.Polynomial(coefs)    # instead of np.poly1d


# In[71]:


fig = px.line( x=x_new, y=ffit(x_new), title='Empiryczna funkcja niezawodności ')
fig.show()


# In[88]:


f_niezawodności.to_csv(r'/home/magda/Dokumenty/projects/F_niezawodności.csv')


# In[72]:


f_niezawodności['F_gest'] =f_niezawodności['running_sum'] / (n * f_niezawodności['age']) 


# In[73]:


f_niezawodności


# In[75]:


# f_niezawodności = f_niezawodności.replace(np.inf,0)
f_niezawodności = f_niezawodności.replace(np.nan,0)


# In[76]:


fig = px.scatter(f_niezawodności, x="age", y="F_gest", title='Funkcja gęstości prawdopodobieństaw czasu pracy')
fig.show()


# In[83]:


fig = px.line(f_niezawodności, x="age", y="F_gest", title='Funkcja gęstości prawdopodobieństaw czasu pracy')
fig.show()


# In[78]:


f_niezawodności['F_zawodności'] = 1-f_niezawodności['R(t)']
f_niezawodności


# In[79]:


fig = px.scatter(f_niezawodności, x="age", y="F_zawodności", title='Funkcja zawodności', )
fig.show()


# In[87]:


fig = px.line(f_niezawodności, x="age", y="F_zawodności", title='Funkcja zawodności')
fig.show()


# In[ ]:




