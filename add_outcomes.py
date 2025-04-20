#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[ ]:


patients_data = pd.read_csv('Close2AdmitData.csv')
computational_patient_data = pd.read_csv('Close2AdmitDTinfo.csv')
rv_outcomes = pd.read_excel('RVoutcomes.xlsx')


# In[8]:


patients_data.rename(columns={'patKey': 'patkey'}, inplace=True)
computational_patient_data.rename(columns={'patKey': 'patkey'}, inplace=True)
print("Columns in patients_data:\n", patients_data.columns)
print("Columns in rv_outcomes:\n", rv_outcomes.columns)
print("Columns in computational_data:\n", computational_patient_data.columns)


# In[14]:


merged_data = pd.merge(patients_data, rv_outcomes, on='patkey', how='left')
merged_data_comp = pd.merge(computational_patient_data, rv_outcomes, on='patkey', how='left')


# print("Updated columns in patients_data:", patients_data.columns)
# print("Updated columns in comp_data:", computational_patient_data.columns)

# In[22]:


merged_csv_file = 'patients_data_with_RVoutcomes.csv'
merged_data.to_csv(merged_csv_file, index=False)
merged_comp_csv_file = 'computational_with_RV.csv'
merged_data_comp.to_csv(merged_comp_csv_file, index=False)


# In[23]:


merged_data = pd.merge(patients_data, rv_outcomes, on='patkey', how='left')
merged_data_comp = pd.merge(computational_patient_data, rv_outcomes, on='patkey', how='left')


# In[ ]:


merged_data.to_csv('Close2AdmitDataWithRV.csv', index=False)
print("Merged data saved to 'merged_patients_data.csv'")


# In[20]:


print(merged_data.head())

