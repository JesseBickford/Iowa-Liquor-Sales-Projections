% matplotlib inline
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics


df = pd.read_csv('Iowa_Liquor_Sales_reduced.csv', infer_datetime_format=True, low_memory=False)


#Clean up the columns first by removing the $ from Cost, Retail, and Dollars
convertstrings = ['State Bottle Cost', 'State Bottle Retail', 'Sale (Dollars)']
for i in convertstrings:
    df[i] = df[i].str.replace('$','')

#change the date to be in datetime format
df['Date']=pd.to_datetime(df['Date'], infer_datetime_format=True)

#change data types of 'State Bottle Cost', 'State Bottle Retail', 'Sale (Dollars)' to floats
convertfloats = ['State Bottle Cost', 'State Bottle Retail', 'Sale (Dollars)']
for i in convertfloats:
    df[i] = df[i].astype(float)

df = df.drop('Volume Sold (Gallons)', axis = 1)

#create margin and priceper liter columns
df['Margin'] = df['State Bottle Retail']-df['State Bottle Cost']
df['Price per Liter'] = df['State Bottle Retail']/df['Bottle Volume (ml)']
df['Price per Liter'] = df['Price per Liter']*1000

#find and add the isnull values to see what is missing
df.isnull().sum()

#create data frame of index that stores first sale and last sale
stores = df['Store Number'].unique()
store_totals = pd.DataFrame(columns = ['First_Sale', 'Last_Sale', 'sales2015','sales2015Q1', 'sales2016Q1'], index = stores)
stores_dates = pd.DataFrame(columns =['First_Sale', 'Last_Sale'], index = stores)
for store in stores:
    dfstore = df[df['Store Number']==store].Date
    storefirst = min(dfstore)
    storelast = max(dfstore)
    store_totals['First_Sale'][store] = storefirst
    store_totals['Last_Sale'][store] = storelast
feb = pd.Timestamp("20150201")
nov = pd.Timestamp("20151130")
store_totals = store_totals[store_totals.First_Sale <= feb]
store_totals = store_totals[store_totals.Last_Sale >= nov]


len(store_totals)

df.head()
##### 2015 sales
df.sort_values(by=["Store Number", "Date"], inplace=True)
start_date = pd.Timestamp("20150101")
end_date = pd.Timestamp("20151231")
mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
sales2015 = df[mask]


#sort new dataframe sales2015 by storenumber as index
sales2015 = sales2015.groupby(by=["Store Number"], as_index=False)
sales2015 = sales2015.agg({"Sale (Dollars)": [np.sum, np.mean],
                   "Volume Sold (Liters)": [np.sum, np.mean],
                   "Margin": np.mean,
                   "Price per Liter": np.mean,
                   "Zip Code": lambda x: x.iloc[0], # just extract once, should be the same
                   "City": lambda x: x.iloc[0],
                   "County Number": lambda x: x.iloc[0],
                   "County": lambda x: x.iloc[0]})


sales2015.columns = [' '.join(col).strip() for col in sales2015.columns.values]
sales2015

sales2015.columns = ['Store#','County', 'City', 'CountyNum','Total_Sales_2015','Sales_Mean_2015','Total_Liters_2015','Avg_Liters_2015','Zip_Code','Avg_PPL_2015', 'Avg_Margin_2015']
sales2015.head()

df.head()

#####Q1 2015 Sales
df.sort_values(by=["Store Number", "Date"], inplace=True)
start_date = pd.Timestamp("20150101")
end_date = pd.Timestamp("20150331")
mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
sales2015Q1 = df[mask]

#sort new dataframe sales2015 by storenumber as index
sales2015Q1 = sales2015Q1.groupby(by=["Store Number"], as_index=False)
sales2015Q1 = sales2015Q1.agg({"Sale (Dollars)": [np.sum, np.mean],
                   "Volume Sold (Liters)": [np.sum, np.mean],
                   "Margin": np.mean,
                   "Price per Liter": np.mean,
                   "Zip Code": lambda x: x.iloc[0], # just extract once, should be the same
                   "City": lambda x: x.iloc[0],
                   "County Number": lambda x: x.iloc[0],
                   "County": lambda x: x.iloc[0]})

sales2015Q1.columns = [' '.join(col).strip() for col in sales2015Q1.columns.values]
sales2015Q1.columns = ['Store#','County', 'City', 'CountyNum','Total_Sales_2015_Q1','Sales_Mean_2015_Q1','Total_Liters_2015_Q1','Avg_Liters_2015_Q1','Zip_Code','Avg_PPL_2015_Q1', 'Avg_Margin_2015_Q1']
sales2015Q1.head()



#join sales2015 and sales2015Q1 into the same dataframe so it can be used for modeling
combined2015 = pd.merge(sales2015Q1, sales2015, on=['Store#','City', 'CountyNum'])

#change the store and county numbers to strings for easier graphing in tableau
combined2015['Store#'] = combined2015['Store#'].astype(str)
combined2015['County#'] = combined2015['CountyNum'].astype(str)



#####Q1 2016 Sales
df.sort_values(by=["Store Number", "Date"], inplace=True)
start_date = pd.Timestamp("20160101")
end_date = pd.Timestamp("20160331")
mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
sales2016Q1 = df[mask]

#sort new dataframe sales2016 by storenumber as index
sales2016Q1 = sales2016Q1.groupby(by=["Store Number"], as_index=False)
sales2016Q1 = sales2016Q1.agg({"Sale (Dollars)": [np.sum, np.mean],
                   "Volume Sold (Liters)": [np.sum, np.mean],
                   "Margin": np.mean,
                   "Price per Liter": np.mean,
                   "Zip Code": lambda x: x.iloc[0], # just extract once, should be the same
                   "City": lambda x: x.iloc[0],
                   "County Number": lambda x: x.iloc[0],
                   "County": lambda x: x.iloc[0]})

sales2016Q1.columns = [' '.join(col).strip() for col in sales2016Q1.columns.values]
sales2016Q1.columns = ['Store#', 'County','City', 'County#','Total_Sales_2016_Q1', 'Sales_Mean_2016_Q1','Total_Liters_2016_Q1','Avg_PPL_2016_Q1','Zip_Code','Avg_Liters_2016_Q1', 'Avg_Margin_2016_Q1']



combined2015.head()
#straight linear regression with NO cross validation
X = combined2015[['Total_Sales_2015_Q1', 'Avg_PPL_2015_Q1', 'Avg_Margin_2015_Q1']]
y = combined2015['Total_Sales_2015']

X2016 = sales2016Q1[['Total_Sales_2016_Q1', 'Avg_PPL_2016_Q1', 'Avg_Margin_2016_Q1']]

lm = linear_model.LinearRegression()
model = lm.fit(X,y)
predictions = lm.predict(X)
predictions2016 = lm.predict(X2016)

#here i misnamed the column, should be without the "Q1"
sales2016Q1['Predicted Q1 2016 TotalSales'] = lm.predict(X2016)
plt.scatter(predictions, y)
plt.show()
print lm.score(X,y)

#this is predicted sales of all 2016 based on Q1 2015 model
np.sum(predictions2016)

#this is predicted sales of 2015 based on Q1 2015 model
np.sum(predictions)

#this is actual sales of all 2015
np.sum(sales2015['Total_Sales_2015'])

print "2015: ", lm.score(X,y)

#again, predicted sales of all 2016 based on Q1 2015 model
sales2016Q1['Predicted Q1 2016 TotalSales'].sum()
s16q1

#here I rename the column so it correctly represents its data
sales2016Q1.columns.values[11] = "Predicted 2016 TotalSales"

sales2016Q1.dtypes

#linear regression WITH cross validation
model_frame = combined2015[['Total_Sales_2015_Q1', 'Avg_PPL_2015_Q1', 'Avg_Margin_2015_Q1']]
y = combined2015['Total_Sales_2015']

model_frame_2016 = sales2016Q1[['Total_Sales_2016_Q1', 'Avg_PPL_2016_Q1', 'Avg_Margin_2016_Q1']]

X_train, X_test, y_train,y_test = train_test_split(model_frame, y, test_size =0.2)
print X_train.shape, y_train.shape
print X_test.shape, y_test.shape

lm2 = linear_model.LinearRegression()
model2 = lm2.fit(X_train, y_train)
predictions2 = lm2.predict(X_test)
scores = cross_val_score(model2, model_frame, y, cv= 12)
print "Cross Validated scores: ", scores

predictions3 = cross_val_predict(model2, model_frame, y, cv=12)
accuracy = metrics.r2_score(y, predictions3)
print "Cross Predicted Accuracy: ", accuracy

np.sum(predictions)
np.sum(predictions3)
np.sum(predictions2016)
#######################################################################################


sales2015.head()
sales2015Q1.head()
sales2016Q1.head()

s15q1 = sales2015Q1['Total_Sales_2015_Q1'].sum()
s16q1 = sales2016Q1['Total_Sales_2016_Q1'].sum()

s16q1-s15q1


print 'Sales in the first quarter of 2016 increased by:',s16q1-s15q1, 'dollars'

sns.regplot(combined2015['Avg_PPL_2015'], combined2015['Total_Sales_2015'])

np.max(sales2015Q1['Total_Sales_2015_Q1'])
np.max(sales2016Q1['Total_Sales_2016_Q1'])
np.max(combined2015['Total_Sales_2015'])

combined2015.to_csv('combined2015final.csv')
combined2015.dtypes


sales2016Q1.to_csv('sales2016Q1.csv')

#Problem statement: find the projected total liquor sales for 2016.
#I project that sales for 2016 will only increase by a minimal amount, $803,196.(I think this is incaccurate)
#Attached are links to the top 10 performing stores of Q1 2016 along with their projected sales for the whole year.
print "Actual 2015 Total sales are: %d" % np.sum(sales2015['Total_Sales_2015'])
print "Actual Q1 2015 sales are: %d" %np.sum(sales2015Q1['Total_Sales_2015_Q1'])
print "Actual Q1 2016 sales are: %d" % np.sum(sales2016Q1['Total_Sales_2016_Q1'])
print "Predicted 2016 Total sales for all stores is %d" % np.sum(sales2016Q1["Predicted Q1 2016 TotalSales"])

from PIL import Image
im = Image.open("2016 Predicted Sales.png")
im
#These are the top 10 performing stores for 2016 Q1 and their predicted sales for the full year.

im2 = Image.open("2015 Sales Cluster.png")
im2
#this shows sales based on zip code, notably that as a new owner you would want a store in Polk county. 
