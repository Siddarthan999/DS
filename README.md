# DS
```
import pandas as pd
df=pd.read_csv("Data_set.csv")
print(df.head())
print(df.tail())
print(df.info())
print(df.dtypes)
print(df.isnull().sum())

1) Data Cleaning:

Dropping Missing rows:
• df.dropna(inplace='True')

Filling empty values with mean values:
• x = df['height'].mean()
• df['height'].fillna(x, inplace=True)

Formatting the date column to the correct format (20201226 to 2020/12/26) :
• df['Date'] = pd.to_datetime(df['Date'])

Dropping duplicate values:
• df.drop_duplicates(inplace = True)


2) Detect and Remove Outliers:

import matplotlib.pyplot as plt
plt.boxplot(data['price'])
plt.show()

''' Detection '''
# IQR
# Calculate the upper and lower limits
Q1 = data['price'].quantile(0.25)
Q3 = data['price'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
 
# Create arrays of Boolean values indicating the outlier rows
upper_array = np.where(data['price']>=upper)[0]
lower_array = np.where(data['price']<=lower)[0]
 
# Removing the outliers
data.drop(index=upper_array, inplace=True)
data.drop(index=lower_array, inplace=True)

plt.boxplot(data['price'])
plt.show()

3) Feature Selection Techniques:

import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Load the data
data = pd.read_csv("CarPrice.csv")

print(data.dtypes)

#Either remove the Object datatype from the dataset or replace with numerical values
data["fueltype"]=data["fueltype"].map({"gas":1,"diesel":0})
data.drop(['enginetype','carbody','symboling','CarName','aspiration','doornumber','drivewheel','enginelocation','cylindernumber','fuelsystem'], axis=1, inplace=True)

# Select the top 10 features using chi-squared test
selector = SelectKBest(chi2, k=10)
data = selector.fit_transform(data, data["horsepower"]) #Enter the column that is of int64 datatype not float64

# Print the number of features after feature selection
print("Selected Features:",data.shape)


4) Data Visualization:

import pandas as pd

# Load the data
data = pd.read_csv("CarPrice.csv")

import matplotlib.pyplot as plt
import seaborn as sns

plt.plot(data['fueltype'])
plt.show()

#Univariate Analysis
sns.boxplot(data['horsepower'])
plt.show()

sns.countplot(data['horsepower'])
plt.show()

sns.histplot(data['horsepower'])
plt.show()

sns.lineplot(data['horsepower'])
plt.show()

x = data['fueltype'].value_counts()
plt.pie(x.values, labels=x.index, autopct='%1.1f%%')
plt.show()

#Bivariate analysis
sns.barplot(x=data['fueltype'],y=data['horsepower'])
plt.show()

sns.scatterplot(data['horsepower'])
plt.show()

#Multivariate Analysis
sns.heatmap(data['horsepower'].corr(), annot=True)
plt.show()
```
