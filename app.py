import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import requests
import mplcursors

#Veri setini indirmek için aşağıdaki yorum satırlarının çalıştırılması gerekiyor. 

# def download(url, filename):
#     response = requests.get(url)
#     if response.status_code == 200:
#         with open(filename, "wb") as f:
#             f.write(response.content)

# file_name = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
# download(file_name, "kc_house_data_NaN.csv")

file_name = "kc_house_data_NaN.csv"
df = pd.read_csv(file_name)

print(df.describe())
features = ['bedrooms', 'bathrooms', 'sqft_living', 'condition', 'grade', 'yr_built']  
target = 'price'

mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)
mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)
mean=df['sqft_living'].mean()
df['sqft_living'].replace(np.nan,mean, inplace=True)
mean=df['condition'].mean()
df['condition'].replace(np.nan,mean, inplace=True)
mean=df['grade'].mean()
df['grade'].replace(np.nan,mean, inplace=True)
mean=df['yr_built'].mean()
df['yr_built'].replace(np.nan,mean, inplace=True)

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

plt.scatter(y_test, predictions)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Konutların Gerçek ve Tahmim Edilen Fiyatlar')

sns.regplot(x=y_test, y=predictions, scatter=False, color='red')

mplcursors.cursor(hover=True).connect(
    "add",
    lambda sel: sel.annotation.set_text(f"Actual: {sel.target[0]:.2f}, Predicted: {sel.target[1]:.2f}")
)

plt.show()
