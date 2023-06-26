#exercise 3 - global imports
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# determinar file path
iowa_file_path = 'S:/KAGGLE/Intro to Machine Learning/train.csv'

# criar variavel pra ler file path
home_data = pd.read_csv(iowa_file_path)

# printar colunas para achar predicition target
print(home_data.columns)
# usar valor da venda -> o que será predicited
y = home_data.SalePrice

# criar DataFrame X to hold the predictive features
feature_names = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF',
                 'FullBath','BedroomAbvGr','TotRmsAbvGrd']

# Criar Variavel X para segurar as features
X = home_data[feature_names]

# review X.head
print(X.head())

# specify and fit model -> DecisionTreeRegressor
iowa_model = DecisionTreeRegressor(random_state=1)

# insert model X, y
iowa_model.fit(X, y)

print("First in-sample predictions:", iowa_model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())

# split data into training and validation data for both Features and Data
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Real model
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit/Train model -> before was -> iowa_model.fit(X, y)
iowa_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predicted_prices = iowa_model.predict(val_X)
print(mean_absolute_error(val_y, val_predicted_prices))