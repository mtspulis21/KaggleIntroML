#exercise 2 - global imports
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

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

# make test X predicitons
predictions = iowa_model.predict(X)
print(predictions)
#compare predictions to y
print(y)
