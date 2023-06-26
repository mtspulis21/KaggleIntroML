#exercise 5 - global imports
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# determinar file path
iowa_file_path = 'S:/KAGGLE/Intro to Machine Learning/train.csv'
# variaveis necessarias
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
feature_names = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF',
                 'FullBath','BedroomAbvGr','TotRmsAbvGrd']
X = home_data[feature_names]
iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(X, y)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# check ex4 para ver o tree size test
best_tree_size = 100

# Modelo final após decidir o tamanho das leafs
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)

final_model.fit(X,y)

from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))


# step 1: use a random forest
# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X,train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
