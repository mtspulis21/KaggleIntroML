# Underfitting and Overfitting
# Fine-tune your model for better performance.

#exercise 4 - global imports
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
# get predicted prices on validation data
val_predicted_prices = iowa_model.predict(val_X)
print(mean_absolute_error(val_y, val_predicted_prices))

# We can use a utility function to help compare MAE scores
# from different values for max_leaf_nodes:

def get_mae(max_leaf_nodes,train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X,train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y,preds_val)
    return(mae)

# We can use a for-loop to compare the accuracy of 
# models built with different values for max_leaf_nodes.
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
for max_leaf_nodes in candidate_max_leaf_nodes:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print('Max leaf nodes: %d \t\t Mean absolute error: %d' %(max_leaf_nodes, my_mae))
    
best_tree_size = 100

# Modelo final após decidir o tamanho das leafs
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)

final_model.fit(X,y)