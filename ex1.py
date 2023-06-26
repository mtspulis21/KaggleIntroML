#exercise 1 - global imports
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# determinar file path
iowa_file_path = 'S:/KAGGLE/Intro to Machine Learning/train.csv'

# criar variavel pra ler file path
home_data = pd.read_csv(iowa_file_path)

home_data.describe()