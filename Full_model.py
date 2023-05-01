import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print('Data in progress')

# reading data from DataSet_Not_Normalized.csv(write here yours way to file)
data = pd.read_csv('C:/Python/Data Science/SK elektrarne/Zadanie/Python Part/DataSet/DataSet_Not_Normalized.csv',
                   parse_dates=['datetime'], index_col='datetime')

# deleting repeating in dataset
data.drop_duplicates(inplace=True)

# deleting of empty parts
data.dropna(inplace=True)

# save the datetime column
datetime_col = data.index

# reset the index to create a column with the datetime information
data = data.reset_index()

# save the datetime column separately
datetime_col = data['datetime']

# delete the datetime column before normalizing the data
data.drop('datetime', axis=1, inplace=True)

# Splitting data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# normalization trough MInMaxScaler
scaler = MinMaxScaler()
normalized_train_data = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)
normalized_test_data = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns)

# add the datetime column back to the normalized dataframes
normalized_train_data.insert(0, 'datetime', datetime_col[train_data.index])
normalized_test_data.insert(0, 'datetime', datetime_col[test_data.index])

# Saving data to new files
normalized_train_data.to_csv('C:/Python/Data Science/SK elektrarne/Zadanie/Python Part/DataSet/DataSet_Normalized_train.csv',
                             index=False)
normalized_test_data.to_csv('C:/Python/Data Science/SK elektrarne/Zadanie/Python Part/DataSet/DataSet_Normalized_test.csv',
                            index=False)

print(data.isnull().sum())
print('Finished. Data in DataSet_Normalized_train.csv and DataSet_Normalized_test.csv')

# Loading data from previously created files
train_data = pd.read_csv('C:/Python/Data Science/SK elektrarne/Zadanie/Python Part/DataSet/DataSet_Normalized_train.csv'
                         , parse_dates=['datetime'], index_col='datetime')
test_data = pd.read_csv('C:/Python/Data Science/SK elektrarne/Zadanie/Python Part/DataSet/DataSet_Normalized_test.csv'
                        , parse_dates=['datetime'], index_col='datetime')

# choosing columns for learning
features = ['tepl_m', 'zraz_m', 'tlak_m', 'v_rh_m', 'v_s_m', 'v_rh_m_50', 'v_rh_m_125', 'vlhk_m']

# choosing target parameter(column)
target = 're_m1_svor'

# Splitting data for two sets
X_train, y_train = train_data[features], train_data[target]
X_test, y_test = test_data[features], test_data[target]

# model declaration
model = RandomForestRegressor(n_estimators=100, random_state=42)

# model training
model.fit(X_train, y_train)

# Model testing(on previously created test set)
y_pred = model.predict(X_test)

# Score
print('R2 score:', r2_score(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))

# Just test with randomly picked normalized data

new_data = pd.DataFrame({'tepl_m': [0.213934178], 'zraz_m': [0], 'tlak_m': [0.942253103],
                         'v_rh_m': [0.158762575], 'v_s_m': [0.103738218], 'v_rh_m_50': [0.164379764],
                         'v_rh_m_125': [0.231473313], 'vlhk_m': [0.580543468]})
y_new_pred = model.predict(new_data[features])

# Showing result with new random data

print('Predicted re_m1_svor with randdom normalized new test data:', y_new_pred)




