# SK_Elektrarne
ML_Prediction_randomforest_model
This is a Python script that performs data normalization and training a machine learning model for regression on the provided dataset.
Specifically in this case, this model calculates the electrical power on the basis of an example dataset already known in advance. 

**Required libraries**: pandas(for data manipulation), scikit-learn(For Machine Learning)

**Data Transformation**: MinMaxScaler (for normalization). This is useful when the value of the attributes has a different scale, which I think is our case. 

**Data Corrections**: I using allowed functions for data corrections - 
data.drop_duplicates(inplace=True) 
# deleting repeating in dataset
data.dropna(inplace=True)
# deleting of empty parts
It's pretty simple and basic. And I used data.isnull() to check that process been finished without any troubles.

**Features choosing**: As a target I'm chose re_m1_svor because this is the parameter which are we wanna to predict. And other parameters I used as a learning data. 

# choosing columns for learning
features = ['tepl_m', 'zraz_m', 'tlak_m', 'v_rh_m', 'v_s_m', 'v_rh_m_50', 'v_rh_m_125', 'vlhk_m']

# choosing target parameter(column)
target = 're_m1_svor'

**Model testing**: I think scikit metrics is good tool for testing because they sipmle and clear. MSE, MAE and R2 Score is the basic and clear metrics according to google. And by the end of the script I do one easy and simple manual test. 

Script loads the data from the provided CSV file, removes any duplicate or empty data, and separates the datetime column.

After that, the data is normalized using MinMaxScaler and split into training and testing sets. The datetime column is added back to the normalized data(I tried to avoid error with missed hh.dd.yyyy in final data(this happens on the one of mine PC, I don't know why))
(**P.S** I created 2 models because I can't denormalize data back(I have some kind of error with 2D/3D data parameters, I can't solve this problem by my self))
(**P.S.S:** First model(Full) normalized all parameters, second model(Full_model_re_m1_svor_DENORMALIZED) leave target parameter denormalized. According the metrics the first one is much better, but becaouse of problem with denormalazing I create second one. Basicly I just need a consultation from someone more expirienced than me).

The normalized data is saved to new CSV files, and then loaded back in for training the model. The script chooses the relevant features and target parameter(column) for training and testing.

Then the RandomForestRegressor is declareting **(I used random_state=42, just to be sure about model accuracy and predictability during the next using(I tried standart 22 parameter, but after testing I find that 42 is working better)** and trained using the training set, then tested on the testing set, and the R2 score, mean squared error, and mean absolute error are printed.

Finally, I just type into the model normalized data which I already know, and model predicts the target value, and prints the result which I already knew. It's just for testing and model doing this good. 
