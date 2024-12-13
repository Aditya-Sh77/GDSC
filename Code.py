import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error


data = pd.read_csv('student-mat.csv', sep=';', index_col=False)
for col in data.columns:
    print(data[col].value_counts())
print(data['school'].value_counts())
data.columns

# Convert categorical columns to integers
for col in data.select_dtypes(include=['object']).columns:
    data[col] = data[col].astype('category').cat.codes

#training and analysis on all features    
X = data.drop('G3', axis=1)
y = data['G3']  # Final grade

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE with all features without parameter tuning: {rmse}")



# Training with tuning parameters
# Train Decision Tree Regressor
model = DecisionTreeRegressor(max_depth=10, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE with all features with parameter tuning: {rmse}")




# Feature Importance
feature_importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
feature_importance_df

#dropping features with no signficance
X = data.drop('G3', axis=1)
X = X.drop('higher', axis=1)
X = X.drop('sex', axis = 1)

y = data['G3']  # Final grade

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE after removing least significant features: {rmse}")



# Training with top 10 important features
# Get the top 10 contributors
top_10_features = feature_importance_df.head(15)

# Create a new DataFrame with only the top 10 features
X_top_10 = data[top_10_features['Feature'].values]

# Add the target variable back to the DataFrame
X_top_10['G3'] = data['G3']


#If you want to retrain your model on the top 10 features:

X = X_top_10.drop('G3', axis=1)
y = X_top_10['G3']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)

print(f"RMSE with top 15 features: {rmse}")


