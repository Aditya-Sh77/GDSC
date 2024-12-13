# GDSC
# PROBLEM STATEMENT
Title: AI-Based Grade Prediction <br>
Description:
Build a model to predict a student’s final grade based on features such as
attendance, participation, assignment scores, and exam marks.

# APPROACH
This code aims to develop a predictive model for student performance, specifically focusing on the final grade (G3).
Data Loading and Preprocessing: The 'student-mat.csv' dataset is loaded using Pandas. Categorical features are converted to numerical representations using label encoding to facilitate model training.

Feature Selection: All features in the dataset, except the target variable (G3), are used as predictors.

Model Selection: A Decision Tree Regressor is chosen as the prediction model due to its ability to capture complex relationships between features and the target variable.

Model Training and Evaluation: The dataset is split into training and testing sets. The Decision Tree Regressor is trained on the training data and evaluated on the testing data using the Root Mean Squared Error (RMSE) as the primary performance metric.
Hyperparameter tuning improved the performace significantly.

Feature Importance Analysis: The importance of each feature in the model's predictions is analyzed to identify key factors influencing student performance.

Re-Training Model with Top-10 Performance features: Model is re-trained withremoving least significant features only top 10 features which gave almost equal performance metric (RMSE).

# CHALLENGES
1.) The dataset included 33 features and less than 400 entries.
2.) Many of these featues were categorical which needed to be converted into numerical categories for training.
3.) Hyperparameter tuning on more features improved the performance but reduced the performance metric for lesser features.
