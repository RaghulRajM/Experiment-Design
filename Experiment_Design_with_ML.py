import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Core libraries
import pandas as pd
import numpy as np

# Modeling libraries
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb

# Connector packages
import seaborn as sns

# 3.3 Import the Data
# Import data
control_df = pd.read_csv("data/control_data.csv")
experiment_df = pd.read_csv("data/experiment_data.csv")

# 3.4 Investigate the Data
# Display the first 5 rows
print(control_df.head(5))

# Display information about the control group
print(control_df.info())

# Display information about the experiment group
print(experiment_df.info())

# 3.5 Data Quality Check
# 3.5.1 Check for Missing Data
# Check for missing data in the control group
print(control_df.isnull().sum())

# Check for missing data in the experiment group
print(experiment_df.isnull().sum())

# Filter and display missing values in the control group
print(control_df[control_df['Enrollments'].isnull()])

# Filter and display missing values in the experiment group
print(experiment_df[experiment_df['Enrollments'].isnull()])

# Drop rows with missing values
control_df = control_df.dropna(subset=['Enrollments'])
experiment_df = experiment_df.dropna(subset=['Enrollments'])

# 3.5.2 Check Data Format
# Display the format of the control group data
print(control_df.dtypes)

# 3.6 Format Data
# Combine control and experiment data
data_formatted_df = pd.concat([control_df, experiment_df], keys=['Control', 'Experiment'])

# Reset index and rename the level_0 column to 'Experiment'
data_formatted_df = data_formatted_df.reset_index().rename(columns={'level_0': 'Experiment'})

# Add a 'row_id' column
data_formatted_df['row_id'] = range(1, len(data_formatted_df) + 1)

# Create a 'Day of Week' feature
data_formatted_df['DOW'] = pd.to_datetime(data_formatted_df['Date']).dt.strftime('%a').astype('category')

# Drop unnecessary columns
data_formatted_df = data_formatted_df.drop(['Date', 'Payments'], axis=1)

# Shuffle the rows
data_formatted_df = data_formatted_df.sample(frac=1, random_state=123)

# Display the formatted data
print(data_formatted_df.head())

# 3.7 Training and Testing Sets
# Split the data into training and testing sets
train_df, test_df = train_test_split(data_formatted_df, test_size=0.2, random_state=123)

# Display the training data
print(train_df.head())

# Display the testing data
print(test_df.head())

# Import necessary libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt

# Helper function to calculate metrics
def calc_metrics(model, new_data):
    predictions = model.predict(new_data.drop(columns=['row_id', 'Enrollments']))
    rmse = np.sqrt(mean_squared_error(new_data['Enrollments'], predictions))
    r_squared = r2_score(new_data['Enrollments'], predictions)
    mae = mean_absolute_error(new_data['Enrollments'], predictions)
    
    return rmse, r_squared, mae

# Helper function to plot predictions
def plot_predictions(model, new_data):
    predictions = model.predict(new_data.drop(columns=['row_id', 'Enrollments']))
    df = pd.DataFrame({'observation': new_data['row_id'], 'actual': new_data['Enrollments'], 'predicted': predictions})
    
    # Visualize
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='observation', y='value', hue='key', data=df.melt(id_vars='observation'), palette="deep")
    plt.title("Enrollments: Prediction vs Actual")
    plt.xlabel("Observation")
    plt.ylabel("Value")
    plt.show()

# 3.8.1 Linear Regression (Baseline)
# Create linear regression model
model_01_lm = LinearRegression()
model_01_lm.fit(train_df.drop(columns=['row_id', 'Enrollments']), train_df['Enrollments'])

# Calculate metrics on the test set
rmse, r_squared, mae = calc_metrics(model_01_lm, test_df)
print(f"RMSE: {rmse}\nR-squared: {r_squared}\nMAE: {mae}")

# Visualize predictions
plot_predictions(model_01_lm, test_df)

# 3.8.2 Helper Functions
# Helper function to calculate metrics
def calc_metrics(model, new_data):
    predictions = model.predict(new_data.drop(columns=['row_id', 'Enrollments']))
    rmse = np.sqrt(mean_squared_error(new_data['Enrollments'], predictions))
    r_squared = r2_score(new_data['Enrollments'], predictions)
    mae = mean_absolute_error(new_data['Enrollments'], predictions)
    
    return rmse, r_squared, mae

# Helper function to plot predictions
def plot_predictions(model, new_data):
    predictions = model.predict(new_data.drop(columns=['row_id', 'Enrollments']))
    df = pd.DataFrame({'observation': new_data['row_id'], 'actual': new_data['Enrollments'], 'predicted': predictions})
    
    # Visualize
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='observation', y='value', hue='key', data=df.melt(id_vars='observation'), palette="deep")
    plt.title("Enrollments: Prediction vs Actual")
    plt.xlabel("Observation")
    plt.ylabel("Value")
    plt.show()

'''
In this Python implementation:

We use scikit-learn's LinearRegression for linear regression modeling.
The calc_metrics function calculates RMSE, R-squared, and MAE metrics for evaluating the model's performance.
The plot_predictions function visualizes the predictions compared to actual values.
The linear regression model is trained on the training set and evaluated on the test set.
Metrics and predictions are displayed and visualized for analysis.'''

# Import necessary libraries
from sklearn.tree import DecisionTreeRegressor, export_text, export_graphviz
import graphviz
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 3.8.3 Decision Trees
# Decision Tree Helper Functions

# Helper function to calculate metrics
def calc_metrics(model, new_data):
    predictions = model.predict(new_data.drop(columns=['row_id', 'Enrollments']))
    rmse = np.sqrt(mean_squared_error(new_data['Enrollments'], predictions))
    r_squared = r2_score(new_data['Enrollments'], predictions)
    mae = mean_absolute_error(new_data['Enrollments'], predictions)
    
    return rmse, r_squared, mae

# Helper function to plot predictions
def plot_predictions(model, new_data):
    predictions = model.predict(new_data.drop(columns=['row_id', 'Enrollments']))
    df = pd.DataFrame({'observation': new_data['row_id'], 'actual': new_data['Enrollments'], 'predicted': predictions})
    
    # Visualize
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='observation', y='value', hue='key', data=df.melt(id_vars='observation'), palette="deep")
    plt.title("Enrollments: Prediction vs Actual")
    plt.xlabel("Observation")
    plt.ylabel("Value")
    plt.show()

# Create decision tree model
model_02_decision_tree = DecisionTreeRegressor(
    criterion='mse',  # mean squared error
    splitter='best',
    max_depth=5,  # max tree depth
    min_samples_split=4  # min samples to split
)

# Fit the decision tree model
model_02_decision_tree.fit(train_df.drop(columns=['row_id', 'Enrollments']), train_df['Enrollments'])

# Calculate metrics on the test set
rmse, r_squared, mae = calc_metrics(model_02_decision_tree, test_df)
print(f"RMSE: {rmse}\nR-squared: {r_squared}\nMAE: {mae}")

# Visualize predictions
plot_predictions(model_02_decision_tree, test_df)

# Visualize the decision tree rules
dot_data = export_graphviz(model_02_decision_tree, out_file=None, feature_names=train_df.drop(columns=['row_id', 'Enrollments']).columns, filled=True, rounded=True, special_characters=True)
graphviz.Source(dot_data)

'''
In this Python implementation:

We use scikit-learn's DecisionTreeRegressor for decision tree modeling.
The calc_metrics function calculates RMSE, R-squared, and MAE metrics for evaluating the model's performance.
The plot_predictions function visualizes the predictions compared to actual values.
The decision tree model is trained on the training set and evaluated on the test set.
Metrics and predictions are displayed and visualized for analysis.
The decision tree rules are visualized using export_graphviz and graphviz. Adjust the parameters accordingly for a clear visualization.
'''

# Import necessary libraries
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt

# 3.8.4 XGBoost
# XGBoost Helper Functions

# Helper function to calculate metrics
def calc_metrics(model, new_data):
    predictions = model.predict(xgb.DMatrix(new_data.drop(columns=['row_id', 'Enrollments'])))
    rmse = np.sqrt(mean_squared_error(new_data['Enrollments'], predictions))
    r_squared = r2_score(new_data['Enrollments'], predictions)
    mae = mean_absolute_error(new_data['Enrollments'], predictions)
    
    return rmse, r_squared, mae

# Helper function to plot predictions
def plot_predictions(model, new_data):
    predictions = model.predict(xgb.DMatrix(new_data.drop(columns=['row_id', 'Enrollments'])))
    df = pd.DataFrame({'observation': new_data['row_id'], 'actual': new_data['Enrollments'], 'predicted': predictions})
    
    # Visualize
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='observation', y='value', hue='key', data=df.melt(id_vars='observation'), palette="deep")
    plt.title("Enrollments: Prediction vs Actual")
    plt.xlabel("Observation")
    plt.ylabel("Value")
    plt.show()

# Set seed for reproducibility (optional)
np.random.seed(123)

# Create XGBoost model
model_03_xgboost = xgb.XGBRegressor(
    objective='reg:squarederror',  # regression task
    n_estimators=1000,  # number of trees
    max_depth=6,  # max tree depth
    learning_rate=0.2,  # boosting learning rate
    min_child_weight=8,  # min sum of instance weight needed in a child
    subsample=1,  # subsample ratio of training instances
    colsample_bytree=100,  # subsample ratio of columns when constructing each tree
    gamma=0.01  # minimum loss reduction required to make a further partition
)

# Fit the XGBoost model
model_03_xgboost.fit(train_df.drop(columns=['row_id', 'Enrollments']), train_df['Enrollments'])

# Calculate metrics on the test set
rmse, r_squared, mae = calc_metrics(model_03_xgboost, test_df)
print(f"RMSE: {rmse}\nR-squared: {r_squared}\nMAE: {mae}")

# Visualize predictions
plot_predictions(model_03_xgboost, test_df)

# Get feature importance from the model
feature_importance = model_03_xgboost.get_booster().get_score(importance_type='weight')
feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
features, importance = zip(*feature_importance)
feature_importance_df = pd.DataFrame({'Feature': features, 'Weight': importance})

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Weight', y='Feature', data=feature_importance_df, palette="deep")
plt.title("XGBoost Feature Importance")
plt.xlabel("Weight")
plt.show()

'''
We use the xgboost library for XGBoost modeling.
The calc_metrics function calculates RMSE, R-squared, and MAE metrics for evaluating the model's performance.
The plot_predictions function visualizes the predictions compared to actual values.
The XGBoost model is trained on the training set and evaluated on the test set.
Metrics and predictions are displayed and visualized for analysis.
Feature importance is extracted and visualized using a bar plot.
'''

