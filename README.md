# Farmer-Income-Prediction-Using-Machine-Learning
The project's main goal is to predict farmer income using a dataset that includes farmer demographics, land holdings, local climate information, and socio-economic indicators. The dataset, sourced from an Excel file named LTF Challenge data with dictionary.xlsx on the TrainData sheet, contains 47,970 entries and 105 features. 

# Step-by-Step Explanation of the Python Code
This code is a comprehensive script for building and evaluating machine learning models to predict farmer income. It follows a standard data science pipeline, from data loading and cleaning to model training, tuning, and evaluation. Here is a breakdown of each section.

# 1. Imports and Custom Functions
This section sets up the necessary tools and defines a key performance metric.

Libraries: The code starts by importing essential libraries:

pandas and numpy: For data manipulation and numerical operations.

sklearn: The core machine learning library for tasks like data splitting, preprocessing, and model evaluation.

xgboost, lightgbm, RandomForestRegressor: These are the specific machine learning models, all of which are powerful "ensemble" models known for high performance on tabular data.

matplotlib and seaborn: For creating data visualizations, such as the bar chart that compares model performance.

calculate_mape function: This custom function calculates the Mean Absolute Percentage Error (MAPE), which is the key evaluation metric for this project. It is defined to avoid errors when the actual income value is zero.

mape_scorer: This adapts the calculate_mape function so it can be used directly within sklearn's hyperparameter tuning tools, like RandomizedSearchCV. The greater_is_better=False argument tells the tool that a lower MAPE score is better.

# 2. Data Loading and Inspection
This part of the code securely loads your data and performs initial checks to ensure it's ready for processing.

File Path and Sheet Name: The code defines the exact location (file_path_excel) and the sheet name (sheet_name) of your data file. It uses a try...except block to gracefully handle potential FileNotFoundError or other issues if the file or sheet is not found.

Column Name Check: A critical step is the check for the target variable's column name. Because the name in your data is a specific string ('Target_Variable/Total Income'), the code first checks for a few possible names before renaming the correct one to Farmer_Income. This ensures the rest of the script can consistently refer to the income column.

Initial Data Overview: The script then prints df.info() and df.describe().

df.info() provides a summary of the data, including the number of entries, columns, data types, and non-null values. This is crucial for identifying which columns have missing data.

df.describe() gives a statistical summary of the numerical columns (e.g., mean, standard deviation, quartiles), which helps in understanding the data's distribution and identifying potential outliers.

# 3. Feature Engineering and Preprocessing
This is the core of the data preparation. The code transforms the raw data into a format that the machine learning models can understand and learn from.

Target Variable Transformation: The target variable, Farmer_Income, is transformed using np.log1p(). This is a common technique for highly skewed data like income, as it helps the models learn more effectively from the distribution. The predictions are later inverse-transformed using np.expm1() to get the values back into their original currency scale.

Temperature Data Parsing: This is a key feature engineering step to fix the previous issue. The code now correctly splits the Ambient temperature (min & max) string values (e.g., "23.34 /30.33") into separate min_temp, max_temp, and Avg_Ambient_Temp numerical columns. This ensures that valuable weather data is now being used by the models.

Pipeline for Preprocessing: To ensure all transformations are applied correctly and consistently, the code uses ColumnTransformer and Pipeline:

numerical_transformer: This pipeline handles all numerical columns by first imputing any missing values with the median and then scaling the data to have a mean of 0 and a standard deviation of 1.

categorical_transformer: This pipeline handles all categorical columns by first imputing missing values with the most frequent value and then converting the text categories into numerical format using one-hot encoding.

# 4. Model Training with Hyperparameter Tuning
Instead of just training a model once, this section uses a more advanced technique to find the best-performing version of each model.

train_test_split: The data is split into a training set (80%) and a testing set (20%). The models learn from the training data and are then evaluated on the unseen test data.

RandomizedSearchCV: This powerful tool is used to automatically find the best hyperparameters for each model.

It defines a range of parameter values to test for each model (e.g., n_estimators, learning_rate, max_depth).

It performs a specified number of randomized searches (n_iter=20) to find the combination of parameters that yields the best performance according to the MAPE score.

It uses KFold cross-validation with 5 splits (cv=KFold(n_splits=5)) to ensure the results are robust and not biased by a single data split.

Training Loop: The code iterates through each model (XGBoost, LightGBM, RandomForest), tunes its hyperparameters, and then trains the best version of that model.

# 5. Results and Visualization
This section presents the results of the model training and tuning.

Prediction and Inverse Transform: The best-tuned model is used to make predictions on the test set. The predictions are then converted back to the original income scale using np.expm1().

Final MAPE Calculation: The calculate_mape function is used one last time to measure the performance of the best-tuned model on the test data.

Model Comparison: A bar plot is generated to visually compare the final MAPE scores of all three models. The plot makes it easy to see which model performed best. The plot also displays the MAPE values directly on the bars for quick analysis.

# 6. Further Optimization Steps
The final part of the script outlines a roadmap for how you could further improve the model's performance to meet or exceed the 10% MAPE target. These steps include:

More extensive hyperparameter tuning.

Advanced feature engineering.

Outlier analysis.

Integrating external data to enhance the model's predictive power.
