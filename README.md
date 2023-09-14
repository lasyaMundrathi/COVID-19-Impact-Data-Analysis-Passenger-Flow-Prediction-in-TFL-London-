# Data-Driven-Analysis-on-the-Impact-of-Covid-19-in-TfL

## Objective

The goal of this project is to predict the number of taps (N) at London Underground stations using machine learning models, with a specific focus on the XGBoost Regressor. The project also involves evaluating the performance of these models for time-series forecasting.

## Background

Understanding the factors affecting passenger volumes at London Underground stations is crucial for optimizing station management and public transportation planning. The dataset used in this project is derived from Transport for London's Oyster and Contactless ticketing databases. It records daily arrivals and exits at each tube station in 15-minute intervals from January 1, 2020, through April 30, 2021. It also includes information about lockdown periods, station attributes, and other relevant features.

## Machine Learning Pipeline

### Data Preprocessing

- **Handling Missing Values**: NaN values in the dataset are replaced with zeros.
- **Mapping Categorical to Numerical**: Days of the week ('DOW') and entry/exit status ('ENTRYEXIT') are mapped to numerical values using label encoding.
- **Date Transformation**: The 'CALENDARDATE' is converted to a datetime format.

### Feature Engineering

- **Lagged Features**: Lagged features for 7, 14, and 21 days are created to capture historical data points.
- **Date Features**: Additional features like 'Year', 'Month', 'Day', 'Quarter', and 'Is_Weekend' are generated based on the 'CALENDARDATE'.
- **One-Hot Encoding**: Categorical variables like 'STATIONTYPE' and 'LOCKDOWN_REGIONAL' are one-hot encoded.
- **Mean Encoding**: The 'stationname' feature is mean encoded to capture station-specific patterns.

### Model Selection

- The primary model used is the XGBoost Regressor, and its performance is compared with other machine learning models like Random Forest, Linear Regression, and SVM.

### Training and Validation

- The dataset is split into training, validation, and test sets considering the time-series nature of the data.
- Evaluation metrics such as RMSE, MAE, and R-squared are used for model evaluation.

### Hyperparameter Tuning

- Model parameters are optimized using techniques like Random Search.

### Evaluation

- Model performance is assessed on the test set, and the results are compared across different models.

## Data Preprocessing

- The data is converted to NumPy arrays for efficient memory utilization.
- Dates earlier than '2021-02-28' are considered as the training dataset.
- The 'CALENDARDATE' column and the 'N' value are separated, with 'N' as the target variable.
- Data from dates after '2021-02-28' are considered as the test dataset.


