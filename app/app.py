from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import kurtosis
import scipy.stats as sp
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Load the trained machine learning models
model_both = joblib.load('both_2_feature_gradboost_model.pkl')
model_male = joblib.load('female_2_feature_gradboost_model.pkl')
model_female = joblib.load('male_2_feature_lgbm_model.pkl')

# Function for preprocessing data
def preprocess_data(data):

    def preprocess_full_days(df, days_to_keep):
        """
        Preprocesses a DataFrame by filtering out rows where the count is not equal to the specified number of days for each date.

        Args:
            df (pandas.DataFrame): The input DataFrame.
            days_to_keep (int): Number of days to keep for each user.

        Returns:
            pandas.DataFrame: The preprocessed DataFrame.
        """

        # number of minutes to keep for each user
        minutes_to_keep = days_to_keep * 1440

        # filter rows for each date
        def filter_rows(group):
            return group.head(minutes_to_keep)

        # filter rows based on the number of days to keep
        filtered_df = df.groupby('date', group_keys=False).apply(filter_rows)

        return filtered_df
    FD_df = preprocess_full_days(data, 7)

    # CSV file
    sunlight_df = pd.read_csv('./Norway_Sunlight.csv')
    
    # change sunrise and sunset to datetime
    sunlight_df['Sunrise'] = pd.to_datetime(sunlight_df['Sunrise'], format='%H:%M')
    sunlight_df['Sunset'] = pd.to_datetime(sunlight_df['Sunset'], format='%H:%M')
    
    #classify each row of data as either light (0) or dark (1)
    FD_df['timestamp'] = pd.to_datetime(FD_df['timestamp'])
    def light_dark(dataframe, sunlight_df):
    
        # merge the sunlight data with the main df
        dataframe['month'] = dataframe['timestamp'].dt.month
        
        merged_df = pd.merge(dataframe, sunlight_df, left_on='month', right_on='Month', how='left')
        
        # convert sunrise and sunset times to datetime.time for comparison
        merged_df['sunrise_time'] = pd.to_datetime(merged_df['Sunrise'], format='%H:%M:%S').dt.time
        merged_df['sunset_time'] = pd.to_datetime(merged_df['Sunset'], format='%H:%M:%S').dt.time
       
        # classify as light or dark based on the timestamp
        merged_df['light_dark'] = merged_df.apply(lambda row: 0 if row['sunrise_time'] <= row['timestamp'].time() < row['sunset_time'] else 1, axis=1)
        return merged_df
    LD_df = light_dark(FD_df, sunlight_df)


    # classify each row of data as either day (0) or night (1)
    def day_or_night(dataframe, day_start, day_end):
        dataframe['day_night'] = dataframe['timestamp'].dt.hour.apply(lambda hour: 0 if day_start <= hour < day_end else 1)
        return dataframe
    DN_df = day_or_night(LD_df, 8, 20 )

    # create a field of active (1) and non-active (0) time
    def active_nonactive(dataframe, activity_threshold=5, rolling_window=11, rolling_threshold=2):
        dataframe['active_inactive'] = dataframe['activity'].apply(lambda x: 1 if x >= activity_threshold else 0)
        dataframe['rolling_sum'] = dataframe['active_inactive'].rolling(window=rolling_window, center=True).sum()
        dataframe['active_inactive_period'] = dataframe['rolling_sum'].apply(lambda x: 1 if x >= rolling_threshold else 0)
        dataframe.drop('rolling_sum', axis=1, inplace=True)
        return dataframe
    AN_df = active_nonactive(DN_df)

    # calculate the percentage of zeros in a series
    def percent_zero(series):
        zeros = (series == 0).sum()
        total_values = series.size
        return zeros / total_values * 100
    zeros = percent_zero(AN_df)

    def extract_features(dataframe):
        grouped = dataframe.groupby(['date'])['activity']
        features_df = grouped.agg(
            mean='mean',
            std='std',
            median='median', 
            q1=lambda x: np.percentile(x, 25),  # Add 1st quartile calculation
            q3=lambda x: np.percentile(x, 75),  # Add 3rd quartile calculation
            percent_zero=percent_zero,
            kurtosis=lambda x: sp.kurtosis(x, fisher=False)
        ).reset_index()
        features_df['kurtosis'] = features_df['kurtosis'].fillna(0)
        return features_df
    features = extract_features(AN_df).reset_index()


    def activity_proportions(dataframe):
        # Create empty lists to store results
        inactive_day_prop = []
        active_night_prop = []
        inactive_light_prop = []
        active_dark_prop = []

        # Calculate proportions
        # inactiveDay
        inactive_day_prop = dataframe.loc[dataframe['day_night'] == 0, 'active_inactive'].mean()

        # activeNight
        active_night_prop = dataframe.loc[dataframe['day_night'] == 1, 'active_inactive'].mean()

        # inactiveLight
        inactive_light_prop = dataframe.loc[dataframe['light_dark'] == 0, 'active_inactive'].mean()

        # activeDark
        active_dark_prop = dataframe.loc[dataframe['light_dark'] == 1, 'active_inactive'].mean()

        # Create DataFrame for results
        results = pd.DataFrame({
            'date': dataframe['date'].unique(),
            'inactiveDay': inactive_day_prop,
            'activeNight': active_night_prop,
            'inactiveLight': inactive_light_prop,
            'activeDark': active_dark_prop
        })

        # Drop unnecessary columns from original DataFrame
        columns_to_drop = ['day_night', 'active_inactive', 'active_inactive_period','light_dark', 'timestamp', 'month', 'Sunrise', 'Sunset', 'sunrise_time', 'sunset_time', 'activity']
        dataframe.drop(columns=columns_to_drop, inplace=True)

        # Merge the results back into the original DataFrame
        dataframe = pd.merge(dataframe, results, on='date', how='left')
        

        # Remove duplicates
        dataframe.drop_duplicates(inplace=True)
        

        return dataframe
    
    features1 = activity_proportions(AN_df)
    print(features1)

    def calculate_all_features(dataframe, sunlight_df):
        # convert 'timestamp' to datetime
        dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'])

        # light/dark classification using Norway sunlight data
        dataframe = light_dark(dataframe, sunlight_df)

        # day/night classification
        dataframe = day_or_night(dataframe, 8, 21)  # day is 08:00-20:59 inclusive

        # active/non-active classification
        dataframe = active_nonactive(dataframe)

        # statistical features
        statistical_features = extract_features(dataframe)

        # active/inactive periods
        period_features = activity_proportions(dataframe)

        # merge all features
        all_features = pd.merge(period_features, statistical_features, on=['date'], how='inner')

        # Select only 'inactiveDay' and 'activeNight' columns
        all_features = all_features[['date', 'inactiveDay', 'activeNight']]

        return all_features
    
    all_features = calculate_all_features(data, sunlight_df)
    
# Function to perform prediction
def perform_prediction(data, gender):
    # Preprocess the data
    preprocessed_data = preprocess_data(data)
    
    # Determine which model to use based on gender
    if gender == 'male':
        model = model_male
    elif gender == 'female':
        model = model_female
    else:
        model = model_both

    try:
        # Perform prediction
        prediction = model.predict(preprocessed_data)
        return prediction.tolist()  # Convert prediction to list before returning
    except Exception as e:
        return str(e)  # Return error message if prediction fails


# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Check if the request contains a file
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        # Check if the file is a CSV file
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if file and file.filename.endswith('.csv'):
            try:
                # Process the uploaded file (read CSV, preprocess data, perform prediction)
                df = pd.read_csv(file)
                gender = request.form.get('gender')  # Get gender information from form
                prediction = perform_prediction(df, gender)  # Call perform_prediction with the data and gender

                # Return raw prediction results as JSON
                return jsonify({'prediction': prediction.tolist()})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid request method'}), 405



if __name__ == '__main__':
    app.run(debug=True)
