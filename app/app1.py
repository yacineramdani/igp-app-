from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import kurtosis
import scipy.stats as sp
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load('trained_qda_model2.pkl')

# Function for preprocessing data
def preprocess_data(data):
    # Function to classify each row of data as either day (0) or night (1) 
    def day_or_night(dataframe, day_start, day_end):
        def day_night_test(time):
            if day_start <= time.hour < day_end:
                return 0
            else:
                return 1

        # Convert 'timestamp' column to datetime type
        dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'])

        # Apply day_night_test function to 'timestamp' column and create 'day_night' column
        dataframe['day_night'] = dataframe['timestamp'].apply(day_night_test)

        return dataframe

    DN_df = day_or_night(data, 8, 20)

    # Function to create a field of active (1) and non-active (0) time
    def active_nonactive(dataframe):
        def time_test(activity):
            if activity < 5:
                return 0
            else:
                return 1 

        new_df = dataframe.copy()

        new_df['col1'] = new_df['activity'].apply(time_test)

        new_df['col2'] = new_df['col1'].rolling(window=11, center=True, min_periods=1).sum()

        def activ_test(value):
            if value >= 2:
                return 1    
            elif value < 2:
                return 0
            else:
                return np.nan

        new_df['active_inactive'] = new_df['col2'].apply(activ_test)

        new_df.drop(['col1','col2'], axis=1, inplace=True)

        return new_df

    active_df = active_nonactive(DN_df)

    def light_dark(dataframe):
        dataframe['time'] = pd.to_datetime(dataframe['timestamp']).dt.time

        sunrise = pd.to_datetime('06:00:00').time()
        sunset = pd.to_datetime('18:00:00').time()

        time = dataframe['time']

        light_dark = []
        for n in range(0, len(time)):
            if sunrise <= time[n] < sunset:
                light_dark.append(0)
            else:
                light_dark.append(1)

        dataframe['light_dark'] = light_dark

        dataframe.drop(['time'], axis=1, inplace=True)

        return dataframe

    LD_df = light_dark(active_df)

    # Function to extract only the full days
    def fullDays(dataframe):
        df_new = pd.DataFrame({})

        for _, participant_data in dataframe.groupby('date'):
            min_timestamp = participant_data['timestamp'].min().date()
            max_timestamp = participant_data['timestamp'].max().date()

            min_date = min_timestamp
            max_date = max_timestamp

            df_maxchange = participant_data[(participant_data['timestamp'].dt.date >= min_date) & 
                                            (participant_data['timestamp'].dt.date < max_date + timedelta(days=1))]

            df_new = pd.concat([df_new, df_maxchange])

        return df_new

    fulldays_df = fullDays(LD_df)

    def weekDays(dataframe):
        df_new = pd.DataFrame({})

        dataframe['date'] = pd.to_datetime(dataframe['date'])

        for _, participant_data in dataframe.groupby('date'):
            min_date = participant_data['date'].min()
            max_date = participant_data['date'].max()

            week_end = min_date + timedelta(7) if (max_date - min_date).days <= 7 else min_date + timedelta(14)

            df_week = participant_data[participant_data['date'] < week_end]

            df_new = pd.concat([df_new, df_week])

        return df_new

    clean_df = weekDays(fulldays_df)

    clean_df['day'] = pd.to_datetime(clean_df['timestamp']).dt.day

    def Percentzero(values):
        zeros = (values == 0).sum().sum()
        total_values = values.size
        return zeros / total_values * 100

    def extract_features(dataframe):
        grouped = dataframe.groupby(['day'])['activity']

        features1_df = pd.DataFrame({
            'mean': grouped.mean(),
            'std': grouped.std(),
            '%zero': grouped.apply(Percentzero),
            'kurtosis': grouped.apply(lambda x: sp.kurtosis(x, fisher=False))
        })

        return features1_df

    features1 = extract_features(clean_df).reset_index()

    def activeAtNight(dataframe):
        grouped = dataframe.groupby('day')

        dfs_to_concat = []

        for day, group in grouped:
            inactive_day = len(group[(group['day_night'] == 0) & (group['active_inactive'] == 0)])
            active_night = len(group[(group['day_night'] == 1) & (group['active_inactive'] == 1)])
            night = len(group[group['day_night'] == 1])
            day_count = len(group[group['day_night'] == 0])
            inactive_light = len(group[(group['light_dark'] == 0) & (group['active_inactive'] == 0)])
            active_dark = len(group[(group['light_dark'] == 1) & (group['active_inactive'] == 1)])
            dark = len(group[group['light_dark'] == 1])
            light = len(group[group['light_dark'] == 0])        

            active_night_percent = active_night / night if night != 0 else 0
            inactive_day_percent = inactive_day / day_count if day_count != 0 else 0
            active_dark_percent = active_dark / dark if dark != 0 else 0
            inactive_light_percent = inactive_light / light if light != 0 else 0

            df_ndld = pd.DataFrame({'day': [day],
                                'activeNight': [active_night_percent],
                                'inactiveDay': [inactive_day_percent],
                                'activeDark': [active_dark_percent],
                                'inactiveLight': [inactive_light_percent]})
        
            dfs_to_concat.append(df_ndld)

        df_ndld = pd.concat(dfs_to_concat, ignore_index=True)

        return df_ndld


    AN = activeAtNight(clean_df)

    features_full = pd.merge(AN, features1, on=['day'], how='inner')

    features_full.drop(['%zero','activeNight','inactiveDay', 'kurtosis', 'day'], axis=1, inplace=True)

    return features_full

# Function to perform prediction
def perform_prediction(data):
    # Preprocess the input data
    df = preprocess_data(data)
    
    # Scale the features (if needed)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    # Make prediction using the trained model
    prediction = model.predict(X_scaled)
    
    return prediction

def get_most_frequent_prediction(predictions):
    """
    Get the most frequent prediction from a list of predictions.
    """
    unique_predictions, counts = np.unique(predictions, return_counts=True)
    most_frequent_prediction = unique_predictions[np.argmax(counts)]
    return most_frequent_prediction


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
                prediction = perform_prediction(df)

                # Determine the most frequent prediction
                most_frequent_prediction = get_most_frequent_prediction(prediction)

                # Map prediction value to meaningful labels
                prediction_label = "This person seems to be Depressed" if most_frequent_prediction == 1 else "This person does not seem to be Depressed"

                # Return prediction result as JSON
                return jsonify({'prediction': prediction_label})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid request method'}), 405


if __name__ == '__main__':
    app.run(debug=True)


