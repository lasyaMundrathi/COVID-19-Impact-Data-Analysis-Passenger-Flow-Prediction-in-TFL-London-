import json
import pickle
from math import ceil
from flask import Flask,request,app,jsonify,url_for,render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import SGDRegressor
from datetime import datetime


app=Flask(__name__)
## Load the model
regmodel=pickle.load(open('XGBoost_model.pkl','rb'))

# load df_city.csv require to map the input station_name to get station_type and zone 
df_station_name = pd.read_csv('df_city.csv')
# load time_order.csv to map time to tfl hour format (e.g. 9.45 to 9.75) 
time_order = pd.read_csv('time_order.csv') 
# load station_mean_encoded.json to map the mean encoded feature to all the records
with open("station_mean_encoded.json","r") as f:
    df_station_mean = json.load(f)
# sample_for_deployment.csv data is from 28th Feb 2021 to 29th April 2021 which is 
# required to calcalute the lag features of the input date w.r.t N
sample_data = pd.read_csv("./sample_for_deployment.csv")

def get_station_name(station_name):
  station_type = list(df_station_name[df_station_name['STATIONNAME']==station_name]['STATIONTYPE'])[0]
  zone = list(df_station_name[df_station_name['STATIONNAME']==station_name]['ZONE'])[0]
  return zone, station_name, station_type

def create_lag_features_for_new_record(new_data, lag_features, lags):
  # Calculate lag features for the new input record
    station_name = new_data['STATIONNAME'].iloc[0]
    nlc = new_data['NLC'].iloc[0]
    entry_exit = new_data['ENTRYEXIT'].iloc[0]
    historical_data = sample_data[["CALENDARDATE","DOW","NLC","HOUR","ENTRYEXIT","N","ZONE","LOCKDOWN_REGIONAL"
                                   ,"SHOPS_CLOSED","PUBS_CLOSED","SCHOOLS_CLOSED","STATIONNAME","STATIONTYPE"]]
    historical_data_filtered = historical_data[
        (historical_data['STATIONNAME'] == station_name) &
        (historical_data['NLC'] == nlc) &
        (historical_data['ENTRYEXIT'] == entry_exit)
    ]
    combined_data = pd.concat([historical_data_filtered, new_data], ignore_index=True)
    combined_data.sort_values(['STATIONNAME', 'CALENDARDATE', 'NLC', 'HOUR', 'ENTRYEXIT'], inplace=True)
    
    for feature in lag_features:
        for lag in lags:
            lag_col_name = f"{feature[0]}_lag{lag}"
            combined_data[lag_col_name] = combined_data.groupby(['STATIONNAME', 'NLC', 'HOUR', 'ENTRYEXIT'])[feature].shift(lag)
    
    new_data_with_lags = combined_data.iloc[-len(new_data):].reset_index(drop=True)
    print(new_data_with_lags)
    if "N_lag7" in new_data_with_lags.columns:
        return [int(new_data_with_lags["N_lag7"][0]), int(new_data_with_lags["N_lag14"][0]), int(new_data_with_lags["N_lag21"][0])]

def get_lags(date_1, station_name, NLC, hour, entry_exit):
    new_single_record = pd.DataFrame({
    'CALENDARDATE': [pd.Timestamp(date_1)],
    'STATIONNAME': [station_name],
    'NLC': [NLC],
    'HOUR': [hour],
    'ENTRYEXIT': [entry_exit],
})
    lag_features = ['N'],
    lags = [7, 14, 21]
    return create_lag_features_for_new_record(new_single_record, lag_features, lags)

def extract_date(date_str, features):
    # Convert the date string to a Pandas Timestamp object
    date = pd.Timestamp(datetime.strptime(date_str, '%Y-%m-%d'))
    # Extract features
    features['Year'] = date.year if date.year not in [2020, 2021] else (0 if date.year == 2020 else 1)
    features['Month'] = date.month
    features['Day'] = date.day
    Dow =  date.strftime('%a').upper() 
    dow_mapping = {'MON': 1, 'TUE': 2, 'WED': 3, 'THU': 4, 'FRI': 5, 'SAT': 6, 'SUN': 7} 
    features['DOW'] =  dow_mapping[Dow] if Dow in dow_mapping else 0
    features['Quarter'] = date.quarter  
    features['Is_Weekend'] = 1 if date.dayofweek in [5, 6]  else 0  # 5 and 6 correspond to Saturday and Sunday
    features['week_of_month'] = (date.day + date.dayofweek) // 7 + 1
    print(f"features:{features}")
    return features

# converting hour format to tfl hour format
def time_hour(a):
  a_list  = a.split(':')
  k = int(a_list[1])//15 
  hour = int(a_list[0])
  if hour < 1:
    hour = hour+24
  if k==0:
    decimel = 0
  if k==1:
    decimel =0.25
  if k==2:
    decimel= 0.50
  if k==3:
    decimel= 0.75
  final_time = hour  + decimel
  return float(final_time)

@app.route('/')
def home():
    return render_template('home_tap.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    
    features = {}
    data=request.json['data']
    station_name = data['STATIONNAME']
    time = data['TIME']
    date_1 = data['DATE']
    is_lockdown = data['LOCKDOWN']
    features['NLC'] = data['NLC']
    features['hour'] = time_hour(time)
    features['entry_exit'] = 0 if data['ENTRY_EXIT'] == "Entry" else 1
    zone,station_name,station_type = get_station_name(station_name)
    features["zone"] =  int(zone)
    features['SHOPS_CLOSED'] = 1 if int(is_lockdown) == 1 else 0
    features['PUBS_CLOSED'] =  1 if int(is_lockdown) == 1 else 0
    features['SCHOOLS_CLOSED'] =  1 if int(is_lockdown) == 1 else 0
    features = extract_date(date_1, features)
    print(station_type)
    station_type_dict = {"Airport":0, "City":0, "Inner Suburb":0, "Outer Suburb":0, "Shopping":0, "Terminus":0, "Tourist":0}
    if station_type in station_type_dict:
        station_type_dict[station_type] = 1
    features['station_type'] = list(station_type_dict.values())
    features['lockdown'] = [0,1,0,0,0] if is_lockdown == 1 else [0,0,0,0,0]
   
    if station_name in df_station_mean:
       features['station_mean_encode'] = df_station_mean[station_name]
    else:
       features['station_mean_encode'] = 0
  
    features['lags7_14_21'] = get_lags(date_1, station_name, features['NLC'], features['hour'], data['ENTRY_EXIT'])
      

    input_features = [features['DOW'], features['NLC'], features['hour'], features['entry_exit'], features["zone"],
     features['SHOPS_CLOSED'], features['PUBS_CLOSED'], features['SCHOOLS_CLOSED'], features['Year'],
     features['Month'], features['Day'], features['Quarter'], features['Is_Weekend'], features['week_of_month']
    ]
    input_features = input_features + features['station_type']
    input_features = input_features + features['lockdown']
    input_features = input_features + features['lags7_14_21']
    input_features.append(features['station_mean_encode'])
    inp = np.array(input_features)
    prediction = abs(regmodel.predict([inp]))

    return str(np.round(prediction))

@app.route('/predict',methods=['POST'])
def predict():
    data=[x for x in request.form.values()]
    print(data)
    features = {}
    station_name = data[0]
    time = data[5]
    date_1 = data[1]
    is_lockdown = data[2]
    features['NLC'] = data[3]
    features['hour'] = time_hour(time)
    features['entry_exit'] = 0 if data[4] == "Entry" else 1
    zone,station_name,station_type = get_station_name(station_name)
    features["zone"] =  int(zone)
    features['SHOPS_CLOSED'] = 1 if int(is_lockdown) == 1 else 0
    features['PUBS_CLOSED'] =  1 if int(is_lockdown) == 1 else 0
    features['SCHOOLS_CLOSED'] =  1 if int(is_lockdown) == 1 else 0
    features = extract_date(date_1, features)
    print(station_type)
    station_type_dict = {"Airport":0, "City":0, "Inner Suburb":0, "Outer Suburb":0, "Shopping":0, "Terminus":0, "Tourist":0}
    if station_type in station_type_dict:
        station_type_dict[station_type] = 1
    features['station_type'] = list(station_type_dict.values())
    features['lockdown'] = [0,1,0,0,0] if is_lockdown == 1 else [0,0,0,0,0]
   
    if station_name in df_station_mean:
       features['station_mean_encode'] = df_station_mean[station_name]
    else:
       features['station_mean_encode'] = 0
  
    features['lags7_14_21'] = get_lags(date_1, station_name, features['NLC'], features['hour'], data[4])
      

    input_features = [features['DOW'], features['NLC'], features['hour'], features['entry_exit'], features["zone"],
     features['SHOPS_CLOSED'], features['PUBS_CLOSED'], features['SCHOOLS_CLOSED'], features['Year'],
     features['Month'], features['Day'], features['Quarter'], features['Is_Weekend'], features['week_of_month']
    ]
    input_features = input_features + features['station_type']
    input_features = input_features + features['lockdown']
    input_features = input_features + features['lags7_14_21']
    input_features.append(features['station_mean_encode'])
    inp = np.array(input_features)
    prediction = abs(regmodel.predict([inp]))

    return render_template("home_tap.html",prediction_text="The number of Taps = {}".format(str(np.round(prediction))))

if __name__=="__main__":
    app.run(debug=True)

