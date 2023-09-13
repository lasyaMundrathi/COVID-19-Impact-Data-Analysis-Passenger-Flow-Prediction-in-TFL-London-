# Predicting the number of taps using Flask API

predict_api function is called as an entry point which takes input data such station name, date, lockdown, NLC, Entry exit, Time
Output is the number of Tap prediction a particular station name at a specific time

## Instructions to run:
1. Install the dependencies using requirements.txt
1. run python ./app.py  
1. I have used POSTMAN for  
1. As the server is running now you can call the API on http://127.0.0.1:5000/predict_api using POST request.
1. In post request send body as 
```
{
      "data" : {
       "STATIONNAME": "Stratford",
        "DATE": "2021-04-30",
        "LOCKDOWN": 1,
        "NLC": 719,
        "ENTRY_EXIT": "Entry",
        "TIME": "09:45"
    }}
```


