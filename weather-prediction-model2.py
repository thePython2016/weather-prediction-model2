import pickle as pickle
import pandas as pd
import streamlit as st

# load model and encoders
model = pickle.load(open('model.pkl', 'rb'))
label = pickle.load(open('label.pkl', 'rb'))

tab1,tab2=st.tabs(['Form','Upload'])

temperature = st.number_input('Temperature')
humidity = st.number_input('Humidity')
wind_speed = st.number_input('Wind Speed')
cloud_cover = st.number_input('Cloud Cover')
pressure = st.number_input('Pressure')
button1 = st.button('Predict Weather')


if button1:
    dataWeather = pd.DataFrame({
        "Temperature": [temperature],
        "Humidity": [humidity],
        "Wind_Speed": [wind_speed],
        "Cloud_Cover": [cloud_cover],
        "Pressure": [pressure]
    })
  
    
    predict = model.predict(dataWeather)
    target=label.inverse_transform(predict)
    dataWeather['Predicted Weather'] = target
    st.write(dataWeather)