from datetime import datetime
import numpy as np
import joblib
import pandas as pd
import pytz
import streamlit as st
from green_mood_tracker.clustering import lda_wordcloud
#from TaxiFareModel.data import get_data
#from TaxiFareModel.utils import geocoder_here

COLS = ['key',
        'pickup_datetime',
        'pickup_longitude',
        'pickup_latitude',
        'dropoff_longitude',
        'dropoff_latitude',
        'passenger_count']


st.markdown("# Green Mood Tracker")
st.markdown("**Energy Sentiment Analysis**")


@st.cache
def read_data():
    df = pd.read_csv('twint_dataset.csv')
    return df


def format_input(pickup, dropoff, passengers=1):
    pickup_datetime = datetime.utcnow().replace(
        tzinfo=pytz.timezone('America/New_York'))
    formated_input = {
        "pickup_latitude": pickup["latitude"],
        "pickup_longitude": pickup["longitude"],
        "dropoff_latitude": dropoff["latitude"],
        "dropoff_longitude": dropoff["longitude"],
        "passenger_count": passengers,
        "pickup_datetime": str(pickup_datetime),
        "key": str(pickup_datetime)}
    return formated_input


def main():
    analysis = st.sidebar.selectbox(
        "Select", ["Prediction", "Data Visualisation"])
    if analysis == "Data Visualisation":
        st.header("TaxiFare Basic Data Visualisation")

        year = st.slider('Year', min_value=2010, max_value=2020)
        year = np.datetime64(str(year))
        country_prediction = st.selectbox("Select Country", ['UK', 'USA'], 1)

        st.markdown("**Graphs**")

        data = 'green_mood_tracker/raw_data/twint_dataset.csv'
        df = pd.read_csv(data)
        df['date'] = pd.to_datetime(
            df['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        mask = (df['date'] <= year)
        df = df.loc[mask]
        # st.write(d)
        # st.write(df['date'])
        # st.write(year.dtype)
        st.write(df)

        #lda_wordcloud(df,'tweet', [2], [300], 'http://clipart-library.com/images/8T6ooLLpc.jpg')
        # st.pyplot()

    if analysis == "Prediction":
        #pipeline = joblib.load('data/model.joblib')
        print("loaded model")
        st.header("Green Mood Tracker Model Predictions")
        # inputs from user
        country_prediction = st.selectbox("Select Country", ['UK', 'USA'], 1)
        topic_prediction = st.selectbox("Select Topic", [
                                        'Climate Change', 'Energy Prices', 'Fossil Fuels', 'Green Energy', 'Nuclear Energy', 'Solar Energy', 'Wind Energy'], 1)
        d3 = st.date_input("Select TimeFrame", [])

        #dropoff_adress = st.text_input("dropoff adress", "434 6th Ave, New York, NY 10011")
        # Get coords from input adresses usung HERE geocoder
        #pickup_coords = geocoder_here(pickup_adress)
        #dropoff_coords = geocoder_here(dropoff_adress)
        # inputs from user
        # passenger_counts = st.selectbox("# passengers", [1, 2, 3, 4, 5, 6], 1)

        #data = pd.DataFrame([pickup_coords, dropoff_coords])
        #to_predict = [format_input(pickup=pickup_coords, dropoff=dropoff_coords, passengers=passenger_counts)]
        #X = pd.DataFrame(to_predict)
        #res = pipeline.predict(X[COLS])
        #st.write("💸 taxi fare", res[0])
        # st.map(data=data)


# print(colored(proc.sf_query, "blue"))
# proc.test_execute()
if __name__ == "__main__":
    #df = read_data()
    main()
