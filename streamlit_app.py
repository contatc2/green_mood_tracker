from green_mood_tracker.clustering import lda_wordcloud

import streamlit as st
import pytz
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

from green_mood_tracker.datavisstreamlit import all_plotting
from green_mood_tracker.datavisstreamlit import altair_plot_like, altair_plot_tweet
from green_mood_tracker.datavisstreamlit import plot_map
import plotly.express as px
import plotly.graph_objects as go
import altair as alt



#from TaxiFareModel.data import get_data
#from TaxiFareModel.utils import geocoder_here

COLS = ['key',
        'pickup_datetime',
        'pickup_longitude',
        'pickup_latitude',
        'dropoff_longitude',
        'dropoff_latitude',
        'passenger_count']

comment_dataframe = pd.read_csv("green_mood_tracker/raw_data/US/solar.csv")
altair_sent_by_year, altair_like_by_year, layout, data_slider = plot_map(comment_dataframe)


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

def sl_predict(country_prediction, topic_prediction, d3):

    st.write(type(country_prediction), type(topic_prediction), type(d3))
    st.write(country_prediction, topic_prediction, d3)

    return None

def main(data_slider,layout):
    analysis = st.sidebar.selectbox("Select", ["Prediction", "Data Visualisation"])
    if analysis == 'Data Visualisation':
        st.header('Sentiment')
        year = st.slider('Year', min_value = 2010, max_value = 2020)
        country_prediction = st.selectbox('Select Country', ['UK', 'USA'], 1)
        like_prediction = st.selectbox('Sentiment factor', ['Per Tweet', 'Likes Per Tweet'], 1)


        st.text(" \n")
        st.text(" \n")
        st.text(" \n")
        st.text(" \n")
        st.markdown('**Percentage positive sentiment towards solar energy by state in the USA since 2010**')
        #data = 'green_mood_tracker/raw_data/twint_US.csv'
        #df = pd.read_csv(data)
        #df['year']= pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S', errors= 'coerce').dt.year
        #df = df[df['year'] == year]
        fig = go.Figure(data=data_slider[abs(year-2020)], layout=layout)
        if like_prediction == 'Per Tweet':
            c= altair_plot_tweet(altair_sent_by_year,year)
            fig_pie = px.pie(altair_sent_by_year[abs(year-2020)].tail(3), values='Percentage of Sentiment', names='sentiment',color_discrete_sequence=px.colors.sequential.algae)


        elif like_prediction == 'Likes Per Tweet':
            c = altair_plot_like(altair_like_by_year,year)
            fig_pie = px.pie(altair_like_by_year[abs(year-2020)].tail(3), values='Percentage of Likes Per Sentiment', names='sentiment',color_discrete_sequence=px.colors.sequential.algae)
        st.plotly_chart(fig,use_container_width=True)
        st.altair_chart(c, use_container_width=True)
        st.plotly_chart(fig_pie)

        #st.write(df['tweet'])

        # lda_wordcloud(df,'tweet', [2], [300], 'http://clipart-library.com/images/8T6ooLLpc.jpg')
        # st.pyplot()

    if analysis == "Prediction":
        # pipeline = joblib.load('data/model.joblib')
        print("loaded model")
        st.header("Green Mood Tracker Model Predictions")
        # inputs from user
        country_prediction = st.selectbox("Select Country", ['UK', 'USA'], 1)
        topic_prediction = st.selectbox("Select Topic", [
                                        'Climate Change', 'Energy Prices', 'Fossil Fuels', 'Green Energy', 'Nuclear Energy', 'Solar Energy', 'Wind Energy'], 1)
        d3 = st.date_input("Select TimeFrame", [])

        sl_predict(country_prediction, topic_prediction, d3)

        # dropoff_adress = st.text_input("dropoff adress", "434 6th Ave, New York, NY 10011")
        # Get coords from input adresses usung HERE geocoder
        # pickup_coords = geocoder_here(pickup_adress)
        # dropoff_coords = geocoder_here(dropoff_adress)
        # inputs from user
        # passenger_counts = st.selectbox("# passengers", [1, 2, 3, 4, 5, 6], 1)

        # data = pd.DataFrame([pickup_coords, dropoff_coords])
        # to_predict = [format_input(pickup=pickup_coords, dropoff=dropoff_coords, passengers=passenger_counts)]
        # X = pd.DataFrame(to_predict)
        # res = pipeline.predict(X[COLS])
        # st.write("💸 taxi fare", res[0])
        # st.map(data=data)


# print(colored(proc.sf_query, "blue"))
# proc.test_execute()
if __name__ == "__main__":
    #df = read_data()
    main(data_slider,layout)