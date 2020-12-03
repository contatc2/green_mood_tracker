from green_mood_tracker.clustering import lda_wordcloud

from green_mood_tracker.predict import twint_prediction
from green_mood_tracker.utils import simple_time_tracker
from green_mood_tracker.data import get_twint_data

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
img = st.image('green_mood_tracker/raw_data/green_mood_tracker_logo.png',
               style='left', width=700, output_format='png')

st.markdown("**Energy Sentiment Analysis**")


@st.cache
def read_data():
    pass


@st.cache
def select_data(topic='Solar Energy', country='USA'):
    if country == 'USA':
        if topic == 'Climate Change':
            comment_dataframe_US_climate = pd.read_csv(
                "green_mood_tracker/raw_data/US/[_climate_, _change_].csv")
            altair_sent_by_year_US_climate, altair_like_by_year_US_climate, layout_US_climate, data_slider_US_climate = plot_map(
                comment_dataframe_US_climate, country='US')
            return altair_sent_by_year_US_climate, altair_like_by_year_US_climate, layout_US_climate, data_slider_US_climate
        elif topic == 'Energy Prices':
            comment_dataframe_US_prices = pd.read_csv(
                "green_mood_tracker/raw_data/US/[_energy_, _prices_].csv")
            altair_sent_by_year_US_prices, altair_like_by_year_US_prices, layout_US_prices, data_slider_US_prices = plot_map(
                comment_dataframe_US_prices, country='US')
            return altair_sent_by_year_US_prices, altair_like_by_year_US_prices, layout_US_prices, data_slider_US_prices
        elif topic == 'Green Energy':
            comment_dataframe_US_green = pd.read_csv(
                "green_mood_tracker/raw_data/US/[_green_, _energy_].csv")
            altair_sent_by_year_US_green, altair_like_by_year_US_green, layout_US_green, data_slider_US_green = plot_map(
                comment_dataframe_US_green, country='US')
            return altair_sent_by_year_US_green, altair_like_by_year_US_green, layout_US_green, data_slider_US_green
        elif topic == 'Nuclear Energy':
            comment_dataframe_US_nuclear = pd.read_csv(
                "green_mood_tracker/raw_data/US/[_nuclear_, _energy_].csv")
            altair_sent_by_year_US_nuclear, altair_like_by_year_US_nuclear, layout_US_nuclear, data_slider_US_nuclear = plot_map(
                comment_dataframe_US_nuclear, country='US')
            return altair_sent_by_year_US_nuclear, altair_like_by_year_US_nuclear, layout_US_nuclear, data_slider_US_nuclear
        elif topic == 'Fossil Fuels':
            comment_dataframe_US_fossil = pd.read_csv(
                "green_mood_tracker/raw_data/US/[_fossil_, _fuels_].csv")
            altair_sent_by_year_US_fossil, altair_like_by_year_US_fossil, layout_US_fossil, data_slider_US_fossil = plot_map(
                comment_dataframe_US_fossil, country='US')
            return altair_sent_by_year_US_fossil, altair_like_by_year_US_fossil, layout_US_fossil, data_slider_US_fossil
        elif topic == 'Solar Energy':
            comment_dataframe_US_solar = pd.read_csv(
                "green_mood_tracker/raw_data/US/[_solar_, _energy_].csv")
            altair_sent_by_year_US_solar, altair_like_by_year_US_solar, layout_US_solar, data_slider_US_solar = plot_map(
                comment_dataframe_US_solar, country='US')
            return altair_sent_by_year_US_solar, altair_like_by_year_US_solar, layout_US_solar, data_slider_US_solar
        elif topic == 'Wind Energy':
            comment_dataframe_US_wind = pd.read_csv(
                "green_mood_tracker/raw_data/US/[_wind_, _energy_].csv")
            altair_sent_by_year_US_wind, altair_like_by_year_US_wind, layout_US_wind, data_slider_US_wind = plot_map(
                comment_dataframe_US_wind, country='US')
            return altair_sent_by_year_US_wind, altair_like_by_year_US_wind, layout_US_wind, data_slider_US_wind
    elif country == 'UK':
        if topic == 'Climate Change':
            comment_dataframe_UK_climate = pd.read_csv(
                "green_mood_tracker/raw_data/UK/[_climate_, _change_].csv")
            altair_sent_by_year_UK_climate, altair_like_by_year_UK_climate, figs_UK_climate = plot_map(
                comment_dataframe_UK_climate, country='UK')
            return altair_sent_by_year_UK_climate, altair_like_by_year_UK_climate, figs_UK_climate

        elif topic == 'Energy Prices':
            comment_dataframe_UK_prices = pd.read_csv(
                "green_mood_tracker/raw_data/UK/[_energy_, _prices_].csv")
            altair_sent_by_year_UK_prices, altair_like_by_year_UK_prices, figs_UK_prices = plot_map(
                comment_dataframe_UK_prices, country='UK')
            return altair_sent_by_year_UK_prices, altair_like_by_year_UK_prices, figs_UK_prices

        elif topic == 'Green Energy':
            comment_dataframe_UK_green = pd.read_csv(
                "green_mood_tracker/raw_data/UK/[_green_, _energy_].csv")
            altair_sent_by_year_UK_green, altair_like_by_year_UK_green, figs_UK_green = plot_map(
                comment_dataframe_UK_green, country='UK')
            return altair_sent_by_year_UK_green, altair_like_by_year_UK_green, figs_UK_green

        elif topic == 'Nuclear Energy':
            comment_dataframe_UK_nuclear = pd.read_csv(
                "green_mood_tracker/raw_data/UK/[_nuclear_, _energy_].csv")
            altair_sent_by_year_UK_nuclear, altair_like_by_year_UK_nuclear, figs_UK_nuclear = plot_map(
                comment_dataframe_UK_nuclear, country='UK')
            return altair_sent_by_year_UK_nuclear, altair_like_by_year_UK_nuclear, figs_UK_nuclear

        elif topic == 'Fossil Fuels':
            comment_dataframe_UK_fossil = pd.read_csv(
                "green_mood_tracker/raw_data/UK/[_fossil_, _fuels_].csv")
            altair_sent_by_year_UK_fossil, altair_like_by_year_UK_fossil, figs_UK_fossil = plot_map(
                comment_dataframe_UK_fossil, country='UK')
            return altair_sent_by_year_UK_fossil, altair_like_by_year_UK_fossil, figs_UK_fossil

        elif topic == 'Solar Energy':
            comment_dataframe_UK_solar = pd.read_csv(
                "green_mood_tracker/raw_data/UK/[_solar_, _energy_].csv")
            altair_sent_by_year_UK_solar, altair_like_by_year_UK_solar, layout_UK_solar, data_slider_UK_solar = plot_map(
                comment_dataframe_UK_solar, country='UK')
            return altair_sent_by_year_UK_solar, altair_like_by_year_UK_solar, figs_UK_solar

        elif topic == 'Wind Energy':
            comment_dataframe_UK_wind = pd.read_csv(
                "green_mood_tracker/raw_data/UK/[_wind_, _energy_].csv")
            altair_sent_by_year_UK_wind, altair_like_by_year_UK_wind, figs_UK_wind = plot_map(
                comment_dataframe_UK_wind, country='UK')
            return altair_sent_by_year_UK_wind, altair_like_by_year_UK_wind, figs_UK_wind


def select_data(topic='Solar Energy', country='USA'):
    if country == 'USA':
        if topic == 'Climate Change':
            comment_dataframe_US_climate = pd.read_csv(
                "green_mood_tracker/raw_data/US/[_climate_, _change_].csv")
            altair_sent_by_year_US_climate, altair_like_by_year_US_climate, layout_US_climate, data_slider_US_climate = plot_map(
                comment_dataframe_US_climate, country='US')
            return altair_sent_by_year_US_climate, altair_like_by_year_US_climate, layout_US_climate, data_slider_US_climate

        elif topic == 'Energy Prices':
            comment_dataframe_US_prices = pd.read_csv(
                "green_mood_tracker/raw_data/US/[_energy_, _prices_].csv")
            altair_sent_by_year_US_prices, altair_like_by_year_US_prices, layout_US_prices, data_slider_US_prices = plot_map(
                comment_dataframe_US_prices, country='US')
            return altair_sent_by_year_US_prices, altair_like_by_year_US_prices, layout_US_prices, data_slider_US_prices

        elif topic == 'Green Energy':
            comment_dataframe_US_green = pd.read_csv(
                "green_mood_tracker/raw_data/US/[_green_, _energy_].csv")
            altair_sent_by_year_US_green, altair_like_by_year_US_green, layout_US_green, data_slider_US_green = plot_map(
                comment_dataframe_US_green, country='US')
            return altair_sent_by_year_US_green, altair_like_by_year_US_green, layout_US_green, data_slider_US_green

        elif topic == 'Nuclear Energy':
            comment_dataframe_US_nuclear = pd.read_csv(
                "green_mood_tracker/raw_data/US/[_nuclear_, _energy_].csv")
            altair_sent_by_year_US_nuclear, altair_like_by_year_US_nuclear, layout_US_nuclear, data_slider_US_nuclear = plot_map(
                comment_dataframe_US_nuclear, country='US')
            return altair_sent_by_year_US_nuclear, altair_like_by_year_US_nuclear, layout_US_nuclear, data_slider_US_nuclear

        elif topic == 'Fossil Fuels':
            comment_dataframe_US_fossil = pd.read_csv(
                "green_mood_tracker/raw_data/US/[_fossil_, _fuels_].csv")
            altair_sent_by_year_US_fossil, altair_like_by_year_US_fossil, layout_US_fossil, data_slider_US_fossil = plot_map(
                comment_dataframe_US_fossil, country='US')
            return altair_sent_by_year_US_fossil, altair_like_by_year_US_fossil, layout_US_fossil, data_slider_US_fossil

        elif topic == 'Solar Energy':
            comment_dataframe_US_solar = pd.read_csv(
                "green_mood_tracker/raw_data/US/[_solar_, _energy_].csv")
            altair_sent_by_year_US_solar, altair_like_by_year_US_solar, layout_US_solar, data_slider_US_solar = plot_map(
                comment_dataframe_US_solar, country='US')
            return altair_sent_by_year_US_solar, altair_like_by_year_US_solar, layout_US_solar, data_slider_US_solar

        elif topic == 'Wind Energy':
            comment_dataframe_US_wind = pd.read_csv(
                "green_mood_tracker/raw_data/US/[_wind_, _energy_].csv")
            altair_sent_by_year_US_wind, altair_like_by_year_US_wind, layout_US_wind, data_slider_US_wind = plot_map(
                comment_dataframe_US_wind, country='US')
            return altair_sent_by_year_US_wind, altair_like_by_year_US_wind, layout_US_wind, data_slider_US_wind

    elif country == 'UK':
        if topic == 'Climate Change':
            comment_dataframe_UK_climate = pd.read_csv(
                "green_mood_tracker/raw_data/UK/[_climate_, _change_].csv")
            altair_sent_by_year_UK_climate, altair_like_by_year_UK_climate, figs_UK_climate = plot_map(
                comment_dataframe_UK_climate, country='UK')
            return altair_sent_by_year_UK_climate, altair_like_by_year_UK_climate, figs_UK_climate

        elif topic == 'Energy Prices':
            comment_dataframe_UK_prices = pd.read_csv(
                "green_mood_tracker/raw_data/UK/[_energy_, _prices_].csv")
            altair_sent_by_year_UK_prices, altair_like_by_year_UK_prices, figs_UK_prices = plot_map(
                comment_dataframe_UK_prices, country='UK')
            return altair_sent_by_year_UK_prices, altair_like_by_year_UK_prices, figs_UK_prices

        elif topic == 'Green Energy':
            comment_dataframe_UK_green = pd.read_csv(
                "green_mood_tracker/raw_data/UK/[_green_, _energy_].csv")
            altair_sent_by_year_UK_green, altair_like_by_year_UK_green, figs_UK_green = plot_map(
                comment_dataframe_UK_green, country='UK')
            return altair_sent_by_year_UK_green, altair_like_by_year_UK_green, figs_UK_green

        elif topic == 'Nuclear Energy':
            comment_dataframe_UK_nuclear = pd.read_csv(
                "green_mood_tracker/raw_data/UK/[_nuclear_, _energy_].csv")
            altair_sent_by_year_UK_nuclear, altair_like_by_year_UK_nuclear, figs_UK_nuclear = plot_map(
                comment_dataframe_UK_nuclear, country='UK')
            return altair_sent_by_year_UK_nuclear, altair_like_by_year_UK_nuclear, figs_UK_nuclear

        elif topic == 'Fossil Fuels':
            comment_dataframe_UK_fossil = pd.read_csv(
                "green_mood_tracker/raw_data/UK/[_fossil_, _fuels_].csv")
            altair_sent_by_year_UK_fossil, altair_like_by_year_UK_fossil, figs_UK_fossil = plot_map(
                comment_dataframe_UK_fossil, country='UK')
            return altair_sent_by_year_UK_fossil, altair_like_by_year_UK_fossil, figs_UK_fossil

        elif topic == 'Solar Energy':
            comment_dataframe_UK_solar = pd.read_csv(
                "green_mood_tracker/raw_data/UK/[_solar_, _energy_].csv")
            altair_sent_by_year_UK_solar, altair_like_by_year_UK_solar, layout_UK_solar, data_slider_UK_solar = plot_map(
                comment_dataframe_UK_solar, country='UK')
            return altair_sent_by_year_UK_solar, altair_like_by_year_UK_solar, figs_UK_solar

        elif topic == 'Wind Energy':
            comment_dataframe_UK_wind = pd.read_csv(
                "green_mood_tracker/raw_data/UK/[_wind_, _energy_].csv")
            altair_sent_by_year_UK_wind, altair_like_by_year_UK_wind, figs_UK_wind = plot_map(
                comment_dataframe_UK_wind, country='UK')
            return altair_sent_by_year_UK_wind, altair_like_by_year_UK_wind, figs_UK_wind


def get_twint_path(topic='Solar Energy', country='USA', time='(datetime.date(2010, 12, 1), datetime.date(2020, 12, 1))'):
    if country == 'USA':
        if topic == 'Climate Change':
            return "green_mood_tracker/raw_data/US/[_climate_, _change_].csv"

        elif topic == 'Energy Prices':
            return "green_mood_tracker/raw_data/US/[_energy_, _prices_].csv"

        elif topic == 'Green Energy':
            return "green_mood_tracker/raw_data/US/[_green_, _energy_].csv"

        elif topic == 'Nuclear Energy':
            return "green_mood_tracker/raw_data/US/[_nuclear_, _energy_].csv"

        elif topic == 'Fossil Fuels':
            return "green_mood_tracker/raw_data/US/[_fossil_, _fuels_].csv"

        elif topic == 'Solar Energy':
            return "green_mood_tracker/raw_data/US/[_solar_, _energy_].csv"

        elif topic == 'Wind Energy':
            return "green_mood_tracker/raw_data/US/[_wind_, _energy_].csv"

    elif country == 'UK':
        if topic == 'Climate Change':
            return "green_mood_tracker/raw_data/UK/[_climate_, _change_].csv"

        elif topic == 'Energy Prices':
            return "green_mood_tracker/raw_data/UK/[_energy_, _prices_].csv"

        elif topic == 'Green Energy':
            return "green_mood_tracker/raw_data/UK/[_green_, _energy_].csv"

        elif topic == 'Nuclear Energy':
            return "green_mood_tracker/raw_data/UK/[_nuclear_, _energy_].csv"

        elif topic == 'Fossil Fuels':
            return "green_mood_tracker/raw_data/UK/[_fossil_, _fuels_].csv"

        elif topic == 'Solar Energy':
            return "green_mood_tracker/raw_data/UK/[_solar_, _energy_].csv"

        elif topic == 'Wind Energy':
            return "green_mood_tracker/raw_data/UK/[_wind_, _energy_].csv"


def sl_predict(country_prediction, topic_prediction, date):

    st.write(type(date))
    st.write(date)

    filepath = 'green_mood_tracker/raw_data/twint_test/uktest.csv'

    get_twint_data(filepath, country='UK', topic='Solar Energy',
                   since=date[0], until=date[1])

    # pred = twint_prediction(filepath)
    # st.write(pred)

    return None


def main():
    analysis = st.sidebar.selectbox(
        "Select", ["Prediction", "Data Visualisation"])
    if analysis == 'Data Visualisation':
        st.header('Sentiment')
        year = st.slider('Year', min_value=2010, max_value=2020)
        country_prediction = st.selectbox('Select Country', ['UK', 'USA'], 1)
        topic_prediction = st.selectbox("Select Topic", [
                                        'Climate Change', 'Energy Prices', 'Fossil Fuels', 'Green Energy', 'Nuclear Energy', 'Solar Energy', 'Wind Energy'], 1)
        like_prediction = st.selectbox(
            'Sentiment factor', ['Per Tweet', 'Likes Per Tweet'], 1)
        st.text(" \n")
        st.text(" \n")
        st.text(" \n")
        st.text(" \n")
        st.markdown(
            f'**Percentage of Positive Sentiment Towards {topic_prediction} by State in the {country_prediction} since 2010**')

        # data = 'green_mood_tracker/raw_data/twint_US.csv'
        # df = pd.read_csv(data)
        # df['year']= pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S', errors= 'coerce').dt.year
        # df = df[df['year'] == year]
        altair_sent_by_year, altair_like_by_year, layout, data_slider = select_data(
            topic=topic_prediction, country=country_prediction)
        fig = go.Figure(data=data_slider[abs(year-2020)], layout=layout)
        colors = ['darkgreen', 'darkseagreen', 'darkolivegreen']

        if like_prediction == 'Per Tweet':
            c = altair_plot_tweet(altair_sent_by_year, year)
            fig_pie = px.pie(altair_sent_by_year[abs(year-2020)].groupby('sentiment').mean().reset_index(
            ), values='Percentage of Sentiment', names='sentiment', color_discrete_sequence=px.colors.sequential.YlGn)
            fig_pie.update_traces(hoverinfo='label+percent', textfont_size=12, textfont_color='#000000',
                                  marker=dict(colors=colors, line=dict(color='#000000', width=1.5)))
        elif like_prediction == 'Likes Per Tweet':
            c = altair_plot_like(altair_like_by_year, year)
            fig_pie = px.pie(altair_like_by_year[abs(year-2020)].groupby('sentiment').mean().reset_index(
            ), values='Percentage of Likes Per Sentiment', names='sentiment', color_discrete_sequence=px.colors.sequential.YlGn)
            fig_pie.update_traces(hoverinfo='label+percent', textfont_size=12, textfont_color='#000000',
                                  marker=dict(colors=colors, line=dict(color='#000000', width=1.5)))

        st.plotly_chart(fig, use_container_width=True)

        st.text(" \n")
        st.markdown('**Evolution of Sentiment Share Over Time**')
        st.altair_chart(c, use_container_width=True)

        st.text(" \n")
        st.markdown('**Share of Each Sentiment**')
        st.plotly_chart(fig_pie)

        # st.write(df['tweet'])

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
    main()
