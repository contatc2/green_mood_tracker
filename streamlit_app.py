
import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from green_mood_tracker.datavisstreamlit import altair_plot_like, altair_plot_tweet
from green_mood_tracker.datavisstreamlit import plot_map
from green_mood_tracker.clustering import lda_wordcloud
from green_mood_tracker.predict import twint_prediction
from green_mood_tracker.data import get_twint_data


st.set_page_config(layout='wide')

img = st.image('green_mood_tracker/assets/green_mood_tracker_logo.png',
               style='left', width=700, output_format='png')

st.markdown("**Energy Sentiment Analysis**")


def sl_predict(country_prediction, topic_prediction, date):

    filepath = 'twint_test/uk-data-test.csv'

    get_twint_data(filepath, country=country_prediction,
                   topic=topic_prediction, since=date[0], until=date[1])

    pred = twint_prediction(filepath, encode=True)
    class_1 = pred['polarity'].mean()
    class_2 = 1 - class_1

    labels = 'Positive', 'Negative'
    sizes = [class_1, class_2]

    fig_1 = px.pie(values=sizes, names=labels, color_discrete_sequence=px.colors.sequential.YlGn)
    fig_1.update_traces(hoverinfo='label+percent', textfont_size=12, textfont_color='#000000',
                                  marker=dict(colors=['#00B050', '#cc2936'], line=dict(color='#EBECF0', width=1.5)))

    for k in range(5):
        st.markdown(
            f"<h6 style='margin-bottom':0!important;> <u>user0{k}@london</u></h6>", unsafe_allow_html=True)
        if pred.iloc[k]['polarity']:
            st.markdown(
                f"<p style=color:#00B050;> '<em>{pred.iloc[k]['tweet']}</em>'</p>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"<p style=color:#cc2936;> '<em>{pred.iloc[k]['tweet']}</em>'</p>", unsafe_allow_html=True)

    st.plotly_chart(fig_1, use_container_width=True)

    wc = lda_wordcloud(pred, 'tweet')

    st.pyplot(wc)


@st.cache
def select_data(topic='Solar Energy', country='USA', like_prediction='Per Tweet'):
    if country == 'USA':
        if topic == 'Climate Change':
            comment_dataframe_US_climate = pd.read_csv(
                "green_mood_tracker/raw_data/US/[_climate_, _change_].csv")
            altair_sent_by_year_US_climate, altair_like_by_year_US_climate, layout_US_climate, data_slider_US_climate = plot_map(
                comment_dataframe_US_climate, country='US', like_prediction=like_prediction)
            return altair_sent_by_year_US_climate, altair_like_by_year_US_climate, layout_US_climate, data_slider_US_climate

        elif topic == 'Energy Prices':
            comment_dataframe_US_prices = pd.read_csv(
                "green_mood_tracker/raw_data/US/[_energy_, _prices_].csv")
            altair_sent_by_year_US_prices, altair_like_by_year_US_prices, layout_US_prices, data_slider_US_prices = plot_map(
                comment_dataframe_US_prices, country='US', like_prediction=like_prediction)
            return altair_sent_by_year_US_prices, altair_like_by_year_US_prices, layout_US_prices, data_slider_US_prices

        elif topic == 'Green Energy':
            comment_dataframe_US_green = pd.read_csv(
                "green_mood_tracker/raw_data/US/[_green_, _energy_].csv")
            altair_sent_by_year_US_green, altair_like_by_year_US_green, layout_US_green, data_slider_US_green = plot_map(
                comment_dataframe_US_green, country='US', like_prediction=like_prediction)
            return altair_sent_by_year_US_green, altair_like_by_year_US_green, layout_US_green, data_slider_US_green

        elif topic == 'Nuclear Energy':
            comment_dataframe_US_nuclear = pd.read_csv(
                "green_mood_tracker/raw_data/US/[_nuclear_, _energy_].csv")
            altair_sent_by_year_US_nuclear, altair_like_by_year_US_nuclear, layout_US_nuclear, data_slider_US_nuclear = plot_map(
                comment_dataframe_US_nuclear, country='US', like_prediction=like_prediction)
            return altair_sent_by_year_US_nuclear, altair_like_by_year_US_nuclear, layout_US_nuclear, data_slider_US_nuclear

        elif topic == 'Fossil Fuels':
            comment_dataframe_US_fossil = pd.read_csv(
                "green_mood_tracker/raw_data/US/[_fossil_, _fuels_].csv")
            altair_sent_by_year_US_fossil, altair_like_by_year_US_fossil, layout_US_fossil, data_slider_US_fossil = plot_map(
                comment_dataframe_US_fossil, country='US', like_prediction=like_prediction)
            return altair_sent_by_year_US_fossil, altair_like_by_year_US_fossil, layout_US_fossil, data_slider_US_fossil

        elif topic == 'Solar Energy':
            comment_dataframe_US_solar = pd.read_csv(
                "green_mood_tracker/raw_data/US/[_solar_, _energy_].csv")
            altair_sent_by_year_US_solar, altair_like_by_year_US_solar, layout_US_solar, data_slider_US_solar = plot_map(
                comment_dataframe_US_solar, country='US', like_prediction=like_prediction)
            return altair_sent_by_year_US_solar, altair_like_by_year_US_solar, layout_US_solar, data_slider_US_solar

        elif topic == 'Wind Energy':
            comment_dataframe_US_wind = pd.read_csv(
                "green_mood_tracker/raw_data/US/[_wind_, _energy_].csv")
            altair_sent_by_year_US_wind, altair_like_by_year_US_wind, layout_US_wind, data_slider_US_wind = plot_map(
                comment_dataframe_US_wind, country='US', like_prediction=like_prediction)
            return altair_sent_by_year_US_wind, altair_like_by_year_US_wind, layout_US_wind, data_slider_US_wind

    elif country == 'UK':
        if topic == 'Climate Change':
            comment_dataframe_UK_climate = pd.read_csv(
                "green_mood_tracker/raw_data/UK/[_climate_, _change_].csv")
            altair_sent_by_year_UK_climate, altair_like_by_year_UK_climate, layout, data_slider = plot_map(
                comment_dataframe_UK_climate, country='UK', like_prediction=like_prediction)
            return altair_sent_by_year_UK_climate, altair_like_by_year_UK_climate, layout, data_slider

        elif topic == 'Energy Prices':
            comment_dataframe_UK_prices = pd.read_csv(
                "green_mood_tracker/raw_data/UK/[_energy_, _prices_].csv")
            altair_sent_by_year_UK_prices, altair_like_by_year_UK_prices, layout, data_slider = plot_map(
                comment_dataframe_UK_prices, country='UK', like_prediction=like_prediction)
            return altair_sent_by_year_UK_prices, altair_like_by_year_UK_prices, layout, data_slider

        elif topic == 'Green Energy':
            comment_dataframe_UK_green = pd.read_csv(
                "green_mood_tracker/raw_data/UK/[_green_, _energy_].csv")
            altair_sent_by_year_UK_green, altair_like_by_year_UK_green, layout, data_slider = plot_map(
                comment_dataframe_UK_green, country='UK', like_prediction=like_prediction)
            return altair_sent_by_year_UK_green, altair_like_by_year_UK_green, layout, data_slider

        elif topic == 'Nuclear Energy':
            comment_dataframe_UK_nuclear = pd.read_csv(
                "green_mood_tracker/raw_data/UK/[_nuclear_, _energy_].csv")
            altair_sent_by_year_UK_nuclear, altair_like_by_year_UK_nuclear, layout, data_slider = plot_map(
                comment_dataframe_UK_nuclear, country='UK', like_prediction=like_prediction)
            return altair_sent_by_year_UK_nuclear, altair_like_by_year_UK_nuclear, layout, data_slider

        elif topic == 'Fossil Fuels':
            comment_dataframe_UK_fossil = pd.read_csv(
                "green_mood_tracker/raw_data/UK/[_fossil_, _fuels_].csv")
            altair_sent_by_year_UK_fossil, altair_like_by_year_UK_fossil, layout, data_slider = plot_map(
                comment_dataframe_UK_fossil, country='UK', like_prediction=like_prediction)
            return altair_sent_by_year_UK_fossil, altair_like_by_year_UK_fossil, layout, data_slider

        elif topic == 'Solar Energy':
            comment_dataframe_UK_solar = pd.read_csv(
                "green_mood_tracker/raw_data/UK/[_solar_, _energy_].csv")
            altair_sent_by_year_UK_solar, altair_like_by_year_UK_solar, layout, data_slider = plot_map(
                comment_dataframe_UK_solar, country='UK', like_prediction=like_prediction)
            return altair_sent_by_year_UK_solar, altair_like_by_year_UK_solar, layout, data_slider

        elif topic == 'Wind Energy':
            comment_dataframe_UK_wind = pd.read_csv(
                "green_mood_tracker/raw_data/UK/[_wind_, _energy_].csv")
            altair_sent_by_year_UK_wind, altair_like_by_year_UK_wind, layout, data_slider = plot_map(
                comment_dataframe_UK_wind, country='UK', like_prediction=like_prediction)
            return altair_sent_by_year_UK_wind, altair_like_by_year_UK_wind, layout, data_slider


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


def main():
    analysis = st.sidebar.selectbox(
        "Select", ["Data Visualisation", "Live Analysis"])
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
        if like_prediction == 'Per Tweet':
            st.markdown(
                f'**Tweet Sentiment Rating Towards {topic_prediction} in the {country_prediction} in  {year}**')
        elif like_prediction == 'Likes Per Tweet':
            st.markdown(
                f'**Tweet Sentiment Rating Towards {topic_prediction} in the {country_prediction} in  {year} Based on Likes **')

            #data = 'green_mood_tracker/raw_data/twint_US.csv'
            #df = pd.read_csv(data)
            #df['year']= pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S', errors= 'coerce').dt.year
            #df = df[df['year'] == year]

        altair_sent_by_year, altair_like_by_year, layout, data_slider = select_data(
            topic=topic_prediction, country=country_prediction, like_prediction=like_prediction)
        fig = go.Figure(data=data_slider[abs(year-2020)], layout=layout)
        fig.update_layout(width=1500, height=500)

        if like_prediction == 'Per Tweet':
            c = altair_plot_tweet(altair_sent_by_year, year)
            fig_pie = px.pie(altair_sent_by_year[abs(year-2020)].groupby('sentiment').mean().reset_index(
            ), values='Percentage of Sentiment', names='sentiment', color_discrete_sequence=px.colors.sequential.YlGn)
            fig_pie.update_traces(hoverinfo='label+percent', textfont_size=12, textfont_color='#000000',
                                  marker=dict(colors=['#cc2936', '#FFA500', '#00B050'], line=dict(color='#EBECF0', width=1.5)))
        elif like_prediction == 'Likes Per Tweet':
            c = altair_plot_like(altair_like_by_year, year)
            fig_pie = px.pie(altair_like_by_year[abs(year-2020)].groupby('sentiment').mean().reset_index(
            ), values='Percentage of Likes Per Sentiment', names='sentiment', color_discrete_sequence=px.colors.sequential.YlGn)
            fig_pie.update_traces(hoverinfo='label+percent', textfont_size=12, textfont_color='#000000',
                                  marker=dict(colors=['#cc2936', '#FFA500', '#00B050'], line=dict(color='#EBECF0', width=1.5)))

        c.properties(width=1000)

        if country_prediction == 'UK':
            fig.update_geos(fitbounds="locations", visible=False)

        fig_pie.update_layout(width=500, height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.text(" \n")
        if like_prediction == 'Per Tweet':
	        st.markdown(
	            f'**Evolution of Sentiment Share Towards {topic_prediction} in the {country_prediction} in {year}**')
        if like_prediction == 'Likes Per Tweet':
        	st.markdown(
            f'**Evolution of Sentiment Share Towards {topic_prediction} in the {country_prediction} in {year} Based on Likes**')
        st.altair_chart(c, use_container_width=True)

        st.text(" \n")
        if like_prediction == 'Per Tweet':
	        st.markdown(
	        	f'**Total Share of Likes Per Sentiment Towards {topic_prediction} in the {country_prediction} in {year}**')
        if like_prediction == 'Likes Per Tweet':
        	st.markdown(
            f'**Total Share of Likes Per Sentiment Towards {topic_prediction} in the {country_prediction} in {year}**')
        st.plotly_chart(fig_pie, use_container_width=True)

    if analysis == "Live Analysis":
        st.header("Green Mood Tracker Model Predictions")
        # inputs from user
        country_prediction = st.selectbox("Select Country", ['USA', 'UK'], 1)
        topic_prediction = st.selectbox("Select Topic", [
            'Climate Change', 'Energy Prices', 'Fossil Fuels', 'Green Energy', 'Nuclear Energy', 'Solar Energy', 'Wind Energy'], 1)
        d3 = st.date_input("Select TimeFrame", [datetime.date(2020, 11, 1), datetime.date(2020,11, 30)])
        sl_predict(country_prediction, topic_prediction, d3)


# proc.test_execute()
if __name__ == "__main__":
    main()
