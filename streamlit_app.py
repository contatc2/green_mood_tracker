
import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from green_mood_tracker.datavisstreamlit import altair_plot, plot_map
from green_mood_tracker.clustering import lda_wordcloud
from green_mood_tracker.predict import twint_prediction
from green_mood_tracker.data import get_twint_data


st.set_page_config(layout='wide')

img = st.image('green_mood_tracker/assets/green_mood_tracker_logo.png',
                width=700, output_format='png')

st.markdown("**Energy Sentiment Analysis**")


def sl_predict(country_prediction, topic_prediction, date,local=False):

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
    topics = topic.lower().split()
    country_code = 'US' if country == 'USA' else country
    comment_dataframe = pd.read_csv(
        f"green_mood_tracker_app/raw_data/{country_code}/[_{topics[0]}_, _{topics[1]}_].csv"
    )
    altair_sent_by_year, altair_like_by_year, layout, data_slider = plot_map(
        comment_dataframe, country=country_code, like_prediction=like_prediction)
    return altair_sent_by_year, altair_like_by_year, layout, data_slider


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
        chart_desc_suffix = '' if like_prediction == 'Per Tweet' else ' Based on Likes'

        st.markdown(
            f'**Tweet Sentiment Rating Towards {topic_prediction} in the {country_prediction} in {year}{chart_desc_suffix}**'
        )

        altair_sent_by_year, altair_like_by_year, layout, data_slider = select_data(
            topic=topic_prediction, country=country_prediction, like_prediction=like_prediction)
        fig = go.Figure(data=data_slider[abs(year-2020)], layout=layout)
        fig.update_layout(width=1500, height=500)

        data = altair_sent_by_year if like_prediction == 'Per Tweet' else altair_like_by_year
        chart_values = 'Percentage of Sentiment' if like_prediction == 'Per Tweet' else 'Percentage of Likes Per Sentiment'
        c = altair_plot(data, year, like_prediction, chart_values)
        fig_pie = px.pie(data[abs(year-2020)].groupby('sentiment').mean().reset_index(
        ), values=chart_values, names='sentiment', color_discrete_sequence=px.colors.sequential.YlGn)

        fig_pie.update_traces(hoverinfo='label+percent', textfont_size=12, textfont_color='#000000',
                              marker=dict(colors=['#cc2936', '#FFA500', '#00B050'], line=dict(color='#EBECF0', width=1.5)))

        c.properties(width=1000)

        if country_prediction == 'UK':
            fig.update_geos(fitbounds="locations", visible=False)

        fig_pie.update_layout(width=500, height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.text(" \n")
        st.markdown(
            f'**Evolution of Sentiment Share Towards {topic_prediction} in the {country_prediction} in {year}{chart_desc_suffix}**'
        )
        st.altair_chart(c, use_container_width=True)

        st.text(" \n")
        st.markdown(
            f'**Share of Sentiment Towards {topic_prediction} in the {country_prediction} in {year}{chart_desc_suffix}**'
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    if analysis == "Live Analysis":
        st.header("Green Mood Tracker Model Predictions")
        # inputs from user
        country_prediction = st.selectbox("Select Country", ['USA', 'UK'], 1)
        topic_prediction = st.selectbox("Select Topic", [
            'Climate Change', 'Energy Prices', 'Fossil Fuels', 'Green Energy', 'Nuclear Energy', 'Solar Energy', 'Wind Energy'], 1)
        d3 = st.date_input("Select TimeFrame", [datetime.date(
            2020, 11, 1), datetime.date(2020, 11, 30)])
        sl_predict(country_prediction, topic_prediction, d3, local=True)


# proc.test_execute()
if __name__ == "__main__":
    main()
