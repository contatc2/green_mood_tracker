import pandas as pd
import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from green_mood_tracker.predict import twint_prediction
from green_mood_tracker.data import get_twint_data

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict_sentiment/?country=UK&topic=Green Energy&start_date=2000-10-11 12:00:00&end_date=2000-12-11 12:00:00


@app.get("/")
def index():
    return {"ok": "True"}


@app.get("/predict_sentiment/")
def predict_sentiment(country,
                topic
                # start_date,
                # end_date
                ):

    filepath = f'twint_test/{country}-data-test.csv'
    date = [datetime.date(
        2020, 11, 1), datetime.date(2020, 11, 30)]
    start_date, end_date = date

    get_twint_data(filepath, country=country,
                   topic=topic, since=start_date, until=end_date)

    pred = twint_prediction(filepath, encode=True)
    print(pred)

    return pred.to_json()
