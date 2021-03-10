FROM python:3.8.8-buster

COPY api /api
COPY green_mood_tracker /green_mood_tracker
COPY models /models
COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt
RUN pip3 install --upgrade git+https://github.com/twintproject/twint.git@origin/master#egg=twint

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
