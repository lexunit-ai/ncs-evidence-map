FROM python:3.8-slim

WORKDIR /app

COPY . .

RUN python -m pip install -r requirements.txt

RUN python -m spacy download en_core_web_sm

EXPOSE 8000

CMD ["uvicorn", "app:app",  "--host", "0.0.0.0"]
