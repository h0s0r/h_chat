FROM python:3.12-slim

WORKDIR /app

RUN pip install poetry

COPY pyproject.toml poetry.lock* ./

RUN poetry install --no-root --no-interaction --no-ansi

COPY h_chat .

EXPOSE 8501

CMD ["poetry","run","streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]