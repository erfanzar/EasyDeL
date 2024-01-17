FROM python:3.11

WORKDIR /app

COPY poetry.lock pyproject.toml ./

RUN curl -sSL https://install.python-poetry.org | python -

RUN poetry install --no-dev

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]