FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN poetry install --no-dev

ENTRYPOINT ["poetry", "run", "easydel"]