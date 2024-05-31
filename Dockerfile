FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml env_requirements.txt ./

RUN pip install --no-cache-dir -r env_requirements.txt

COPY . .

RUN poetry install --no-dev

ENTRYPOINT ["poetry", "run", "easydel"]