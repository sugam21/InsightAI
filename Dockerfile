FROM python:3.12.4-slim

RUN pip install --no-cache-dir poetry==1.8.4

ENV POETRY_NO_INTERACTION=1 \
  POETRY_VIRTUALENVS_IN_PROJECT=1 \
  POETRY_VIRTUALENVS_CREATE=1 \
  POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /usr/src/app

COPY pyproject.toml poetry.lock ./

RUN poetry install --without dev && rm -rf $POETRY_CACHE_DIR

COPY . .

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

ENTRYPOINT ["poetry", "run", "python", "-m", "src.app.gradio_app"]
