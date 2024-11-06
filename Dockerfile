FROM python:3.12.4-slim

WORKDIR /usr/src/app
COPY . .
RUN pip install -r requirements.txt

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

ENTRYPOINT ["python", "main.py"]
