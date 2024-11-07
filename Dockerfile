FROM python:3.12.4-slim

WORKDIR /usr/src/app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# ENTRYPOINT ["python", "-m", "gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker"]
# ENTRYPOINT ["uvicorn", "main:app"]
CMD ["python", "main.py"]