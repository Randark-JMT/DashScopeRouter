FROM python:3.12.12

RUN mkdir /app
WORKDIR /app

COPY . /app/
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python", "main.py"]
