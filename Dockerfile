FROM python:3.7-slim

# remember to expose the port your app'll be exposed on.
EXPOSE 8080
RUN apt-get update -y && apt-get install -y gcc
RUN pip install -U pip

COPY requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt

# copy into a directory of its own (so it isn't in the toplevel dir)
COPY . /app
WORKDIR /app

# run it!
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.enableCORS=false"]