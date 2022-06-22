FROM python:3.8.2
WORKDIR /usr/src/project_guillermo
COPY . .
RUN apt-get update
RUN pip install -r requirements.txt
ENTRYPOINT ["python3"]

