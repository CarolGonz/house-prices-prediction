# base image
FROM python:3.9

# define working directory
WORKDIR /src

# copy requirements into the working directory
COPY requirements.txt requirements.txt

# upgrade pip
RUN python3 -m pip install --upgrade pip

# install requirements
RUN python3 -m pip install -r requirements.txt

# copy all other files into the working directory
COPY . .

# define default command for executing container
CMD python main.py

