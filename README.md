# house-prices-prediction

The goal of this repository is to build a Docker container for training and calling a linear regression model based on the housing 
prices dataset to predict respective house prices.


## How to run the image locally

1. Clone this repository and build the container locally: 

   - git clone https://github.com/CarolGonz/house-prices-prediction.git
     - checkout to the "main" branch
   - docker build -t <image-name> .
   - docker run <container-id>


2. Download the image hosted on Docker Hub:

   - docker pull carolgonz/house-prince-predictions:latest
   - docker run carolgonz/house-prince-predictions:latest


## Description

The model load the data from the sklearn api and return a dictionary of prices basic stats and the r2 score of the baseline model.