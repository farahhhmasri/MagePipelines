## Overview
### This Mage project has two pipelines:
#### 1. Training pipeline called *"training_models"*, it'll train multiple models on the diamonds dataset and keep track of the best model. The user can specify which models to train and if they'd like to export the cleaned data to an external postgres database.
#### 2. Inferencing pipeline called *"inference"*, it's called by an API request. It uses the best model that was provided from the training pipeline, loads raw data from postgres database and return predictions. An example of how to create an API request to run it is provided in the repo.
#### *Inside each pipeline there's a guidence cell that shows details regarding how each one works.*
</br>

### To run the project, you need to pull the repo and run this:
##### docker run -it -p 6789:6789 -v "$(pwd):/home/src" -e USER_CODE_PATH=/home/src/[training] mageai/mageai /app/run_app.sh mage start [training]
