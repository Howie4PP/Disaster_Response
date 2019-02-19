# Disaster Response Project

## Table of contents
1. [Installations.](#install)       
2. [Project Motivation.](#proj)      
3. [File Descriptions.](#file)
4. [Command.](#Command)   
5. [ScreenShot](#screen)
6. [Acknowledgements.](#author)    

<a name="install"></a>
## Installations
The code was written in Python, using Jupyter Notebook. Coding Environment Controlled by Anaconda, Default Version of Python is 3.6. The Liberaries will be used:
- Pandas
- Numpy
- plotly
- nltk
- nltk.stem.WordNetLemmatizer
- nltk.tokenize.word_tokenize
- Flask
- sklearn
- sqlalchemy
- pickle

<a name="proj"></a>
## Project Motivation
This project targets to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. There are three main steps to complete the whole project, which is:
1. ETL Pipeline
2. ML Pipeline
3. Flask Web App

Each step will address a technical problem and will be covered in detail in the next section. The basic concept of the project is extract data from csv file, then cleaning and storing into database. A model will be traind by the data loads from these database. Obvisouly, there is necessary to test and improve model. The different algorithms of Machine Learning will be used. Next, the model that be trained will be used in Web-app(Flask).

<a name="file"></a>
## File Descriptions
There are 3 folders in the project, which corresponds to 3 different steps(funtions) respectively.
1. data -> ETL pipeline: 
   - Load data from csv file
   - Clean data
   - Stores it in a SQLite database
2. models -> ML Pipeline:
   - Loads data from the SQLite database
   - Splits the dataset into training and test sets
   - Builds a text processing and machine learning pipeline
   - Trains and tunes a model using GridSearchCV
   - Outputs results on the test set
   - Exports the final model as a pickle file
3. app -> Flask Web App
   - Load database and model
   - Data visualizations using Plotly in the web app

<a name="Command"></a>
## Command
Run process_data.py         
The following command: <code>python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db</code>

Run train_classifier.py        
The following command: <code>python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl</code>

Run the web app        
Run the following command: <code>python run.py</code>          
Go to http://0.0.0.0:3001/

<a name="screen"></a>
## ScreenShot
This is the complete web-app
<img src="https://github.com/Howie4PP/Disaster_Response/blob/master/Screenshot%202019-02-19%20at%203.14.15%20PM.png">   

This is the result after entering one message, the categories which the message belongs to highlighted in green.
<img src="https://github.com/Howie4PP/Disaster_Response/blob/master/Screenshot%202019-02-19%20at%203.13.30%20PM.png">   

<a name="author"></a>
## Acknowledgements
Framework and knowledges of project was provided by <a href='https://www.udacity.com/'>Udacity.</a>       
DataSet from <a href='https://www.figure-eight.com/'> Figure eight</a>
