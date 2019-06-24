# Disaster-Reponse

This project is aimed to classify the reallife disaster tweets into pre defined categories

This project is divided in the following sections
1.Data Processing: ETL Pipeline to extract data from source, clean data and save them in a proper databse structure
2. Machine Learning Pipeline to train a model able to classify text message in categories
3. Web App to show model results in real time.

# Dependencies
1) Python 3.5+ (I used Python 3.7)
2) Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
3) Natural Language Process Libraries: NLTK
4) SQLlite Database Libraqries: SQLalchemy
5) Web App and Data Visualization: Flask, Plotly

## Exectuing the program

1) Run the following commands in the project's root directory to set up your database and model.
 a) To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
 b) To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
2) Run the following command in the app's directory to run your web app. python run.py
3) Go to http://0.0.0.0:3001/

## Screenshots

This has been given seperately in the screenshot directory of the file
