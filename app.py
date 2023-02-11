from flask import Flask, Response, render_template
from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
app = Flask(__name__)

@app.route('/')
def index():
    # Call the download_data function and get the CSV data
    
    download_data()
    print("hi")
    # Load the CSV dataset into a Pandas dataframe
    df = pd.read_csv('new_file.csv')

    # Define a list of the required features
    attributes = ['sold_homes_count', 'median_sale_to_list_ratio', 'homes_sold_over_list_price_count','median_sale_price', 'days_to_pending', 'days_to_sell']
    

    # Create a new dataframe with only the required features
    df2 = df[attributes]

    # Save the new dataframe to a new CSV file
    df2.to_csv('my_dataset_required.csv', index=False)
    # Define the attributes and their weights
    df2= pd.read_csv('my_dataset_required.csv')
    weights = [0.5, 0.6, 0.4, 0.7, 0.3,0.5]

    # Define the rating scale
    rating_scale = range(1, 6)

    # Create a function to score each demand based on the decision matrix approach
    def score_demand(demand_row):
        score = 0
        for i in range(len(attributes)):
            attribute = attributes[i]
            rating = demand_row[attribute]
            weight = weights[i]
            score += rating * weight
            print(score)
        return score

    # Apply the scoring function to each demand in the dataset
    df2['score'] = df2.apply(score_demand, axis=1)

    # Sort the demands by their score in descending order
    df2 = df2.sort_values(by='score', ascending=False)
    df2['score'] = (df2['score'] - df2['score'].min()) / (df2['score'].max() - df2['score'].min())
    # Print the top-scoring demands
    print(df2.head())
    
    csv2=df2.to_csv(index=False)
    df2.to_csv('final.csv',index=False)
    rows = [row.split(',') for row in csv2.split('\n')]
    return render_template('index.html', csv2=rows)







def download_data():
    # Connect to the PostgreSQL database using SQLAlchemy
    engine = create_engine("postgresql://postgres:admin@localhost/question2")
    print("hi")
    # Query the data and store it in a DataFrame
    df = pd.read_sql("SELECT * FROM market_metrics limit 100", engine)
    print("hi")
    # Save the DataFrame to a CSV file and get the CSV data as a string
    df.to_csv('new_file.csv',index=False)

    # Close the database connection
    engine.dispose()

    # Return the CSV data as a string
    return 

if __name__ == '__main__':
    app.run()

