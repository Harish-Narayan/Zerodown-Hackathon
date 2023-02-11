from flask import Flask, Response, render_template,request
from sqlalchemy import create_engine
import seaborn as sns
import matplotlib.pyplot as plt
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
    return render_template('index.html')





@app.route('/process-form', methods=['POST'])
def process_form():
    print("hi")
    input_value = request.form['id']
    # Do something with the input value
    print(input_value)
    download_data(input_value)
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
        return score

    # Apply the scoring function to each demand in the dataset
    df2['score'] = df2.apply(score_demand, axis=1)

    # Sort the demands by their score in descending order
    df2 = df2.sort_values(by='score', ascending=False)
    df2['score'] = (df2['score'] - df2['score'].min()) / (df2['score'].max() - df2['score'].min())
    market_score=df2['score'].mean()
    print(market_score)
    # Print the top-scoring demands
    print(df2.head())
    heatmap_data = df2.pivot(index='sold_homes_count', columns='median_sale_price', values='score')

    # Create the heatmap
    sns.set(font_scale=1.2)
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu")

    # Set the axis labels
    plt.xlabel('Median Sale Price')
    plt.ylabel('Sold Homes Count')

    # Display the heatmap
    plt.show()
    csv2=df2.to_csv(index=False)
    df2.to_csv('final.csv',index=False)

    rows = [row.split(',') for row in csv2.split('\n')]
    
    return render_template('display_input.html', input_int=market_score)







def download_data(id):
    # Connect to the PostgreSQL database using SQLAlchemy
    engine = create_engine("postgresql://postgres:admin@localhost/question2")
    fin_id = id  # the user ID you want to select
    fin_id=int(fin_id)
    df = pd.read_sql("SELECT * FROM market_metrics WHERE market_id = %d  limit 10;" % fin_id, engine)
    # Save the DataFrame to a CSV file and get the CSV data as a string
    df.to_csv('new_file.csv',index=False)

    # Close the database connection
    engine.dispose()

    # Return the CSV data as a string
    return 
    '''
@app.route('/input/<int:input_int>')
def display_input(input_int):
    return render_template('display_input.html', input_int=input_int)
    '''
if __name__ == '__main__':
    app.run()

