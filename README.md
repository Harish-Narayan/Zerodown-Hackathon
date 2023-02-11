# Zerodown-Hackathon
# E-R Diagram for first problem
![image](https://user-images.githubusercontent.com/75531922/218244229-737a9c18-0886-4b98-9bdc-2acb8e0549bf.png)
# Second question of hackathon
### 1.Database created in postgres
### 2.Use the decision matrix to predict the score required
### Decision matrix is used to systematically identify the relationships between the attributes
### By assigning weights to all the items, we try to give out a score that gives a score between 0 and 1
### Algorithm for the decision matrix
#### df = pd.read_csv('my_data.csv')
#### attributes = ['attribute1', 'attribute2', 'attribute3']
#### df_selected = df[attributes]
#### weights = [0.5, 0.3, 0.2]
#### df_normalized = (df_selected - df_selected.min()) / (df_selected.max() - df_selected.min())
#### df_normalized['score'] = (df_normalized * weights).sum(axis=1)
#### df_sorted = df_normalized.sort_values(by='score', ascending=False)
#### df_sorted.to_csv('my_data_sorted.csv', index=False)

### 3.Created a simple web app that takes a market id and displays the score calculated.

### 4.To prove the accuracy of the score, a heatwave map is generated
### A Heatwave map is used to visualize an attribute in comparison to two other attributes
### Seaborn library in python is used for visualization
### Number of houses sold and average price was used for visualization
### We can compare the selling price, number of houses sold and score and verify the geniuneness of the score
