## Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee
### DATE:19.03.2024
### AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

### Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

### Algorithm
1. Import the standard libraries
2. Upload the dataset and check for any null values using .isnull() function
3. Import LabelEncoder and encode the dataset
4. Import DecisionTreeRegressor from sklearn and and apply the model on dataset
5. Predict the values of arrays
6. Import metrices from sklearn and calculate the MSE and R2 of the model on the dataset
7. Predict the values of array
8. 8.Apply to new unknown values

### Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Lathika Sunder
RegisterNumber: 212221230054 
```
```python
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Salary_EX7.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import  DecisionTreeRegressor,plot_tree
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()
```
### Output:
### HEAD VALUES:
![image](https://github.com/gpavana/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118787343/cc9e8bea-cd57-4744-9433-5396a59cde3e)
### INFO:
![image](https://github.com/gpavana/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118787343/39067b5b-a7c7-4b0c-8fdb-7acaba347fc0)
### CONVERTING CATEGORICAL VALUES INTO NUMERICAL VALUES:
![image](https://github.com/gpavana/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118787343/5491d6db-e4a1-44e9-9685-bf45f0653cd7)
### MEAN SQUARED ERROR:
![image](https://github.com/gpavana/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118787343/34c8a444-7e07-4a42-9bd2-ec59404af529)
### PREDICTED VALUE:
![image](https://github.com/gpavana/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118787343/f8593cd6-d674-457d-becd-ce0e5b4b0cd4)
### DECISION TREE:
![image](https://github.com/gpavana/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118787343/21554b8b-5706-42b7-a4b1-dc0d7c0bdbf1)
### Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
