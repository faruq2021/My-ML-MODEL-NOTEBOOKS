# End to end Machine Learning Project with Deployment
The primary goal of this project is to deploy a Machine Learning solution using various different tools for deployment.

## About the project
This is a dataset about student performance in math, reading and writing based on several factors such as gender, race/ethnicity, parental level of education, access to lunch, access to a test preparatory course. For the sake of this project we will use the math scores as a target variable whose values need to be predicted and the rest of the fields will be treated as dependent variables. <br>
Since this is going to be a regression problem, we will use traditional Machine Learning algorithms like ```Linear Regression, Lasso, Ridge, K-Neighbors Regressor, Decision Tree Regressor, Random Forest Regressor, XGBRegressor, CatBoosting Regressor and Adaboost Regressor```. We will perform an analysis of the predictions of each of these algorithms and choose the one that gives us the best accuracy score after hyper-parameter tuning of the model during training.

## Deployed model
Before going further check out the model and its performance. I have implemented two different implementations of the deployment. One to Amazon Web Services using Elactic Beanstalk and another to Microsoft Azure cloud. Below are the links to both of them.<br>
Note: This is a simple HTML template with no extra features and beautications to keep the deployment simple.<br>
* [**Link to the AWS Deployment**](http://studentperformance-env.eba-vmsivpx6.us-east-1.elasticbeanstalk.com/predictdata) <br>
* [**Link to the Azure Cloud Deployment**](https://studentperformanceprediction.azurewebsites.net/predictdata)

Its usage is pretty simple, just enter any values into the fields and you can get the appropriate resulting score.

## Project Guidelines
1. Good code with a high degree of readability with comments
2. Well-structured code following software engineering principles
3. Modular code structure to replicate industry grade code
4. Deployment ready code for any platform


## Dependencies
1. pandas
2. numpy
3. seaborn
4. scikit-learn
5. catboost
6. xgboost
7. dill
8. Flask
9. Python 3.8
