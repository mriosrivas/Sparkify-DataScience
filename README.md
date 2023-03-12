# Sparkify-DataScience

Sparkify-DataScience is a project that aims to perform an ETL creation, an Exploratory Data Analysis, and models three machine learning classifiers using PySpark. The classifiers are a logistic regression, a random forest classifier, and a naive Bayes classifier. This project uses features such as time spent on the app and a likeliness classifier.

## Installation

To install Sparkify-DataScience, first ensure that you have the following dependencies installed:

- `numpy==1.23.5`
- `pandas==1.2.4`
- `pyspark==3.3.1`

Next, download the project files by running the following command in your terminal:

```bash
aws s3 sync s3://udacity-dsnd/sparkify/ .
```

#### Dashboard

To see a deployed dashboard for the Sparkify-DataScience project, please visit [https://mriosrivas-sparkify-dashboard-sparkify-crrui4.streamlit.app/](https://mriosrivas-sparkify-dashboard-sparkify-crrui4.streamlit.app/).

The code for the dashboard can be found in the [GitHub - mriosrivas/Sparkify-Dashboard: Sparkify&#39;s dashboard and prediction service](https://github.com/mriosrivas/Sparkify-Dashboard) repository.



## Project Analysis

### High-level overview

This project aims to work with large and realistic datasets using Spark, a distributed computing framework, and how to engineer relevant features for predicting customer churn. The project will cover how to use Spark MLlib to build and tune machine learning models with large datasets, which is not feasible with non-distributed 
technologies like scikit-learn. Predicting churn rates is a common and challenging problem that data scientists and analysts regularly encounter in customer-facing businesses, and being able to efficiently manipulate large datasets with Spark is a highly sought-after skill in the data field. The essential skills learned in this project 
include loading large datasets into Spark, manipulating them using Spark  SQL and Spark Dataframes, using machine learning APIs within Spark ML  to build and tune models.

### Description of Input Data

There are two datasets available for use in this project. The first dataset, called `sparkify_event_data.json`, is a large dataset that contains a significant amount of data, with a file size of 12.8 gigabytes (GB). The second dataset, `mini_sparkify_event_data.json`, is a smaller version of the same dataset and has a file size of 128.5 megabytes (MB).

To work with the larger dataset, the project suggests using the AWS EMR (Elastic MapReduce) platform. AWS EMR is a managed Hadoop framework that makes it easy to process large amounts of data using open-source tools like Apache Spark, Apache Hadoop, and Apache Hive. By leveraging the scalability of AWS EMR, data scientists can process and analyze large amounts of data without having to worry about managing the underlying infrastructure.



## Strategy for solving the problem

The following describes the strategy for solving the given problem:

1. Clean all the data and remove outliers: The first step is to clean the data and remove any irrelevant or redundant information. This includes handling missing or null values, dealing with outliers, and removing any features that are not useful for predicting churn.

2. With an exploratory data analysis remove unnecessary features: Next, an exploratory data analysis should be conducted to identify any patterns or relationships in the data. This will help to determine which features are important for predicting churn and which can be removed to simplify the model.

3. Train three base models: Once the data has been cleaned and the relevant features have been identified, three base models should be trained: logistic regression, random forest, and naive Bayes classifier. These models are chosen because they are widely used for classification problems and provide a good starting point for building more complex models.

4. Cross validate each model using AUC values: To evaluate the performance of each model, cross-validation should be performed using the area under the ROC curve (AUC) as the evaluation metric. This involves splitting the data into training and validation sets, fitting the model to the training set, and evaluating its performance on the validation set. This process is repeated multiple times with different training and validation sets to ensure that the results are robust.

5. Select the best model upon best metrics: Once all three models have been cross-validated and their AUC values have been calculated, the model with the best performance should be selected as the final model. This model can then be used to predict churn rates and identify which customers are most likely to leave.



### Discussion of the expected solution

The goal of this project is to develop a model that can predict customer churn based on a set of features from the dataset. This means that the model should be able to analyze the customer data and determine which customers are most likely to leave the service.

To accomplish this, the project will involve cleaning and processing the data, selecting relevant features, and training and evaluating several machine learning models to identify the best one. The end result will be a model that can be used to predict churn and help businesses retain their customers.

In addition to the churn prediction model, the project will also create a dashboard with aggregated data to enhance the user experience. By presenting information in a visually appealing and easy-to-use format, the dashboard will enable users to quickly and easily access the insights they need to make data-driven decisions and take action to reduce churn.

### EDA

The following steps were performed on the dataset:

#### Step 1: Remove Outliers

The first step in our EDA is to identify and remove any outliers from the data. Outliers are data points that are significantly different from other data points in the dataset and can have a significant impact on statistical analysis. To identify outliers, we can use various methods, such as box plots or scatter plots, and statistical techniques like Z-score analysis or interquartile range (IQR).

#### Step 2: Select Features using Kendall's Tau Correlation

After removing the outliers, we can select the features that are most relevant to our analysis. We will use Kendall's Tau correlation coefficient to identify the features that are most strongly correlated with the label variable. Kendall's Tau is a non-parametric measure of correlation that is useful when dealing with ordinal data or when the relationship between variables is not linear.

#### Step 3: Plot Selected Features

Finally, we will plot the selected features to further investigate their relationship with the label variable. Plotting the data can help us identify any patterns or trends that may exist within the data and can provide insights into the relationship between the features and the label variable. Storing Cleaned Data

For more detail on the EDA you can take a look at the [ETL notebook](ETL.ipynb).



### Data Preprocessing

The following is a list of steps performed for data preprocessing:

#### Step 1: Load the Data

The first step is to load the data into PySpark using the `spark.read.json()` function. We will load the data from the `sparkify_event_data.json` file and store it in a DataFrame called `df`.

#### Step 2: Data Cleaning

The next step is to clean the data. We will perform the following cleaning steps:

##### Remove Null Values

We will remove any rows that contain null values.

##### Select Users that had the 'paid' Level

We will select only the users who had a 'paid' level using the PySpark SQL functions. We will create a new DataFrame called `df_filter` that contains only the relevant rows.

#### Step 3: Create a Table of Occurrences

Next, we will create a table that counts the number of occurrences for the cleaned group using the PySpark SQL functions. We will create a new DataFrame called `data` that contains the counts.

#### Step 4: Convert Gender and Churn into Numbers

Then, we will convert the genders into a numeric form, where `male` will ve assgined a value of `1` and `female` a value of `0`. In the case of churning, a column named `label` will be created if a `submit_downgrade` is greater than `1`.

#### Step 5: Store the Data

Finally, we will store the data as a single CSV file in the `features/` folder.

More detailed information can be obtained in the [ETL.ipynb](ETL.ipynb) notebook.



### Modeling

The following is the procedure for modeling our classifiers:

#### Data Loading

The first step in this process is to load the clean_data from the EDA.ipynb notebook. This dataset should have been saved in a file format such as CSV, so it can be easily loaded into the current notebook using a data loading function or library.

#### Data Preparation

Before training the models, the data needs to be prepared by creating a VectorAssembler object. This object will take in all the features and merge them into a single vector. This is required by some of the machine learning models, including logistic regression and naive Bayes classifier.

After creating the VectorAssembler object, the data is split into training and test sets. The training data will be used to train the models, while the test data will be used to evaluate their performance.

#### Model Training

With the data prepared, the next step is to train the three machine learning models - logistic regression, random forest, and naive Bayes classifier. Each of these models will be trained using the training data set.

#### Model Comparison

Once the models have been trained, their ROC curves are compared. The ROC curve is a graphical representation of the performance of a classifier. A good classifier will have a curve that is close to the upper left corner of the plot. This indicates that the classifier has a high true positive rate and a low false positive rate.

After comparing the ROC curves, the confusion matrix is calculated for each model. The confusion matrix is a table that summarizes the performance of a classifier. It shows the number of true positives, false positives, true negatives, and false negatives.

More detailed information can be obtained in the [ML.ipynb](ML.ipynb) notebook.



### Hyperparameter Tuning

For hyperparameter tuning we perform the following:

#### Model Training

With the data prepared and engineered, the next step is to train the models. We use PySpark's Pipeline class to define a pipeline that includes the preprocessing, feature engineering, and model training steps. We train three different classifiers - logistic regression, random forest, and naive Bayes classifier. We also perform cross-validation to determine the best model.

#### Model Evaluation

After training the models, we evaluate their performance using metrics such as accuracy, precision, recall, and F1-score. In this case we use the ROC value. We also calculate the confusion matrix for each model.

#### Model Selection and Saving

Based on the evaluation results, we select the best model, which turns out to be the random forest classifier. We save this model for future use.

More detailed information can be obtained in the [ML-Pipeline.ipynb](ML-Pipeline.ipynb) notebook.

### Results

The best model for the churn prediction was the random forest classifier. In this case the model with `numTrees = 10` and `maxDepth = 10` performed best.



### Comparison Table

|     | precision | recall   | f1-score | model               |
| --- | --------- | -------- | -------- | ------------------- |
| 0   | 0.850932  | 0.521905 | 0.646989 | logistic regression |
| 1   | 0.951456  | 0.560000 | 0.705036 | random forest       |
| 2   | 0.562667  | 0.401905 | 0.468889 | naive bayes         |

## Conclusion

After training and cross-validating three different machine learning models - logistic regression, random forest, and naive Bayes classifier - the results indicate that the random forest classifier has the best performance, as measured by the AUC metric. This suggests that the random forest model is the most effective at predicting customer churn based on the features from the dataset.

This finding is important because it provides a clear recommendation for which model to use for predicting churn in this particular dataset. By selecting the random forest classifier, businesses can be confident that they are using a model that is likely to generate accurate predictions and help them retain their customers.

Additionally, the fact that the analysis was performed using Spark is significant because it demonstrates the power of this platform for handling and analyzing large amounts of data. The size of the dataset used in this project - 12.8 GB - is well beyond the capacity of many traditional data analysis tools, such as Excel or even R or Python. By leveraging Spark, it was possible to process and analyze this dataset in a scalable and efficient manner. This is a valuable capability for businesses that need to analyze large volumes of data, as it allows them to generate insights that might otherwise be inaccessible.



## Improvement

While the results of this project demonstrate that the random forest classifier is currently the most effective model for predicting customer churn in this particular dataset, there are always opportunities for further development and improvement. One possible avenue for future work would be to explore the use of other machine learning models, such as XGBoost, and evaluate their performance on similar datasets.

XGBoost is a powerful and popular machine learning algorithm that is commonly used for predictive modeling tasks, and it has been shown to outperform other models in a variety of contexts. By testing XGBoost on future projects, it may be possible to further improve the accuracy and reliability of churn prediction models, which could have important implications for businesses that rely on customer retention.

In addition to exploring new models, there may also be opportunities to refine the existing models by tweaking their hyperparameters or using more advanced feature engineering techniques. By continuing to iterate and experiment with different approaches, it may be possible to further optimize the performance of churn prediction models and achieve even better results in the future.

## Acknowledgment

This project was completed as part of the Udacity Data Scientist Nanodegree program. The dataset was provided by Udacity.

## License

Sparkify-DataScience is licensed under the MIT License. See the LICENSE file for more information.
