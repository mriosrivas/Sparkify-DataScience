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

## License

Sparkify-DataScience is licensed under the MIT License. See the LICENSE file for more information.

## Credits

This project was completed as part of the Udacity Data Scientist Nanodegree program. The dataset was provided by Udacity.
