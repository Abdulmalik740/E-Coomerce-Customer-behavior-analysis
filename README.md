# E-Commerce Customer Behavior Analysis

This project explores customer behavior based on an e-commerce dataset. It involves data cleaning, visualization, and analysis to gain insights into customer satisfaction, spending habits, and purchasing trends.

## 📁 Files

- `E_commerce_behavior_analysis.py`: The main Python script that loads the dataset, processes the data, performs exploratory data analysis (EDA), and visualizes the results.
- `E-commerce Customer Behavior.csv`: The dataset used in the analysis (assumed to be placed in the same directory).

## 📊 Features and Analysis Performed

- **Initial Data Inspection**
  - Dataset shape, data types, missing values, and summary statistics.
  
- **Data Cleaning**
  - Missing value imputation
  - Duplicate removal
  - Categorical type conversion

- **Visualizations**
  - Pie charts for city distribution and customer satisfaction
  - Box plots for outliers in numerical features
  - Histograms and scatter plots for deeper insights
  - Correlation heatmap

- **Feature Engineering**
  - Age group bins
  - Spending segment labels

- **Advanced Analysis**
  - Gender distribution across membership types
  - Satisfaction based on discount application
  - RFM (Recency, Frequency, Monetary) segmentation

## 🛠 Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn

You can install the dependencies using:

```bash
pip install pandas numpy matplotlib seaborn
