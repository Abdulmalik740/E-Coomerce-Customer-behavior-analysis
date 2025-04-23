import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Load the dataset
df = pd.read_csv('E-commerce Customer Behavior.csv')

# Initial inspection
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData types and missing values:")
print(df.info())
print("\nSummary statistics:")
print(df.describe(include='all'))

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

GenderwithCity = df['City'].value_counts().reset_index()
plt.figure(figsize=(8,8))
plt.pie(GenderwithCity['count'],
        shadow=True,labels= GenderwithCity['City'],
        autopct='%1.2f%%'
       )
plt.legend(loc=(0.8,1))
plt.show()

# Handle missing values in Satisfaction Level (2 missing values) - CORRECTED
df = df.assign(**{'Satisfaction Level': df['Satisfaction Level'].fillna('Neutral')})

# Check for duplicates
print("\nNumber of duplicates:", df.duplicated().sum())

# Convert boolean to more readable format
df['Discount Applied'] = df['Discount Applied'].map({True: 'Yes', False: 'No'})

# Convert categorical columns to proper data types
categorical_cols = ['Gender', 'City', 'Membership Type', 'Discount Applied', 'Satisfaction Level']
for col in categorical_cols:
    df[col] = df[col].astype('category')

# Check for outliers in numerical columns
numerical_cols = ['Age', 'Total Spend', 'Items Purchased', 'Average Rating', 'Days Since Last Purchase']
plt.figure(figsize=(15, 8))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# Feature engineering - create bins for age groups
df['Age Group'] = pd.cut(df['Age'], 
                         bins=[20, 30, 40, 50], 
                         labels=['20-29', '30-39', '40-49'])

# Create customer segment based on spending
df['Spending Segment'] = pd.qcut(df['Total Spend'], 
                                q=3, 
                                labels=['Low', 'Medium', 'High'])

# Set style for visualizations
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

# 1. Customer Satisfaction Distribution
plt.subplot(2, 2, 1)
satisfaction_counts = df['Satisfaction Level'].value_counts()
plt.pie(satisfaction_counts, labels=satisfaction_counts.index, autopct='%1.1f%%', 
        colors=['#66b3ff','#99ff99','#ff9999'])
plt.title('Customer Satisfaction Distribution')

# 2. Total Spend by Membership Type - CORRECTED
plt.subplot(2, 2, 2)
sns.boxplot(x='Membership Type', y='Total Spend', data=df, palette='Set2')
plt.title('Total Spend by Membership Type')

# 3. Average Rating Distribution
plt.subplot(2, 2, 3)
sns.histplot(df['Average Rating'], bins=10, kde=True, color='skyblue')
plt.title('Distribution of Average Ratings')

# 4. Days Since Last Purchase by Satisfaction - CORRECTED
plt.subplot(2, 2, 4)
sns.boxplot(x='Satisfaction Level', y='Days Since Last Purchase', data=df, palette='pastel')
plt.title('Days Since Last Purchase by Satisfaction')

plt.tight_layout()
plt.show()

# Additional analysis
# Correlation between numerical variables
corr_matrix = df[numerical_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Variables')
plt.show()

# Gender distribution across membership types - CORRECTED
plt.figure(figsize=(8, 6))
sns.countplot(x='Membership Type', hue='Gender', data=df, palette='Set2')
plt.title('Gender Distribution Across Membership Types')
plt.show()

# Average rating by city - CORRECTED
plt.figure(figsize=(12, 6))
sns.barplot(x='City', y='Average Rating', data=df, errorbar=None, palette='viridis')
plt.title('Average Rating by City')
plt.ylim(3, 5)
plt.show()

# 1. Relationship between Age, Total Spend and Satisfaction
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Total Spend', hue='Satisfaction Level', 
                size='Items Purchased', sizes=(20, 200), data=df, alpha=0.7)
plt.title('Age vs Total Spend by Satisfaction Level')
plt.show()

# 2. Discount impact on spending and satisfaction - CORRECTED
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='Discount Applied', y='Total Spend', data=df, palette='Set2')
plt.title('Total Spend by Discount Applied')

plt.subplot(1, 2, 2)
sns.countplot(x='Satisfaction Level', hue='Discount Applied', data=df, palette='Set2')
plt.title('Satisfaction by Discount Applied')
plt.tight_layout()
plt.show()

# 3. RFM Analysis (Recency, Frequency, Monetary)
rfm = df[['Customer ID', 'Days Since Last Purchase', 'Items Purchased', 'Total Spend']].copy()
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Create RFM segments (simplified)
rfm['R_Score'] = pd.qcut(rfm['Recency'], q=4, labels=[4, 3, 2, 1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'], q=4, labels=[1, 2, 3, 4])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=4, labels=[1, 2, 3, 4])
rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

# Merge back with original data
df = df.merge(rfm[['CustomerID', 'RFM_Score']], left_on='Customer ID', right_on='CustomerID', how='left')

# Visualize RFM segments
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Recency', y='Monetary', hue='RFM_Score', data=rfm, palette='viridis', alpha=0.7)
plt.title('RFM Analysis: Recency vs Monetary Value')
plt.show()
