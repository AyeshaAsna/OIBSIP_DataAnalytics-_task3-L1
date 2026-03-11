# OIBSIP_DataAnalytics-_task3-L1
This project focuses on cleaning raw datasets by handling missing values, removing duplicates, correcting inconsistencies, and detecting outliers. The goal is to improve data quality and reliability so that accurate analysis and insights can be generated from the dataset.
# Data Cleaning Project

## Objective
The objective of this project is to clean and preprocess raw datasets to improve data quality and reliability for analysis. Data cleaning ensures that the dataset is accurate, consistent, and ready for further data analysis or machine learning tasks.

## Dataset
Dataset 1 Link: AB_NYC_2019.csv.zip
Dataset 2 Link: (Add your dataset link)

The datasets contain raw data that requires preprocessing such as handling missing values, removing duplicates, and standardizing formats.

## Tools & Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- VS Code

## Steps Performed
1. Loaded the datasets using the Pandas library.
2. Explored the structure of the datasets and checked for missing values.
3. Handled missing data using appropriate methods such as removal or imputation.
4. Identified and removed duplicate records.
5. Standardized data formats and ensured consistency across columns.
6. Detected and analyzed outliers that could affect analysis.
7. Prepared the cleaned dataset for further analysis.

## Code (Example)

```python
import pandas as pd

data = pd.read_csv("dataset.csv")

# Check missing values
print(data.isnull().sum())

# Remove duplicates
data = data.drop_duplicates()

# Fill missing values
data = data.fillna(method="ffill")

Key Insights

Data cleaning improves dataset reliability.

Removing duplicates and handling missing values prevents incorrect analysis.

Standardized data ensures consistency across the dataset.

Outcome

The dataset was successfully cleaned and prepared for accurate data analysis and modeling. This process ensures reliable insights and better decision-making.

Author

Ayesha Asna
