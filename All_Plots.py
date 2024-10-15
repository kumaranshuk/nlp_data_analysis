import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
# Load the Excel file
file_path = 'Output Data Structure.xlsx'  # Replace with your actual file path
data = pd.read_excel(file_path, sheet_name='Sheet1')  # Adjust the sheet name if necessary

#Distribution Plots----------------------------------------------


# Distribution of Scores and Metrics
metrics = ['POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE',
           'FOG INDEX', 'AVG SENTENCE LENGTH', 'WORD COUNT', 'SYLLABLE PER WORD']

plt.figure(figsize=(20, 20))

for i, metric in enumerate(metrics):
    plt.subplot(4, 2, i + 1)
    sns.histplot(data[metric], bins=20, kde=True)
    plt.title(f'Distribution of {metric}')

plt.tight_layout()
plt.show()


#Correlation Heatmap-----------------------------------------



# Correlation Heatmap
numerical_data = data.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
correlation_matrix = numerical_data.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Draw the heatmap
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Key Metrics')
plt.show()




#Scatter Plots with Regression Lines-----------------------------------------------------------
plt.figure(figsize=(15, 10))

# Positive Score vs Word Count
plt.subplot(2, 2, 1)
sns.regplot(x='WORD COUNT', y='POSITIVE SCORE', data=data, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Word Count vs Positive Score')

# Negative Score vs Word Count
plt.subplot(2, 2, 2)
sns.regplot(x='WORD COUNT', y='NEGATIVE SCORE', data=data, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Word Count vs Negative Score')

# Fog Index vs Sentence Length
plt.subplot(2, 2, 3)
sns.regplot(x='AVG SENTENCE LENGTH', y='FOG INDEX', data=data, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Average Sentence Length vs Fog Index')

# Polarity Score vs Subjectivity Score
plt.subplot(2, 2, 4)
sns.regplot(x='POLARITY SCORE', y='SUBJECTIVITY SCORE', data=data, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Polarity Score vs Subjectivity Score')

plt.tight_layout()
plt.show()



#Box Plots----------------------------------------------------------------

plt.figure(figsize=(15, 10))

# Box plot for Positive Score
plt.subplot(2, 2, 1)
sns.boxplot(y=data['POSITIVE SCORE'], color='green')
plt.title('Box Plot of Positive Score')

# Box plot for Negative Score
plt.subplot(2, 2, 2)
sns.boxplot(y=data['NEGATIVE SCORE'], color='red')
plt.title('Box Plot of Negative Score')

# Box plot for Fog Index
plt.subplot(2, 2, 3)
sns.boxplot(y=data['FOG INDEX'], color='orange')
plt.title('Box Plot of Fog Index')

# Box plot for Subjectivity Score
plt.subplot(2, 2, 4)
sns.boxplot(y=data['SUBJECTIVITY SCORE'], color='purple')
plt.title('Box Plot of Subjectivity Score')

plt.tight_layout()
plt.show()


## Bar Plot for Top and Bottom Positive Scores------------------------------------------------------------


top_n = 10
plt.figure(figsize=(14, 6))

# Top 10 Positive Scores
plt.subplot(1, 2, 1)
top_positive = data.nlargest(top_n, 'POSITIVE SCORE')
sns.barplot(x='POSITIVE SCORE', y='URL_ID', data=top_positive, hue='URL_ID', dodge=False, palette='Greens_r', legend=False)
plt.title(f'Top {top_n} URLs by Positive Score')

# Bottom 10 Positive Scores
plt.subplot(1, 2, 2)
bottom_positive = data.nsmallest(top_n, 'POSITIVE SCORE')
sns.barplot(x='POSITIVE SCORE', y='URL_ID', data=bottom_positive, hue='URL_ID', dodge=False, palette='Reds', legend=False)
plt.title(f'Bottom {top_n} URLs by Positive Score')

plt.tight_layout()
plt.show()


#Heatmap of Counts------------------------------------------------------------



# Binning the data
data['POSITIVE_BIN'] = pd.cut(data['POSITIVE SCORE'], bins=5)
data['NEGATIVE_BIN'] = pd.cut(data['NEGATIVE SCORE'], bins=5)

# Creating the heatmap
heatmap_data = pd.crosstab(data['POSITIVE_BIN'], data['NEGATIVE_BIN'])
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, cmap='Blues')
plt.title('Heatmap of Positive vs Negative Scores')
plt.show()
