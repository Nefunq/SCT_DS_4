import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('Road Accident Data.csv')

# Someone must have written 'Fetal' instead of 'Fatal'. Noticed it in 'Accident Severity Distribution Graph'.
# The csv has more than 300k entries, Not sure how many times the error occured.
# Replacing 'Fetal' with 'Fatal', thus solving the issue.
df['Accident_Severity'] = df['Accident_Severity'].replace('Fetal', 'Fatal')

# Display basic info
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nColumn data types:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isnull().sum())

# Data cleaning and preprocessing

# Convert 'Accident Date' to datetime (format: month/day/year)
df['Accident Date'] = pd.to_datetime(df['Accident Date'], format='%m/%d/%Y')

# Extract hour from 'Time'
df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.hour

# Create time period categories
def time_period(hour):
    if pd.isna(hour):
        return 'Unknown'
    elif 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

df['Time_Period'] = df['Hour'].apply(time_period)

# Check unique values in key categorical columns
print("\nUnique Road Surface Conditions:", df['Road_Surface_Conditions'].unique())
print("Unique Weather Conditions:", df['Weather_Conditions'].unique())
print("Unique Light Conditions:", df['Light_Conditions'].unique())
print("Unique Accident Severity:", df['Accident_Severity'].unique())

# Fill or drop missing values if necessary (for simplicity, we'll drop rows with missing critical columns)
critical_cols = ['Latitude', 'Longitude', 'Road_Surface_Conditions', 'Weather_Conditions', 'Light_Conditions']
df_clean = df.dropna(subset=critical_cols)

print("\nAfter dropping missing critical values, shape:", df_clean.shape)

# Analysis and Visualizations

# 1. Accidents by Road Surface Conditions
plt.figure(figsize=(10,6))
sns.countplot(data=df_clean, y='Road_Surface_Conditions', order=df_clean['Road_Surface_Conditions'].value_counts().index)
plt.title('Number of Accidents by Road Surface Condition')
plt.xlabel('Count')
plt.tight_layout()
plt.savefig('ds_task4.1.png')
plt.show()

# 2. Accidents by Weather Conditions
plt.figure(figsize=(10,6))
sns.countplot(data=df_clean, y='Weather_Conditions', order=df_clean['Weather_Conditions'].value_counts().index)
plt.title('Number of Accidents by Weather Condition')
plt.xlabel('Count')
plt.tight_layout()
plt.savefig('ds_task4.2.png')
plt.show()

# 3. Accidents by Time of Day (Hour)
plt.figure(figsize=(12,6))
sns.histplot(data=df_clean, x='Hour', bins=24, discrete=True)
plt.title('Accidents by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Count')
plt.xticks(range(0,24))
plt.tight_layout()
plt.savefig('ds_task4.3.png')
plt.show()

# 4. Accidents by Time Period
plt.figure(figsize=(8,5))
sns.countplot(data=df_clean, x='Time_Period', order=['Morning','Afternoon','Evening','Night'])
plt.title('Accidents by Time Period')
plt.xlabel('Time Period')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('ds_task4.4.png')
plt.show()

# 5. Accident Severity distribution
plt.figure(figsize=(8,5))
sns.countplot(data=df_clean, x='Accident_Severity')
plt.title('Accident Severity Distribution')
plt.xlabel('Severity')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('ds_task4.5.png')
plt.show()

# 6. Cross-tabulation: Road Surface vs Weather (contributing factors)
# Create a crosstab of Road Surface and Weather (top conditions)
top_weather = df_clean['Weather_Conditions'].value_counts().nlargest(5).index
top_road = df_clean['Road_Surface_Conditions'].value_counts().nlargest(5).index
df_subset = df_clean[df_clean['Weather_Conditions'].isin(top_weather) & df_clean['Road_Surface_Conditions'].isin(top_road)]
ct = pd.crosstab(df_subset['Road_Surface_Conditions'], df_subset['Weather_Conditions'])
plt.figure(figsize=(10,8))
sns.heatmap(ct, annot=True, fmt='d', cmap='Blues')
plt.title('Accident Count: Road Surface vs Weather (Top Categories)')
plt.tight_layout()
plt.savefig('ds_task4.6.png')
plt.show()

# 7. Light Conditions vs Time Period
plt.figure(figsize=(10,6))
sns.countplot(data=df_clean, x='Time_Period', hue='Light_Conditions')
plt.title('Light Conditions by Time Period')
plt.xlabel('Time Period')
plt.ylabel('Count')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('ds_task4.7.png')
plt.show()

# 8. Accident hotspots (density map) using folium
# Create a base map centered on mean coordinates
map_center = [df_clean['Latitude'].mean(), df_clean['Longitude'].mean()]
m = folium.Map(location=map_center, zoom_start=12)

# Prepare data for heatmap: list of [lat, lon] pairs
heat_data = [[row['Latitude'], row['Longitude']] for index, row in df_clean.iterrows()]

# Add heatmap layer
HeatMap(heat_data, radius=10, blur=15, max_zoom=1).add_to(m)

# Save map to HTML
m.save('accident_hotspots.html')
print("Hotspot map saved as 'accident_hotspots.html'")

# Optionally, display in notebook (if in Jupyter)
# from IPython.display import display
# display(m)

# 9. Additional: Accidents by Speed Limit
plt.figure(figsize=(10,6))
sns.countplot(data=df_clean, x='Speed_limit')
plt.title('Accidents by Speed Limit')
plt.xlabel('Speed Limit (mph)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('df_task4.8.png')
plt.show()

# 10. Vehicle Type analysis
plt.figure(figsize=(12,6))
top_vehicles = df_clean['Vehicle_Type'].value_counts().nlargest(10).index
sns.countplot(data=df_clean[df_clean['Vehicle_Type'].isin(top_vehicles)], y='Vehicle_Type', order=top_vehicles)
plt.title('Top 10 Vehicle Types Involved in Accidents')
plt.xlabel('Count')
plt.tight_layout()
plt.savefig('ds_task4.9.png')
plt.show()
