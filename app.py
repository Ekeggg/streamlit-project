import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import gdown
import os
import warnings
import plotly.express as px
warnings.filterwarnings("ignore")

file_id = "1n7cREgviHR9PJjMZtgverCPIB3F1blm2"
output_path = "filled_output.csv"
if not os.path.exists(output_path):
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
with st.spinner('Loading data...'):
    chunk_iter = pd.read_csv(output_path, chunksize=30000)
    df = pd.concat(chunk_iter)
st.title("Clustering Analysis of Temperature Data")
#csv file path
enso_labels = {
    2017: "L", 2016: "N", 2015: "E", 2014: "E", 2013: "N", 2012: "N", 2011: "L", 2010: "L",
    2009: "E", 2008: "L", 2007: "L", 2006: "E", 2005: "N", 2004: "E", 2003: "N", 2002: "E",
    2001: "N", 2000: "L", 1999: "N", 1998: "L", 1997: "E", 1996: "L", 1995: "N", 1994: "E",
    1993: "E", 1992: "E", 1991: "E", 1990: "N", 1989: "N", 1988: "L", 1987: "E", 1986: "N",
    1985: "N", 1984: "N", 1983: "N", 1982: "E", 1981: "N", 1980: "N", 1979: "N", 1978: "N",
    1977: "E", 1976: "N", 1975: "L", 1974: "L", 1973: "L", 1972: "E", 1971: "L", 1970: "L",
    1969: "E", 1968: "N", 1967: "N", 1966: "N", 1965: "E", 1964: "L", 1963: "E", 1962: "N",
    1961: "N", 1960: "N", 1959: "N", 1958: "N", 1957: "E", 1956: "L", 1955: "L", 1954: "N",
    1953: "E", 1952: "N", 1951: "E", 1950: "L", 1949: "N", 1948: "N", 1947: "L", 1946: "E",
    1945: "N", 1944: "N", 1943: "N", 1942: "N", 1941: "E", 1940: "E", 1939: "N", 1938: "L",
    1937: "N", 1936: "N", 1935: "N", 1934: "N", 1933: "N", 1932: "E", 1931: "N", 1930: "N",
    1929: "N", 1928: "N", 1927: "N", 1926: "N", 1925: "E", 1924: "L", 1923: "E", 1922: "N",
    1921: "L", 1920: "N", 1919: "E", 1918: "E", 1917: "L", 1916: "L", 1915: "N", 1914: "E",
    1913: "E", 1912: "N", 1911: "E", 1910: "L", 1909: "L", 1908: "L", 1907: "N", 1906: "L",
    1905: "E", 1904: "N", 1903: "N", 1902: "E", 1901: "N", 1900: "N", 1899: "N", 1898: "N",
    1897: "N", 1896: "E", 1895: "N", 1894: "N", 1893: "L", 1892: "L", 1891: "N", 1890: "N"
}
df['temp_smooth'] = df['temperature'].rolling(window=3, center=True).mean()
X = df[['latitude', 'longitude', 'month', 'temp_smooth']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
clustering_df = df[(df['year'] >= 1890) & (df['year'] <= 2017)]

# keep only years that are in the ENSO labels
clustering_df = clustering_df[clustering_df['year'].isin(enso_labels.keys())]
clustering_df = clustering_df.copy()
clustering_df['enso_label'] = clustering_df['year'].map(enso_labels)
# sidebar filters
selected_phases = st.sidebar.multiselect("Select ENSO Phases", options=['E', 'L'], default=['E', 'L'])
selected_months = st.sidebar.multiselect("Select Months", options=[12, 1, 2], default=[12, 1, 2])

clustering_df = clustering_df[clustering_df['enso_label'].isin(selected_phases)]
clustering_df = clustering_df[clustering_df['month'].isin(selected_months)]
clustering_df = pd.get_dummies(clustering_df, columns=['enso_label'], prefix='enso')

melted = clustering_df.melt(
    id_vars=['latitude', 'longitude', 'temp_smooth'],
    value_vars=['enso_E', 'enso_L'],
    var_name='enso_phase',
    value_name='is_phase'
)

melted = melted[melted['is_phase'] == 1]


pivot = melted.groupby(['latitude', 'longitude', 'enso_phase'])['temp_smooth'] \
              .mean().unstack(fill_value=0).reset_index()

clustering_cols = ['enso_E', 'enso_L']
X = pivot[clustering_cols]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
k = st.sidebar.slider("Select number of clusters (k)", min_value=2, max_value=10, value=6)
kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
pivot['cluster'] = kmeans.fit_predict(X_scaled)


fig = px.scatter(
pivot, x='longitude', y='latitude', color='cluster',
title='Interactive Cluster Map',
color_continuous_scale='Viridis',
labels={'longitude': 'Longitude', 'latitude': 'Latitude'},
hover_data={'enso_E': True, 'enso_L': True}
)
st.plotly_chart(fig)
st.write("This image shows the result of a clustering analysis that groups global land locations based on how their December–January–February (DJF) temperatures respond to ENSO (El Niño–Southern Oscillation) events." \
" K-means clustering was applied to temperature anomalies during El Niño (E) and La Niña (L) winters, resulting in distinct clusters that represent different temperature response patterns. Each color represents a different cluster, indicating how locations respond to ENSO events.")
pivot['temp_diff'] = pivot['enso_E'] - pivot['enso_L']
cluster_summary = pivot.groupby('cluster')[['enso_E', 'enso_L', 'temp_diff']].mean().reset_index()
st.dataframe(cluster_summary)
# Show statistics
st.subheader('Cluster Statistics')
st.write(pivot.groupby('cluster')[['enso_E', 'enso_L']].describe())
st.write("""
This map shows the clustering of global locations based on their temperature response during El Niño and La Niña winters (December–February). 
Each color corresponds to a different response pattern. For example, Cluster 5 shows strong warming during El Niño events, particularly in the tropics.

Clusters were identified using a machine learning algorithm on temperature anomaly data, and their average responses are summarized above.
""")
st.subheader('Cluster Summary')
st.write("""
The strongest El Niño warming (Cluster 5) appears in equatorial and tropical regions, aligning with canonical ENSO teleconnection patterns. 
Cluster 3 exhibits notable cooling during El Niño, consistent with responses in western North America and Asia. 
Some clusters (e.g. Cluster 2) display warming in both phases, indicating potential contamination by the long-term warming trend or insufficient detrending.
""")