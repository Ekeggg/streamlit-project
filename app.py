import streamlit as st
import polars as pl
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import gdown
import os

# Silence Polars warnings
import warnings
warnings.filterwarnings("ignore")

# ENSO phase labels
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

file_id = "1n7cREgviHR9PJjMZtgverCPIB3F1blm2"
output_path = "filled_output.csv"

# Download CSV if needed
if not os.path.exists(output_path):
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

st.title("Clustering Analysis of Temperature Data")

# Sidebar filters
selected_phases = st.sidebar.multiselect(
    "Select ENSO Phases", options=['E', 'L'], default=['E', 'L']
)
selected_months = st.sidebar.multiselect(
    "Select Months", options=[12, 1, 2], default=[12, 1, 2]
)
k = st.sidebar.slider("Select number of clusters (k)", min_value=2, max_value=10, value=6)

# Show loading spinner
with st.spinner('Loading and processing data...'):

    # Lazy read
    df = pl.scan_csv(output_path)

    # Filter
    df = df.filter(
        pl.col("month").is_in(selected_months) &
        pl.col("year").is_in(list(enso_labels.keys()))
    )

    # Join ENSO labels
    enso_df = pl.DataFrame({
        "year": list(enso_labels.keys()),
        "enso_label": list(enso_labels.values())
    }).lazy()
    df = df.join(enso_df, on="year", how="inner")

    # Smooth temperature
    # Polars rolling mean requires sorted data
    df = df.sort(["latitude", "longitude", "year", "month"])
    df = df.with_columns(
        pl.col("temperature")
        .rolling_mean(window_size=3, center=True)
        .alias("temp_smooth")
    )

    # Aggregate mean temp_smooth per phase
    agg_df = (
        df.groupby(["latitude", "longitude", "enso_label"])
        .agg(pl.col("temp_smooth").mean().alias("temp_mean"))
    )

    # Collect to pandas
    agg_pd = agg_df.collect().to_pandas()

    # Pivot
    pivot = agg_pd.pivot(
        index=["latitude", "longitude"],
        columns="enso_label",
        values="temp_mean"
    ).reset_index()

    # Rename columns to match your previous naming
    pivot = pivot.rename(columns={"E": "enso_E", "L": "enso_L"})

    # Fill any missing values
    pivot = pivot.fillna(0)

    # Clustering
    clustering_cols = ['enso_E', 'enso_L']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pivot[clustering_cols])

    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    pivot['cluster'] = kmeans.fit_predict(X_scaled)

    # Difference
    pivot['temp_diff'] = pivot['enso_E'] - pivot['enso_L']

# Plotly scatter
fig = px.scatter(
    pivot,
    x='longitude',
    y='latitude',
    color='cluster',
    title='Interactive Cluster Map',
    color_continuous_scale='Viridis',
    labels={'longitude': 'Longitude', 'latitude': 'Latitude'},
    hover_data={'enso_E': True, 'enso_L': True, 'temp_diff': True}
)
st.plotly_chart(fig)

# Description
st.markdown("""
This image shows the result of a clustering analysis that groups global land locations based on how their December–January–February (DJF) temperatures respond to ENSO (El Niño–Southern Oscillation) events. 
K-means clustering was applied to temperature anomalies during El Niño (E) and La Niña (L) winters, resulting in distinct clusters that represent different temperature response patterns. Each color represents a different cluster.
""")

# Cluster summary table
cluster_summary = pivot.groupby('cluster')[['enso_E', 'enso_L', 'temp_diff']].mean().reset_index()
st.subheader('Cluster Summary')
st.dataframe(cluster_summary)

# Descriptive statistics
st.subheader('Cluster Statistics')
st.write(pivot.groupby('cluster')[['enso_E', 'enso_L']].describe())

st.write("""
The strongest El Niño warming (Cluster with highest enso_E) appears in equatorial and tropical regions, aligning with canonical ENSO teleconnection patterns.
Clusters exhibiting cooling during El Niño are consistent with known responses in North America and Asia.
""")
