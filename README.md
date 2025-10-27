# Overview

This streamlit application displays the results of a clustering analysis of temperature anomalies based on ENSO conditions (El Niño and La Niña). On the map, multiple clusters are shown with different colors. scikit-learn's K-means clustering was used for this clustering analysis.

# Data

Data from Berkeley Earth was used for this project. This data provides temperature anomalies at each land coordinate for each month in NetCDF format. The data can be found [here](https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Gridded/Complete_TAVG_LatLong1.nc).

# Features

This Streamlit project consists of an interactive map that displays the cluster to which a particular coordinate point belongs and the average temperature anomaly for that coordinate during El Niño and La Niña, as well as a table displaying the average temperature anomaly for each ENSO condition within each cluster. 

# How to reproduce

This streamlit project can be found at ensoanomalyclustering.streamlit.app.


