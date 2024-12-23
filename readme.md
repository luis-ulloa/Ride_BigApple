# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) App: Ride BigApple 🚖🍎

### Problem Statement

Can yellow cab fares be predicted within New York City's five boroughs based on time of the day, time of the year and certain high passenger areas?
The aim here is to use existing historical data of NYC cab rides spanning several years and building a model that can be integrated into a simple app to request a taxi ride and predict the expected fare and distance for the passenger.


-----

### Datasets

The original <a href = 'https://www.kaggle.com/c/new-york-city-taxi-fare-prediction'>dataset</a> contains 55.4 million rows and eight columns. Given the vast size of this dataset, only 60 thousand rows were randomly sampled. This subset is labeled `nyc_taxidata_60k.csv`. The original eight columns, in order, are: `key`, `fare_amount`, `pickup_datetime`, `pickup_longitude`, `pickup_latitude`, `dropoff_longitude`, `dropoff_latitude` and `passenger_count`.
After the data cleaning process was completed, the following datasets were generated and saved: `taxi_clean_set_v1.csv`, `taxi_clean_set_v2.csv`, `taxi_clean_set_v3.csv`.  The data dictionary below corresponds to the latest dataset, which contains 35 columns. 

|Feature|Type|Description|
|---|---|---|
|**fare_amount**|_float_|Fare paid by passenger. Float format, unit in dollars.|
|**pickup_longitude**|_float_|Longitude coordinate of pickup location. Float format, unit in degrees.|
|**pickup_latitude**|_float_|Latitude coordinate of pickup location. Float format, unit in degrees.|
|**dropoff_longitude**|_float_|Longitude coordinate of dropoff location. Float format, unit in degrees.|
|**dropoff_latitude**|_float_|Latitude coordinate of dropoff location. Float format, unit in degrees.|
|**passenger_count**|_integer_|Number of passengers in taxi ride. Integer format, unit in persons.|
|**year**|_integer_|Year in which taxi ride took place. Integer format, no units.|
|**month**|_object_|Month in which taxi ride took place. String format, no units.|
|**day_of_week**|_object_|Day of the week in which taxi ride took place. String format, no units.|
|**hour**|_integer_|Hour of the day in which taxi ride took place. Integer format, unit in hours.|
|**minute**|_integer_|Minute within the hour in which taxi ride took place. Integer format, unit in minutes.|
|**geodesic_distance**|_float_|Straight line distance over earth surface between pickup and dropoff points. Float format, unit in kilometers.|
|**pickup_clusters**|_integer_|Cluster label assigned by DBSCAN algorithm. Integer format, no units.|
|**pickup_clusters_color**|_object_|Color corresponding to assigned cluster label. String format, unit in hexadecimals.|
|**p_0**|_integer_|Pickup cluster number 0, it is `1` if the ride belongs to it, otherwise it is `0`. Integer format, no units.|
|**p_1**|_integer_|Pickup cluster number 1, it is `1` if the ride belongs to it, otherwise it is `0`. Integer format, no units.|
|**p_2**|_integer_|Pickup cluster number 2, it is `1` if the ride belongs to it, otherwise it is `0`. Integer format, no units.|
|**p_3**|_integer_|Pickup cluster number 3, it is `1` if the ride belongs to it, otherwise it is `0`. Integer format, no units.|
|**p_4**|_integer_|Pickup cluster number 4, it is `1` if the ride belongs to it, otherwise it is `0`. Integer format, no units.|
|**dropoff_clusters**|_integer_|Cluster label assigned by DBSCAN algorithm. Integer format, no units.|
|**dropoff_clusters_color**|_object_|Color corresponding to assigned cluster label. String format, unit in hexadecimals.|
|**d_0**|_integer_|Dropoff cluster number 0, it is `1` if the ride belongs to it, otherwise it is `0`. Integer format, no units.|
|**d_1**|_integer_|Dropoff cluster number 1, it is `1` if the ride belongs to it, otherwise it is `0`. Integer format, no units.|
|**d_2**|_integer_|Dropoff cluster number 2, it is `1` if the ride belongs to it, otherwise it is `0`. Integer format, no units.|
|**d_3**|_integer_|Dropoff cluster number 3, it is `1` if the ride belongs to it, otherwise it is `0`. Integer format, no units.|
|**d_4**|_integer_|Dropoff cluster number 4, it is `1` if the ride belongs to it, otherwise it is `0`. Integer format, no units.|
|**p_5**|_integer_|Dropoff cluster number 5, it is `1` if the ride belongs to it, otherwise it is `0`. Integer format, no units.|
|**estimated_distance**|_float_|Distance estimated from column `geodesic_distance` to better approximate actual taxi ride distance. Float format, unit in kilometers.|
|**distance_hour**|_float_|Interaction term formed by multiplying `hour` and `estimated_distance`. Float format, no units.|
|**airport_ride**|_object_|Airport code label assigned to taxi ride if it started or ended at either of the three major airports in the NYC metro area. Possible values are `LGA`, `JFK` or `EWR`. String format, no units.|
|**JFK**|_integer_|Actual aiport designator column, it is `1` if the ride started or ended at JFK, otherwise it is `0`. Integer format, no units.|
|**LGA**|_integer_|Actual aiport designator column, it is `1` if the ride started or ended at LGA, otherwise it is `0`. Integer format, no units.
|**peak_rides**|_integer_|Rush hour designator. It is `1` if the ride took place within the rush hour window, otherwise it is `0`. Rush hour is Monday thru Friday only. The morning hours are from 6:30am to 9:30am and the afternoon/evening hours are from 3:00pm to 8:00pm. Integer format, no units.|
|**weekend_rides**|_integer_|Weekend designator. It is `1` if the ride took place on either Saturday or Sunday, otherwise it is `0`. Integer format, no units.|
|**holiday_rides**|_integer_|Christmas holiday designator. It is `1` if the ride took place on either day in November or December, otherwise it is `0`. Integer format, no units.|

---

### Summary

Given the available data, predicting a yellow cab fare in NYC, as well as the estimated ride distance is possible. However, the final model yielded a RMSE of 3.60 dollars. At first glance, that's a good performance, but the base fare for a cab ride in NYC is 2.50 dollars, so trying to get the RMSE below that threshold is important since a very short ride may end up with a predicted fare that's below the base, or even negative.


---

### Recommendation

Version 2.0, which aims to improve RMSE should take into account the following:
* Going back to the EDA section to deeper explore identified relationships between features.
* For some of the engineered features, dissect the data into more granular form. For example, expanding weekend rides to include rides taking place on Fridays past noon (Friday has the highest rides), as well as start counting the holiday rides on Thanksgiving week rather than the whole month of November.
* Clustering: Ignore to a certain degree the silhouette scores and focus more on building smaller, localized pickup and dropoff clusters around local landmarks known to draw high passenger traffic, such as baseball stadiums, Madison Square Garden, Grand Central Station, Penn Station, Columbus Circle, Wall Street, Times Square, etc.
Exploring all these possibilities and employing more in-depth feature engineering, may help make version 2.0 even more accurate.

---

### Technical Report

Complete and thorough analysis in various Jupyter notebooks are located below:<br> 
[01_Intro_and_Cleaning.ipynb](./code/01_Intro_and_Cleaning.ipynb).<br>
[02_EDA_and_Visualizations.ipynb](./code/02_Exploratory_Data_Analysis.ipynb).<br>
[03_Pre-Processing_and_Modeling.ipynb](./code/03_Pre-Processing_and_Modeling.ipynb).<br>

---
### Models and App
Pickled models are located [here](./models/).<br>
Interactive Streamlit App is located [here](./scripts/).

---


### Presentation

Accompanying presentation in PDF format is located [here](./presentation/).

