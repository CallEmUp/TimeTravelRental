##Setup and Installation

# install external libraries necessary for my project:
# geopandas, pandas, matplotlib, requests, folium
pip install geopandas pandas matplotlib requests folium scikit-learn seaborn

# to install traveltimepy, I am using version 3.9.6 because
# 3.9.7 is currently having stability issues that affect my code
pip install traveltimepy==3.9.6

# import packages from libraries for analysis, data pre-processing, and visualizations
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from traveltimepy import TravelTimeSdk
from IPython.display import display, HTML
import sklearn.preprocessing as skp
from sklearn.cluster import KMeans
import folium
from scipy import stats

app_id = os.getenv('TRAVELTIME_APP_ID', '981c6f8d')
api_key = os.getenv('TRAVELTIME_API_KEY', 'c5fffce90d3ee3c78d1aa2f07ca93be3')

sdk = TravelTimeSdk(app_id=app_id, api_key=api_key)







##Data Pre-processing

# read csv file, store as 'df'
df = pd.read_csv("ZillowScrape.csv")

for column in df.columns:
  print(column) # print out columns to see all of them, pick out which ones are relevant

df.head(3) # to get a general idea of the dataset


# create a list of tentative columns that I plan on using for my project
tentative_columns = ['address', 'area', 'baths', 'latLong/latitude', 'latLong/longitude']

# for loop that prints the number of na values in each of these columns
print("# of NA values by column")
for column in tentative_columns:
  print(column, ':', np.sum(df[column].isna()))

print()
print(f'The dataframe has {df.shape[0]} rows') # print number of rows to get a grasp of how impactful the NaN values are


# helper function clean_price, removes unnecessary characters from 'price' column
# to only get price in float form. If no price, returns NaN
def clean_price(price):
    if pd.notna(price):
        return float(price.replace('$', '').replace(',', '').replace('+', '').replace('/mo', ''))
    return np.nan

expanded_rows = []

# for loop that iterates over each row in the original df
for _, row in df.iterrows():
    row_added = False  # flag to check if any row was added for this entry

    # for loop that checks each unit column from 0 to 5 (assuming a maximum of 6 units as an example cause i saw nothing above 6)
    for i in range(6):
        unit_beds = row.get(f'units/{i}/beds')
        unit_price = row.get(f'units/{i}/price')
        unit_baths = row.get(f'baths')

        # if statement that checks if unit beds is a valid value to insert
        if pd.notna(unit_beds):
            beds = int(unit_beds) # cast as int

            # normalize NaN 'bath' values for studios and 1 bedroom units, if applicable
            if pd.isna(unit_baths) and beds == 1:
                unit_baths = 1.0

            # skip rows where beds are zero (no apartments available)
            if beds > 0:
                # create a new row for each bed unit based on the number of beds
                for _ in range(beds):
                    new_row = {
                        'address': (row['address']),
                        'beds': beds,
                        'baths': unit_baths if pd.notna(unit_baths) else np.nan,
                        'price': clean_price(unit_price),
                        'latitude': row.get('latLong/latitude', None),
                        'longitude': row.get('latLong/longitude', None),
                    }
                    expanded_rows.append(new_row)
                    row_added = True

    # if statement for case when no rows have been added from any unit, consider adding the original row if necessary
    if not row_added:
        new_row = {
            'address': row['address'],
            'beds': row.get('beds'),
            'baths': row.get('baths') if pd.notna(row['baths']) else np.nan,
            'price': clean_price(row.get('price')),
            'latitude': row.get('latLong/latitude', None),
            'longitude': row.get('latLong/longitude', None),
        }
        expanded_rows.append(new_row)

# create a new df 'expanded_df' from the expanded rows
expanded_df = pd.DataFrame(expanded_rows)

# some price values within the DataFrame had irregular values for those columns. The prices are adjusted here
indices_to_update = [154, 155, 156, 157]
expanded_df.loc[indices_to_update, 'price'] = 7250

expanded_df.head(3) # print to check if code works properly
expanded_df.shape


# remove rows with NaN values in 'beds', 'baths', and 'price'
# declare how as 'all' so that ONLY columns where ALL 3 columns are NaN are dropped
temp_df = expanded_df.dropna(subset=['beds', 'baths', 'price'], how='all')
temp_df.shape


### Now I need to replace NaN 'baths' values with its average based on the number of beds in the apartment ###

# create a function that determines average number of bathrooms in apartments that have NaN as bathroom, round to nearest whole number
bath_means = temp_df.groupby('beds')['baths'].mean().round(0)

# function fill_bathrooms, takes a row in dataframe as input. if the row has an
# NaN value for 'baths', replace with the calculated average
def fill_bathrooms(row):
    if pd.isna(row['baths']):
        return bath_means.get(row['beds'], np.nan)  # default to NaN if no mean available for that bed count
    return row['baths']

cleaned_df = temp_df.copy() # create a copy of the temporary dataframe to store as cleaned_df
cleaned_df['baths'] = cleaned_df.apply(fill_bathrooms, axis=1) # apply function 'fill_bathrooms' to 'baths' column

# Feature engineering, adding 'neighborhood' column
# To implement the column 'neighborhood', I created a mapping dictionary based on
# neighborhood name, latitude, and longitude.

neighborhood_data = { # Hand collected via Google Maps
    "name": ["Alphabet City", "Battery Park City", "Carnegie Hill", "Chelsea", "Chinatown",
             "Clinton", "East Harlem", "East Village", "Financial District", "Flatiron District",
             "Gramercy Park", "Greenwich Village", "Harlem", "Hells Kitchen", "Inwood",
             "Kips Bay", "Lincoln Square", "Little Italy", "Lower East Side", "Manhattan Valley",
             "Midtown East", "Morningside Heights", "Murray Hill", "Roosevelt Island", "SoHo",
             "Tribeca", "Upper East Side", "Upper West Side", "Washington Heights", "West Village"],
    "latitude": [40.726000, 40.713000, 40.784726, 40.746000, 40.715000,
                 40.764000, 40.795000, 40.728000, 40.708000, 40.740800,
                 40.737800, 40.734000, 40.815130, 40.764000, 40.867000,
                 40.741000, 40.773828, 40.719000, 40.715000, 40.799353,
                 40.747000, 40.809722, 40.748000, 40.761389, 40.723000,
                 40.718000, 40.769000, 40.787000, 40.840000, 40.736000],
    "longitude": [-73.979000, -74.016000, -73.956070, -74.001000, -73.997000,
                  -73.992000, -73.939000, -73.986000, -74.011000, -73.989600,
                  -73.986100, -74.002000, -73.947515, -73.992000, -73.922000,
                  -73.978000, -73.984472, -73.997000, -73.985000, -73.962919,
                  -73.986000, -73.960278, -73.978000, -73.950833, -74.000000,
                  -74.008000, -73.966000, -73.975000, -73.940000, -74.004000]
}

neighborhoods = pd.DataFrame(neighborhood_data) # store as a dataframe

# Implementation of haversine function to find the distance between two coordinates (latitude and longitude)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat/2) * np.sin(dLat/2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLon/2) * np.sin(dLon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c  # Distance in kilometers
    return distance

# function find_neighborhood, takes latitude and longitude values of apartment in dataframe
# and calculates the distances for each neighborhood, returning the name of the neighborhood
# that the apartment has the minimum distance to
def find_neighborhood(lat, lon):
    distances = neighborhoods.apply(lambda row: haversine(lat, lon, row['latitude'], row['longitude']), axis=1)
    return neighborhoods.loc[distances.idxmin(), 'name']

# apply function to dataframe to get neighborhood values, store in new column 'neighborhood'
cleaned_df['neighborhood'] = cleaned_df.apply(lambda row: find_neighborhood(row['latitude'], row['longitude']), axis=1)

# Add another instance of feature engineering: 'total rooms'.
# Default is 1, which is aggregated with the number of bedrooms and bathrooms
cleaned_df['total rooms'] = (1 + cleaned_df['beds'] + cleaned_df['baths'])

# Add one last feature engineering: 'grade'
# This is a rough overall 'grade' of the apartments avaliable. This is determined by the ratio of 'total rooms' vs 'price'
# The scale is 1-10 and normalized. This is to help detemine the best price for your space
cleaned_df['room price ratio'] = cleaned_df['total rooms'] / cleaned_df['price']
# Usage of the MinMax scaler because of the desire to make the values on a 1-10 scale
scaler = skp.MinMaxScaler(feature_range=(1, 10))
cleaned_df['grade'] = scaler.fit_transform(cleaned_df[['room price ratio']])
cleaned_df.drop('room price ratio', axis=1, inplace=True)

# now that I have implemented feature engineering, I can remove rows that have outliers in columns 'price'
processed_df = cleaned_df[cleaned_df['price'].apply(lambda x: np.abs(x - (cleaned_df['price']).mean()) / cleaned_df['price'].std() < 3)]


print("Shape: ", processed_df.shape) # print shape to better understand how many rows

processed_df.head(5)





##General Dataset Exploration and Analysis

import seaborn as sns

# for column 'beds' plot a histogram using seaborn
beds_distribution = sns.displot(data = processed_df, x = 'beds').set(title = "Distribution of 'beds' Data")
# print out descriptive statistics
print("Beds Descriptive Statistics")
print(f"   Mean:   {round(processed_df['beds'].mean(), 2)}")
print(f"   Median: {round(processed_df['beds'].median(), 2)}")
print(f"   Mode:   {round(processed_df['beds'].mode()[0], 0)}")
print(f"   St Dev: {round(processed_df['beds'].std(), 2)}")
print()
# to further confirm the distribution of beds, I are using stats.describe to get variance, skewness, and kurtosis
stats.describe(processed_df.beds)


# for column 'baths'
baths_distribution = sns.displot(data = processed_df, x = 'baths').set(title = "Distribution of 'baths' Data")
print("Baths Descriptive Statistics")
print(f"   Mean:   {round(processed_df['baths'].mean(), 2)}")
print(f"   Median: {round(processed_df['baths'].median(), 2)}")
print(f"   Mode:   {round(processed_df['baths'].mode()[0], 1)}")
print(f"   St Dev: {round(processed_df['baths'].std(), 2)}")
print()


# for column 'price', print a histogram to visualize the distribution of data
price_distribution = sns.displot(data = processed_df, x = 'price').set(title = "Distribution of 'price' Data")
# print descriptive statistics
print("Price Descriptive Statistics")
print(f"   Mean:   ${round(processed_df['price'].mean(), 2)}")
print(f"   Median: ${round(processed_df['price'].median(), 2)}")
print(f"   Mode:   ${round(processed_df['price'].mode()[0], 1)}")
print(f"   St Dev: ${round(processed_df['price'].std(), 2)}")
print()
# to further confirm the distribution of price, I are using stats.describe to get variance, skewness, and kurtosis
stats.describe(processed_df.price)


# use value_counts to aggregate the number of apartments for each neighborhood
top_neighborhoods = processed_df['neighborhood'].value_counts()

# index to get top 5 neighborhoods, split into index and values
top5_neighborhoods = top_neighborhoods[0:5]

top5_neighborhood = top5_neighborhoods.index
top5_count = top5_neighborhoods.values

# plot findings of top 5 neighborhoods by count
plt.figure(figsize=(8,4))
plt.bar(top5_neighborhood, top5_count)
plt.title('5 Most Common Neighborhoods in NYC Apartment Data')
plt.xlabel('Neighborhood')
plt.ylabel('Number of Apartments')
plt.show()

print("")
temp_df = processed_df[processed_df['neighborhood'].apply(lambda x: x in top5_neighborhood)]

# plot boxplot to visualize distribution of price between each neighborhood
top5_boxplots = temp_df.loc[:,['neighborhood','price']].boxplot(by='neighborhood', column='price', figsize = (8,5))
top5_boxplots.plot()
plt.title("Boxplots of Rental Property Prices in 5 Most Common Neighborhoods")
plt.xlabel("Neighborhood")
plt.ylabel("Price ($)")
plt.suptitle("")
plt.show()


# I can also take a look at the five neighborhoods with the least amount of listed apartments
# start at '-5' to get 5 least occurring neighborhoods, sort_values to reverse list
bottom5_neighborhoods = top_neighborhoods[-5::].sort_values()
bottom5_neighborhoods

bottom5_neighborhood = bottom5_neighborhoods.index
bottom5_count = bottom5_neighborhoods.values

# plot figure
plt.figure(figsize=(8,4))
plt.bar(bottom5_neighborhood, bottom5_count)
plt.title('5 Least Common Neighborhoods in NYC Apartment Data')
plt.xlabel('Neighborhood')
plt.ylabel('Number of Apartments')
plt.show()

print("Corr between # of beds and price: ", stats.pearsonr(processed_df['total rooms'], processed_df['latitude']))
print("Corr between # of baths and price: ", stats.pearsonr(processed_df['baths'], processed_df['price']))
print("Corr between # of total rooms and price: ", stats.pearsonr(processed_df['total rooms'], processed_df['price']))

sample_one = processed_df[processed_df['neighborhood']=="East Village"]
sample_two = processed_df[processed_df['neighborhood']=="Murray Hill"]

stats.ttest_ind(sample_one['beds'], sample_two['beds'])






##TravelTime Function Implementation

# import necessary libraries for function implementation, and later isochrone/intersection visualization
import asyncio
from datetime import datetime
from traveltimepy import Driving, PublicTransport, Walking, Cycling, Coordinates, TravelTimeSdk
import geopandas as gpd
from shapely.geometry import Polygon, shape
import matplotlib.pyplot as plt

# these functions were sourced from the official TravelTime API, comments included to help explain the purpose of each

# function geocode_address: takes an address as an input, returns latitude and longitude values based on given query
async def geocode_address(address):
    results = await sdk.geocoding_async(query=address, limit=1, format_name=True)
    if results.features:
        point = shape(results.features[0].geometry)
        longitude = point.x
        latitude = point.y
        return Coordinates(lat=latitude, lng=longitude)
    else:
        print("No valid coordinates found.")
        return None

# function fetch_isochrones: takes coordinates, user inputted arrival time, travel time, and transportation
# returns isochrone
async def fetch_isochrones(coordinates, arrival_time, travel_time, transportation):
    results = await sdk.time_map_async(
        coordinates=coordinates,
        arrival_time=arrival_time,
        travel_time=travel_time,
        transportation=transportation
    )
    return results

# function find_intersection_or_union: takes coordinates and mode (intersection or union) as an input
# for the purpose of my project, I will only be using intersection
# returns intersection of isochrones
async def find_intersection_or_union(coordinates, mode):
    arrival_time = datetime.now()
    if mode == "intersection":
        return await sdk.intersection_async(
            coordinates=coordinates,
            arrival_time=arrival_time,
            transportation=Driving()
        )
    elif mode == "union":
        return await sdk.union_async(
            coordinates=coordinates,
            arrival_time=arrival_time,
            transportation=Driving()
        )

# function fetch_distance_map: takes coordinates, arrival time, travel distance, and transportation information into account
async def fetch_distance_map(coordinates, arrival_time, travel_distance, transportation):
    results = await sdk.distance_map_async(
        coordinates=coordinates,
        arrival_time=arrival_time,
        travel_distance=travel_distance,
        transportation=transportation
    )
    return results





## Isochrone and Intersection Mapping

async def main():

    # declare variables and data fields
    sdk = TravelTimeSdk("981c6f8d", "c5fffce90d3ee3c78d1aa2f07ca93be3")
    global origin_coordinates
    global transportation
    global travel_time_seconds

    # initialize array containing origin coordinates
    origin_coordinates = []

    # dictionary containing different modes of transportation
    transportation_options = {
    'walking': Walking,
    'cycling': Cycling,
    'driving': Driving,
    'public_transport': PublicTransport
    }

    # declare nyc_map, a folium Map of NYC
    nyc_map = folium.Map(location=[40.7415696, -74.0022074], zoom_start=13, tiles='CartoDB dark_matter', control_scale=True)

    # user input for total preferred travel time. in my instance, I inputted 30
    travel_time_minutes = input("Enter the total travel time you woud like in minutes: ")
    try:
        travel_time_seconds = int(travel_time_minutes) * 60 # convert to seconds for use in TravelTime functions
    except ValueError: # catch potential error, print error, set travel time to 1200 (20 min)
        print("Invalid input for travel time. Using a default of 20 minutes.")
        travel_time_seconds = 1200

    # next ask user input for mode of transportation
    print("Choose a method of transportation: walking, cycling, driving, public_transport")
    transport_method = input("Enter your choice: ").lower()
    # if statement to check for valid transportation method. if invalid, set to public_transport
    if transport_method in transportation_options:
        transportation = transportation_options[transport_method]()
    else:
        print("Invalid transportation method selected. Defaulting to public transport.")
        transportation = PublicTransport()

    # for loop that iterates through a maximum of 5 possible addresses
    for i in range(5):
        global user_address
        user_address = input(f"Please enter address #{i+1} (up to 5): ")
        user_coordinates = await geocode_address(user_address) # use geocode_address to get user_coordinates
        # if statement to check that coordinates are valid, in which case add to nyc_map
        if user_coordinates:
            origin_coordinates.append(Coordinates(lat=user_coordinates.lat, lng=user_coordinates.lng))
            nyc_map.location = [user_coordinates.lat, user_coordinates.lng]
            popup_content = f'<div style="width:200px;">{user_address}</div>'
            folium.Marker(
                [user_coordinates.lat, user_coordinates.lng],
                popup=folium.Popup(popup_content, max_width=265),
                icon=folium.Icon(color='blue')
            ).add_to(nyc_map)
        # if statement that iterates to ask if user would like to continue inputting more addresses, break loop if 'no'
        if i < 4:
            add_more = input("Would you like to input another address? (yes/no): ")
            if add_more.lower() != 'yes':
                break

    # fetch the isochrones for the center coordinates
    results = await sdk.time_map_geojson_async(
        coordinates=origin_coordinates,
        arrival_time=datetime.now(),
        travel_time= travel_time_seconds,
        transportation=transportation
    )

    # create list of colors that isochrone will use in visualization
    colors = ['red', 'blue', 'green', 'yellow', 'purple']

    # for loop that plots each isochrone on the map
    for i, feature in enumerate(results.features):
        folium.GeoJson(feature, style_function=lambda x, color=colors[i]: {'color': color}).add_to(nyc_map)

    # display folium map
    display(nyc_map)

# try&except block, this is a precaution as the TravelTime API has a max call amount per minute with the free plan I am using
try:
    await main()
except RuntimeError:
    asyncio.run(main())


import asyncio
from datetime import datetime
import folium
from traveltimepy import TravelTimeSdk, Coordinates, PublicTransport
from shapely.geometry import shape
from shapely.ops import unary_union

# function fetch_isochrones: takes coordinates, user inputted arrival time, travel time, and transportation
# returns isochrone
# this function provides the intersected area of all isochrones developed in the above graph
async def fetch_isochrones(origin_coordinates):
    sdk = TravelTimeSdk("981c6f8d", "c5fffce90d3ee3c78d1aa2f07ca93be3")

    # fetch the isochrones for the center coordinates
    features = await sdk.time_map_geojson_async(
        coordinates=origin_coordinates,
        arrival_time=datetime.now(),
        travel_time=travel_time_seconds,
        transportation=transportation
    )

    geoms = [shape(feature.geometry) for feature in features.features]

    # compute the intersection of all geometries
    intersection = unary_union(geoms)
    for geom in geoms:
        intersection = intersection.intersection(geom)

    # create a map using folium of New York
    m = folium.Map(location=[40.7415696, -74.0022074], zoom_start=13, tiles='CartoDB dark_matter')

    # plot the intersection if it's not empty
    if not intersection.is_empty:
        folium.GeoJson(intersection, style_function=lambda x: {'color': 'orange', 'weight': 5}).add_to(m)

    return m

# execute the async function and display the map in a cell
map_obj = await fetch_isochrones(origin_coordinates)
map_obj



import asyncio
from datetime import datetime
import folium
from folium.plugins import MarkerCluster
from traveltimepy import TravelTimeSdk, Coordinates, PublicTransport
from shapely.geometry import shape, Point
from shapely.ops import unary_union
import geopandas as gpd

# function fetch_isochrones: takes the intersected isochrone and the housing dataFrame
# returns a map of plotted apartments within the intersected isochrone
# this function is different from that above as it provides the housing data overtop the area
async def fetch_isochrones():
    global intersection
    sdk = TravelTimeSdk("981c6f8d", "c5fffce90d3ee3c78d1aa2f07ca93be3")

    # Fetch the isochrones
    features = await sdk.time_map_geojson_async(
        coordinates=origin_coordinates,
        arrival_time=datetime.now(),
        travel_time=travel_time_seconds,
        transportation=transportation
    )

    geoms = [shape(feature.geometry) for feature in features.features]

    # calculate the intersection of all geometries
    # if statement for geoms
    if geoms:
        intersection = geoms[0]  # Start with the first geometry
        for geom in geoms[1:]:
            intersection = intersection.intersection(geom)  # Intersect with each subsequent geometry

    m = folium.Map(location=[40.7415696, -74.0022074], zoom_start=13, tiles='CartoDB dark_matter')
    if intersection and not intersection.is_empty:
        folium.GeoJson(intersection, style_function=lambda x: {'color': 'orange', 'weight': 5}).add_to(m)
    return m

async def main():
    map_obj = await fetch_isochrones()

    marker_cluster = MarkerCluster().add_to(map_obj)

    # create a temporary geodataframe gdf that contains the geometry information
    gdf = gpd.GeoDataFrame(processed_df, geometry=gpd.points_from_xy(processed_df['longitude'], processed_df['latitude']), crs='EPSG:4326')
    # create column 'geometry' that stores the points derived from longitude and latitude
    gdf['geometry'] = gdf.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)

    # filter gdf so that it only includes data within intersection
    gdf_within_intersection = gdf[gdf.geometry.within(intersection)]
    global intersected_df

    # create a dataframe intersected_df, dropping the 'geometry' column and resetting index
    intersected_df = pd.DataFrame(gdf_within_intersection.drop(columns='geometry')).reset_index(drop=True)

    # create a marker based on lat and long values for folium map, include pop-up which includes address information
    for idx, row in gdf_within_intersection.iterrows():
        folium.Marker([row['latitude'], row['longitude']], popup=row['address']).add_to(map_obj)
    # return map
    return map_obj

# run the script and display the map
map_obj = await main()
map_obj








#Isochrone Data Analysis

print(intersected_df.shape)
# print shape of intersected_df, as well as price descriptive statistics
print("Price Descriptive Statistics")
print(f"   Mean:   ${round(intersected_df['price'].mean(), 2)}")
print(f"   Median: ${round(intersected_df['price'].median(), 2)}")
print(f"   Mode:   ${round(intersected_df['price'].mode()[0], 2)}")
print(f"   St Dev: ${round(intersected_df['price'].std(), 2)}")
print()

# rescale the grade column to be proportional to the applicable area
intersected_df['room price ratio'] = intersected_df['total rooms'] / intersected_df['price']
scaler = skp.MinMaxScaler(feature_range=(1, 10))
intersected_df['grade'] = scaler.fit_transform(intersected_df[['room price ratio']])
intersected_df.drop('room price ratio', axis=1, inplace=True)

# count top 5 neighborhoods in intersected_df
top_neighborhoods_intersection = intersected_df['neighborhood'].value_counts()
top5_neighborhood_intersection = top_neighborhoods_intersection[0:5]

top5_neighborhood = top5_neighborhood_intersection.index
top5_count = top5_neighborhood_intersection.values

# plot findings in barchart
plt.figure(figsize=(8,4))
plt.bar(top5_neighborhood, top5_count)
plt.title('Top 5 Most Common Neighborhoods Within Isochrone')
plt.xlabel('Neighborhood')
plt.ylabel('Number of Apartments')
plt.show()

print(f'\nOut of the {intersected_df.shape[0]} apartments in the isochrone, here are the top 5 most commonly appearing neighborhoods.\n')

# create a temporary dataframe that only takes rows from top5 neighborhoods in intersected_df
temp_df = intersected_df[intersected_df['neighborhood'].apply(lambda x: x in top5_neighborhood)]

# plot boxplots that depict distribution of price between each neighborhood
top5_boxplots = temp_df.loc[:,['neighborhood','price']].boxplot(by='neighborhood', column='price', figsize = (8,5))
top5_boxplots.plot()
plt.title("Boxplots of Rental Property Prices in 5 Most Common Neighborhoods of Isochrone")
plt.xlabel("Neighborhood")
plt.ylabel("Price ($)")
plt.suptitle("")
plt.show()
print("")
# plot boxplots for grade
grade_df = temp_df[['neighborhood', 'beds', 'grade']]
grade_df['beds'] = grade_df['beds'].astype(str) + ' bed'
plt.figure(figsize=(12, 6))  # Set the figure size or adjust based on your dataset
sns.boxplot(data=grade_df, x='neighborhood', y='grade', hue='beds')  # 'hue' differentiates the bedroom counts
plt.title("Boxplots of Rental Property Grades by Neighborhood and Number of Bedrooms")
plt.xlabel("Neighborhood")
plt.ylabel("Grade")
plt.legend(title='Bedrooms', loc='upper right')
plt.xticks(rotation=45)
plt.show()
top5_neighborhood_intersection


from scipy import stats
from sklearn.model_selection import train_test_split

# create columns that normalize the numeric data in my intersected_df
intersected_df['price_norm']=skp.MinMaxScaler().fit_transform(intersected_df['price'].values.reshape(-1,1))
intersected_df['beds_norm']=skp.MinMaxScaler().fit_transform(intersected_df['beds'].values.reshape(-1,1))
intersected_df['baths_norm']=skp.MinMaxScaler().fit_transform(intersected_df['baths'].values.reshape(-1,1))
intersected_df['total_rooms_norm']=skp.MinMaxScaler().fit_transform(intersected_df['total rooms'].values.reshape(-1,1))

# print correlation data between X (beds, baths, total rooms) and Y (price)
print("Corr between # of beds and price: ", stats.pearsonr(intersected_df['beds'], intersected_df['price']))
print("Corr between # of baths and price: ", stats.pearsonr(intersected_df['baths'], intersected_df['price']))
print("Corr between # of total rooms and price: ", stats.pearsonr(intersected_df['total rooms'], intersected_df['price']))



# data for regression of price and total rooms

# initialize X (normalized bed and bath counts) and Y (normalized price)
X = intersected_df[['beds_norm', 'baths_norm']]
y = intersected_df['price_norm']

# use train_test_split to get X_train/_test and Y_train/_test data splits
X_train, X_test, Y_train, Y_test = train_test_split(X,y)

# import necessary libraries
from sklearn import linear_model
import sklearn.metrics as m

# create linear regression model
my_model = linear_model.LinearRegression()
my_model.fit(X_train, Y_train)


# predict on the test data
Y_hat = my_model.predict(X_test)

# calculate and print metrics
mse = m.mean_squared_error(Y_test, Y_hat)
r2 = m.r2_score(Y_test, Y_hat)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)


import numpy as np
from sklearn.model_selection import learning_curve

# conduct an evaluation of the model, focusing on over and underfitting
train_sizes, train_scores, validation_scores = learning_curve(my_model, X, y, train_sizes= np.linspace(.1,1,10), scoring='neg_mean_squared_error')

# store values
train_mean_score = -np.mean(train_scores, axis=1)
train_std_score = np.std(train_scores, axis=1)
validation_mean_score = -np.mean(validation_scores, axis=1)
validation_std_score = np.std(validation_scores, axis=1)

# plot learning curve
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean_score, label='Training error')
plt.plot(train_sizes, validation_mean_score, label='Validation error')
plt.title('Learning Curve')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.show()



features = intersected_df[['beds', 'baths', 'price']]  # these are numeric
scaler = skp.StandardScaler()

# scale the features
features_scaled = scaler.fit_transform(features)
# import library to use KMeans
from sklearn.cluster import KMeans
# determine the optimal number of clusters using the Elbow method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    kmeans.fit(features_scaled)
    sse.append(kmeans.inertia_)

# plot SSE for each k to find the elbow
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.show()

# assuming the elbow is at k=4 from the plot
kmeans = KMeans(n_clusters=4, n_init=10, random_state=0)
intersected_df['Cluster'] = kmeans.fit_predict(features_scaled)

# calculate mean values for each cluster
clustered_data = intersected_df[['beds', 'baths', 'price', 'Cluster']].groupby('Cluster').mean()
print(clustered_data)



def summarize_clusters(intersected_df):
    cluster_summary = intersected_df.groupby('Cluster').agg({
        'beds': ['mean', 'median', 'std'],
        'baths': ['mean', 'median', 'std'],
        'price': ['mean', 'median', 'std', 'min', 'max']
    })
    return cluster_summary

def plot_cluster_relationships(intersected_df):
    sns.pairplot(intersected_df, vars=['beds', 'baths', 'price'], hue='Cluster', palette='viridis', plot_kws={'alpha': 0.6})
    plt.show()

def interpret_cluster(row):
    beds, baths, price = row['beds', 'mean'], row['baths', 'mean'], row['price', 'mean']
    # High-end Luxury
    if price > 40000:
        return "This cluster represents luxury properties, likely in prestigious areas, with spacious and high-quality interiors."
    # Family-sized Homes
    elif beds > 3 and baths > 2:
        return "This cluster includes large family homes with multiple bedrooms and bathrooms, suitable for family living and offering ample space for comfort."
    # Upscale Apartments
    elif beds > 2 and baths >= 2 and price > 10000:
        return "This cluster likely consists of upscale apartments or townhouses with excellent amenities and considerable living space, appealing to professionals or small families."
    # Mid-range Apartments
    elif beds >= 2 and baths >= 1 and 6000 < price and price < 10000:
        return "This cluster represents mid-range properties, providing a good balance of affordability and space, ideal for couples or small families."
    # Budget, Smaller Apartments
    elif beds < 2 and price < 7000:
        return "This cluster likely includes budget-friendly, smaller apartments or studios, ideal for singles or young couples starting out."
    # Default catch-all for other cases
    else:
        return "This cluster includes properties that might not fit neatly into other categories, offering unique living arrangements."

def generate_narrative(cluster_summary):
    narratives = []
    for cluster in cluster_summary.index:
        narrative = (
            f"Cluster {cluster}: Average of {cluster_summary.loc[cluster, ('beds', 'mean')]:.2f} beds and "
            f"{cluster_summary.loc[cluster, ('baths', 'mean')]:.2f} baths at a price of "
            f"${cluster_summary.loc[cluster, ('price', 'mean')]:,.2f}. "
            f"\nInterpretation: {interpret_cluster(cluster_summary.loc[cluster])}"
        )
        narratives.append(narrative)
    return "\n\n".join(narratives)


cluster_stats = summarize_clusters(intersected_df)
plot_cluster_relationships(intersected_df)
cluster_narratives = generate_narrative(cluster_stats)
print(cluster_narratives)