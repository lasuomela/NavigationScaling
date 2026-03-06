import pandas as pd
from geopy.geocoders import Nominatim
from tqdm import tqdm
import time

# Load your CSV file
df = pd.read_csv("ride_cluster_stats.csv")

val_clusters = pd.read_csv("val_rides_per_location.csv")["cluster"]
train_clusters = pd.read_csv("train_rides_per_location.csv")["cluster"]

# Filter the dataframe to only include rows with clusters in val or train sets
df = df[df["cluster"].isin(val_clusters) | df["cluster"].isin(train_clusters)]
geolocator = Nominatim(user_agent="country_lookup")

countries = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    try:
        location = geolocator.reverse((row["centroid_lat"], row["centroid_lon"]), language="en")
        country = location.raw["address"].get("country")
    except Exception:
        country = None
    countries.append(country)
    time.sleep(0.5)  # be polite to the API (Nominatim has strict rate limits)

df["country"] = countries
unique_countries = df["country"].dropna().unique()
print(f"Number of unique countries: {len(unique_countries)}")
print(unique_countries)