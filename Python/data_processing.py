import musicbrainzngs
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import json
import requests
import os
import time
import torch

# Set up your user agent for MusicBrainz API
musicbrainzngs.set_useragent("Project-Playlist", "1.0", "connorxspears@gmail.com")

# Load the generalized genres list from a JSON file
json_file_path = os.path.abspath('../genre_data.json')

with open(json_file_path, 'r') as genre_json:
    gen_genres = json.load(genre_json)

# Extract genre names and reshape them for OneHotEncoder
genre_names = [genre["name"] for genre in gen_genres["genres"]]
genres_reshaped = np.array(genre_names).reshape(-1, 1)

# Initialize and fit the OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(genres_reshaped)

# Function to fetch song data from MusicBrainz
def fetch_acousticbrainz_data(song_id, isLow, retries=5):
    if(isLow):
        url = f"https://acousticbrainz.org/{song_id}/low-level"
    else:
        url = f"https://acousticbrainz.org/{song_id}/high-level"
    
    for attempt in range(retries):
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()  # Return JSON data if request is successful
        elif response.status_code == 429:
            print("Rate limit exceeded. Waiting to retry...")
            time.sleep(2 ** attempt)  # Exponential backoff: wait longer with each retry
        else:
            print("Error fetching song data:", response.status_code)
            return None
    
    print("Failed to retrieve data after multiple attempts.")
    return None

# Function to map API genres to generalized genres
def map_genre_to_general(api_genre, general_genres):
    api_genre = api_genre.lower().strip()
    
    for general_genre in general_genres:
        if general_genre.lower() in api_genre or api_genre in general_genre.lower():
            print(general_genre)
            return general_genre
    return None

def normalize(value, min_val, max_val):
    return(value - min_val) / (max_val - min_val) if value is not None else 0.5

# Example song ID
song_id = "b849acd4-0638-49ea-8e40-7391613d4890"
acoustic_data_low = fetch_acousticbrainz_data(song_id, True)
acoustic_data_high = fetch_acousticbrainz_data(song_id, False)

if acoustic_data_low and acoustic_data_high:
    print("Song:", acoustic_data_high["metadata"].get("tags").get("title"))
    print("Artist:", acoustic_data_high["metadata"].get("tags").get("artist"))
    print("\nValues which will be passed as input to Neural Network: ")

    # Extract and normalize features
    input_data = {
        'key': acoustic_data_low["tonal"].get('key_key'),  # Categorical
        'key_scale': acoustic_data_low['tonal'].get('key_scale'),  # Categorical
        'bpm': normalize(acoustic_data_low["rhythm"].get('bpm'), 40, 200),  # Assuming bpm range
        'loudness': acoustic_data_low["lowlevel"].get('average_loudness'),
        'aggressive': acoustic_data_high['highlevel'].get('mood_aggressive').get('all').get('aggressive'),
        'acoustic': acoustic_data_high['highlevel'].get('mood_acoustic').get('all').get('acoustic'),
        'danceable': acoustic_data_high["highlevel"].get('danceability').get('all').get('danceable'),
        'timbre': acoustic_data_high["highlevel"].get('timbre').get('all').get('bright')
    }

    print(f"Key: {input_data['key']}")
    print(f"Key Scale: {input_data['key_scale']}")
    print(f"BPM (normalized): {input_data['bpm']:.2f}")
    print(f"Loudness (normalized): {input_data['loudness']:.2f}")
    print(f"Aggressiveness (normalized): {input_data['aggressive']:.2f}")
    print(f"Acousticness (normalized): {input_data['acoustic']:.2f}")
    print(f"Danceability (normalized): {input_data['danceable']:.2f}")
    print(f"Timbre (normalized): {input_data['timbre']:.2f}")

    categorical_features = [[input_data['key'], input_data['key_scale']]]
    categorical_encoded = encoder.fit_transform(categorical_features)

    continuous_features = [input_data['bpm'], input_data['loudness'], input_data['aggressive'],
                           input_data['acoustic'], input_data['danceable'], input_data['timbre']]
    input_features = np.concatenate([categorical_encoded[0], continuous_features])

    np.save("processed_data.npy", input_features)
    print("Data saved to processed_data.npy")
else:
    print("Failed to retrieve song data.")
