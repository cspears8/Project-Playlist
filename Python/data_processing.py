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

# # Load the generalized genres list from a JSON file
# json_file_path = os.path.abspath('../genre_data.json')

# with open(json_file_path, 'r') as genre_json:
#     gen_genres = json.load(genre_json)

# # Extract genre names and reshape them for OneHotEncoder
# genre_names = [genre["name"] for genre in gen_genres["genres"]]
# genres_reshaped = np.array(genre_names).reshape(-1, 1)

# # Initialize and fit the OneHotEncoder
# encoder = OneHotEncoder(sparse_output=False)
# encoder.fit(genres_reshaped)

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

def parseData():
    with(open('low_level_data.json', 'r') as low_level):
        lowData = json.load(low_level)
    with(open('high_level_data.json', 'r') as high_level):
        highData = json.load(high_level)

    filtered_data = {}

    for genre, songs in lowData.items():
        filtered_data[genre] = []
        for song in songs:
            mbid = song["mbid"]
            lowlevel_data = song["data"]
            highlevel_data = next((item["data"] for item in highData.get(genre, []) if item["mbid"] == mbid), {})

            # Define the desired traits and explicitly specify the nested paths
            filtered_song_data = {
                "key_key": lowlevel_data["tonal"].get("key_key"),
                "key_scale": lowlevel_data["tonal"].get("key_scale"),
                "bpm": lowlevel_data["rhythm"].get("bpm"),
                "average_loudness": lowlevel_data["lowlevel"].get("average_loudness"),
                'aggressive': highlevel_data['highlevel'].get('mood_aggressive').get('all').get('aggressive'),
                'acoustic': highlevel_data['highlevel'].get('mood_acoustic').get('all').get('acoustic'),
                'danceable': highlevel_data["highlevel"].get('danceability').get('all').get('danceable'),
                'timbre': highlevel_data["highlevel"].get('timbre').get('all').get('bright')
            }

            filtered_data[genre].append({
                "mbid": mbid,
                "data": filtered_song_data
            })
        print(f"Genre {genre} complete!")
    with open('parsed_data.json', 'w') as outfile:
        json.dump(filtered_data, outfile, indent=4)
    print(f"Filtered data saved to low_level_data_parsed.json")

parseData()
# with open('genre_songs_ids.json', 'r') as mbid_file:
#     mbids = json.load(mbid_file)

# for genre in mbids:
#     cur_genre_list = mbids.get(genre)
#     for song in cur_genre_list:
#         acoustic_data_high = fetch_acousticbrainz_data(song, True)
#         acoustic_data_low = fetch_acousticbrainz_data(song, False)

#         if acoustic_data_high and acoustic_data_low:
#             # Extract and normalize features
#             input_data = {
#                 'key': acoustic_data_low["tonal"].get('key_key'),  # Categorical
#                 'key_scale': acoustic_data_low['tonal'].get('key_scale'),  # Categorical
#                 'bpm': normalize(acoustic_data_low["rhythm"].get('bpm'), 40, 200),  # Assuming bpm range
#                 'loudness': acoustic_data_low["lowlevel"].get('average_loudness'),
#                 'aggressive': acoustic_data_high['highlevel'].get('mood_aggressive').get('all').get('aggressive'),
#                 'acoustic': acoustic_data_high['highlevel'].get('mood_acoustic').get('all').get('acoustic'),
#                 'danceable': acoustic_data_high["highlevel"].get('danceability').get('all').get('danceable'),
#                 'timbre': acoustic_data_high["highlevel"].get('timbre').get('all').get('bright')
#             }

#             categorical_features = [[input_data['key'], input_data['key_scale']]]
#             categorical_encoded = encoder.fit_transform(categorical_features)

#             continuous_features = [input_data['bpm'], input_data['loudness'], input_data['aggressive'],
#                                 input_data['acoustic'], input_data['danceable'], input_data['timbre']]
#             input_features = np.concatenate([categorical_encoded[0], continuous_features])

#             np.save("processed_data.npy", input_features)
#             print("Data for song", acoustic_data_high[] saved to processed_data.npy")