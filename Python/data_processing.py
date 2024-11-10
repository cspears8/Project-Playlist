import musicbrainzngs
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import json

# Set up your user agent for MusicBrainz API
musicbrainzngs.set_useragent("Project-Playlist", "1.0", "connorxspears@gmail.com")

def normalize(value, min_val, max_val):
    return(value - min_val) / (max_val - min_val) if value is not None else 0.5

def parseData():
    encoder = OneHotEncoder(sparse_output=False)
    
    with(open('low_level_data.json', 'r') as low_level):
        lowData = json.load(low_level)
    with(open('high_level_data.json', 'r') as high_level):
        highData = json.load(high_level)

    all_song_features = []

    for genre, songs in lowData.items():
        for song in songs:
            mbid = song["mbid"]
            lowlevel_data = song["data"]
            highlevel_data = next((item["data"] for item in highData.get(genre, []) if item["mbid"] == mbid), {})

            categorical_features = [[
                lowlevel_data["tonal"].get("key_key"),
                lowlevel_data["tonal"].get("key_scale")
            ]]
            categorical_encoded = encoder.fit_transform(categorical_features)[0]
            
            continuous_features = [
                normalize(lowlevel_data["rhythm"].get("bpm"), 40, 200),
                lowlevel_data["lowlevel"].get("average_loudness"),
                highlevel_data['highlevel'].get('mood_aggressive').get('all').get('aggressive'),
                highlevel_data['highlevel'].get('mood_acoustic').get('all').get('acoustic'),
                highlevel_data["highlevel"].get('danceability').get('all').get('danceable'),
                highlevel_data["highlevel"].get('timbre').get('all').get('bright')
            ]

            input_features = np.concatenate([categorical_encoded, continuous_features])
            
            all_song_features.append(input_features)

        print(f"Genre {genre} complete!")
    all_song_features_array = np.array(all_song_features)

    np.save("all_songs_data.npy", all_song_features_array)
    print(f"Filtered data saved to low_level_data_parsed.json")

parseData()