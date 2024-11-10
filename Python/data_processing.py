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
    
    # Load data
    with open('low_level_data.json', 'r') as low_level:
        lowData = json.load(low_level)
    with open('high_level_data.json', 'r') as high_level:
        highData = json.load(high_level)

    all_song_features = []
    all_song_labels = []

    # Collect all possible values of key_key and key_scale for one-hot encoding
    key_key_values = []
    key_scale_values = []

    for genre, songs in lowData.items():
        for song in songs:
            lowlevel_data = song["data"]

            # Collect all values for key_key and key_scale
            key_key_values.append(lowlevel_data["tonal"].get("key_key"))
            key_scale_values.append(lowlevel_data["tonal"].get("key_scale"))

    # Fit the encoder once
    encoder_key_key = OneHotEncoder(sparse_output=False)
    encoder_key_scale = OneHotEncoder(sparse_output=False)

    # Fit the encoder on all the categorical values
    encoder_key_key.fit(np.array(key_key_values).reshape(-1, 1))
    encoder_key_scale.fit(np.array(key_scale_values).reshape(-1, 1))

    # Now process the songs
    for genre, songs in lowData.items():
        for song in songs:
            lowlevel_data = song["data"]
            highlevel_data = next((item["data"] for item in highData.get(genre, []) if item["mbid"] == song["mbid"]), {})

            # Encode the categorical features with the fitted encoder
            categorical_features = np.concatenate([
                encoder_key_key.transform([[lowlevel_data["tonal"].get("key_key")]])[0],
                encoder_key_scale.transform([[lowlevel_data["tonal"].get("key_scale")]])[0]
            ])

            # Normalize and gather continuous features
            continuous_features = np.array([
                normalize(lowlevel_data["rhythm"].get("bpm"), 40, 200),
                lowlevel_data["lowlevel"].get("average_loudness"),
                highlevel_data['highlevel'].get('mood_aggressive').get('all').get('aggressive'),
                highlevel_data['highlevel'].get('mood_acoustic').get('all').get('acoustic'),
                highlevel_data["highlevel"].get('danceability').get('all').get('danceable'),
                highlevel_data["highlevel"].get('timbre').get('all').get('bright')
            ]).reshape(1, -1)
            continuous_features = continuous_features.flatten()
            
            try:
                input_features = np.concatenate([categorical_features, continuous_features])
                all_song_features.append(input_features)
            except ValueError as e:
                print(f"Error concatenating: {e}")
                print(f"Categorical shape: {categorical_features.shape}")
                print(f"Continuous shape: {continuous_features.shape}")

            all_song_labels.append(genre)

        print(f"Genre {genre} complete!")

    # Convert to numpy arrays
    all_song_features_array = np.array(all_song_features)
    all_song_labels_array = np.array(all_song_labels)

    # Encode labels (genres)
    genre_encoder = OneHotEncoder(sparse_output=False)
    all_song_labels_encoded = genre_encoder.fit_transform(all_song_labels_array.reshape(-1, 1))

    np.save("all_songs_data.npy", all_song_features_array)
    np.save("all_song_labels.npy", all_song_labels_encoded)
    
    print(f"Filtered data saved to all_songs_data.npy and all_song_labels.npy")

parseData()