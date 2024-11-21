from flask import Flask, request, json
from startingNN import GenreNN
import torch
import musicbrainzngs as mb
import requests
import numpy as np
import joblib
import os

app = Flask(__name__)

def predict():
    songName = request.get_json()
    getMBID(songName)

def getMBID(title, artist):
    try:
        # Query MusicBrainz for songs by genre
        result = mb.search_recordings(
            query=f'recording:"{title}" artist:"{artist}"', 
            limit=10,
            )
        
        if not result.get('recording-list'):
            print(f"No recordings found for title '{title}'.")
            return None
        
        non_live_recordings = [
            recording for recording in result['recording-list']
            if recording.get('type') != 'Live'
        ]

        if not non_live_recordings:
            print(f"No non-live recordings found for title '{title}'")
            return None
        
        for recording in non_live_recordings:
            mbid = recording['id']
            print(f"Found recording: {recording['title']} (MBID: {mbid})")

            high_level_data = fetch_acousticbrainz_data(mbid, high_quality=True)
            if high_level_data:
                low_level_data = fetch_acousticbrainz_data(mbid, high_quality=False)
                if low_level_data:
                    print(f"Successfully fetched AcousticBrainz data for MBID {mbid}")
                    print(f"Artist:", recording['artist-credit'][0]['artist']['name'], "Song:", recording['title'])
                    parsed_data = parse_data(high_level_data, low_level_data)
                    return parsed_data
                else:
                    print(f"Song ID: {mbid} had high level data but not low level")
            else:
                print(f"Song ID: {mbid} did not have high level data")
        raise Exception()
    except Exception as e:
        print(f"Error searching for song or fetching data: {e}")
        return None

def fetch_acousticbrainz_data(mbid, high_quality):
    quality = "high-level" if high_quality else "low-level"
    url = f"http://acousticbrainz.org/{mbid}/{quality}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        print(f"Error fetching AcousticBrainz {quality} data: {e}")
        return None

def parse_data(high_level_data, low_level_data):
    encoder_key_key = joblib.load("encoder_key_key.pkl")
    encoder_key_scale = joblib.load("encoder_key_scale.pkl")
    
    # Encode the categorical features with the fitted encoder
    categorical_features = np.concatenate([
        encoder_key_key.transform([[low_level_data["tonal"].get("key_key")]])[0],
        encoder_key_scale.transform([[low_level_data["tonal"].get("key_scale")]])[0]
    ])
    
    # Normalize and gather continuous features
    continuous_features = np.array([
        normalize(low_level_data["rhythm"].get("bpm"), 40, 200),
        low_level_data["lowlevel"].get("average_loudness"),
        high_level_data['highlevel'].get('mood_aggressive').get('all').get('aggressive'),
        high_level_data['highlevel'].get('mood_acoustic').get('all').get('acoustic'),
        high_level_data["highlevel"].get('danceability').get('all').get('danceable'),
        high_level_data["highlevel"].get('timbre').get('all').get('bright')
    ]).reshape(1, -1)
    continuous_features = continuous_features.flatten()

    try:
        input_features = np.concatenate([categorical_features, continuous_features])
        return input_features
    except ValueError as e:
        print(f"Error concatenating: {e}")
        print(f"Categorical shape: {categorical_features.shape}")
        print(f"Continuous shape: {continuous_features.shape}")
        return None

def normalize(value, min_val, max_val):
    return(value - min_val) / (max_val - min_val) if value is not None else 0.5

# Set up user agent for MusicBrainz API
mb.set_useragent("Project-Playlist", "1.0", "connorxspears@gmail.com")

songName = input("Enter the song name:")
songArtist = input(f"Enter the artist for {songName}:")

songData = getMBID(songName, songArtist)
model = GenreNN(20, 64, 50) # 20 inputs for 12 musical keys + 2 scales + 8 continuous features
model.load_state_dict(torch.load('GenreClassifier.pth'))
model.eval()

json_file_path = os.path.abspath('../genre_data.json')
with open(json_file_path, 'r') as genre_json:
    genres = json.load(genre_json)
genre_names = [genre["name"] for genre in genres["genres"]]

with torch.no_grad():
    songData = torch.tensor(songData, dtype=torch.float32)
    songData = songData.unsqueeze(0)
    output = model(songData)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    print("Probabilities:", probabilities)
    top3_indices = torch.topk(probabilities, 3).indices.squeeze().tolist()
    top3_genres = [(genre_names[i], probabilities[0, i].item()) for i in top3_indices]
    print(f"Top 3 predicted genres: {top3_genres}")