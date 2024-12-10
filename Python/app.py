from flask import Flask, request, json
from startingNN import GenreNN
from sklearn.preprocessing import MinMaxScaler
import torch
import musicbrainzngs as mb
import requests
import numpy as np
import pandas as pd
import os
import joblib

app = Flask(__name__)

def predict():
    songName = request.get_json()
    getMBID(songName)

def getMBID(title, artist):
    try:
        # Query MusicBrainz for songs by genre
        result = mb.search_recordings(
            query=f'recording:"{title}" artist:"{artist}"', 
            limit=20,
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
                    parsed_data = parse_data(high_level_data, low_level_data, recording)
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

def parse_data(highlevel_data, lowlevel_data, recording):
    features_list = []
    song_id = recording['id']

    flat_entry = {"mbid": song_id, "genre": None}
    if lowlevel_data:
        lowlevel_data = lowlevel_data["lowlevel"]
        for key, value in lowlevel_data.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_entry[f"{key}_{subkey}"] = subvalue
            else:
                flat_entry[key] = value
    if highlevel_data:
        high_level_features = {
        # Danceability and Timbre
        "danceable": highlevel_data["highlevel"].get('danceability').get('all').get('danceable'),  # Danceability feature
        "bright_timbre": highlevel_data["highlevel"].get('timbre').get('all').get('bright'),  # Timbre: Brightness
        "instrumental": highlevel_data["highlevel"].get("voice_instrumental").get("all").get("instrumental"),  # Instrumental voice
        "atonal": highlevel_data["highlevel"].get("tonal_atonal").get("all").get("atonal"),  # Tonal vs Atonal
        "female_gender": highlevel_data["highlevel"].get("gender").get("all").get("female"),  # Gender (Female)

        # Genre Dortmund features
        "alternative_dortmund": highlevel_data["highlevel"].get("genre_dortmund").get("all").get("alternative"),  # Genre Dortmund: Alternative
        "blues_dortmund": highlevel_data["highlevel"].get("genre_dortmund").get("all").get("blues"),  # Genre Dortmund: Blues
        "electronic_dortmund": highlevel_data["highlevel"].get("genre_dortmund").get("all").get("electronic"),  # Genre Dortmund: Electronic
        "folkcountry_dortmund": highlevel_data["highlevel"].get("genre_dortmund").get("all").get("folkcountry"),  # Genre Dortmund: Folk/Country
        "funksoulrnb_dortmund": highlevel_data["highlevel"].get("genre_dortmund").get("all").get("funksoulrnb"),  # Genre Dortmund: Funk/Soul/RnB
        "jazz_dortmund": highlevel_data["highlevel"].get("genre_dortmund").get("all").get("jazz"),  # Genre Dortmund: Jazz
        "pop_dortmund": highlevel_data["highlevel"].get("genre_dortmund").get("all").get("pop"),  # Genre Dortmund: Pop
        "raphiphop_dortmund": highlevel_data["highlevel"].get("genre_dortmund").get("all").get("raphiphop"),  # Genre Dortmund: Rap/HipHop
        "rock_dortmund": highlevel_data["highlevel"].get("genre_dortmund").get("all").get("rock"),  # Genre Dortmund: Rock

        # Genre Electronic features
        "ambient_electronic": highlevel_data["highlevel"].get("genre_electronic").get("all").get("ambient"),  # Genre Electronic: Ambient
        "dnb_electronic": highlevel_data["highlevel"].get("genre_electronic").get("all").get("dnb"),  # Genre Electronic: DnB
        "house_electronic": highlevel_data["highlevel"].get("genre_electronic").get("all").get("house"),  # Genre Electronic: House
        "techno_electronic": highlevel_data["highlevel"].get("genre_electronic").get("all").get("techno"),  # Genre Electronic: Techno
        "trance_electronic": highlevel_data["highlevel"].get("genre_electronic").get("all").get("trance"),  # Genre Electronic: Trance

        # Genre Rosamerica features
        "cla_rosamerica": highlevel_data["highlevel"].get("genre_rosamerica").get("all").get("cla"),  # Genre Rosamerica: Classical
        "dan_rosamerica": highlevel_data["highlevel"].get("genre_rosamerica").get("all").get("dan"),  # Genre Rosamerica: Dance
        "hip_rosamerica": highlevel_data["highlevel"].get("genre_rosamerica").get("all").get("hip"),  # Genre Rosamerica: Hip Hop
        "jaz_rosamerica": highlevel_data["highlevel"].get("genre_rosamerica").get("all").get("jaz"),  # Genre Rosamerica: Jazz
        "pop_rosamerica": highlevel_data["highlevel"].get("genre_rosamerica").get("all").get("pop"),  # Genre Rosamerica: Pop
        "rhy_rosamerica": highlevel_data["highlevel"].get("genre_rosamerica").get("all").get("rhy"),  # Genre Rosamerica: Rhythm
        "roc_rosamerica": highlevel_data["highlevel"].get("genre_rosamerica").get("all").get("roc"),  # Genre Rosamerica: Rock
        "spe_rosamerica": highlevel_data["highlevel"].get("genre_rosamerica").get("all").get("spe"),  # Genre Rosamerica: Special

        # Genre Tzanetakis features
        "blu_tzanetakis": highlevel_data["highlevel"].get("genre_tzanetakis").get("all").get("blu"),  # Genre Tzanetakis: Blues
        "cla_tzanetakis": highlevel_data["highlevel"].get("genre_tzanetakis").get("all").get("cla"),  # Genre Tzanetakis: Classical
        "cou_tzanetakis": highlevel_data["highlevel"].get("genre_tzanetakis").get("all").get("cou"),  # Genre Tzanetakis: Country
        "dis_tzanetakis": highlevel_data["highlevel"].get("genre_tzanetakis").get("all").get("dis"),  # Genre Tzanetakis: Disco
        "hip_tzanetakis": highlevel_data["highlevel"].get("genre_tzanetakis").get("all").get("hip"),  # Genre Tzanetakis: Hip Hop
        "jaz_tzanetakis": highlevel_data["highlevel"].get("genre_tzanetakis").get("all").get("jaz"),  # Genre Tzanetakis: Jazz
        "met_tzanetakis": highlevel_data["highlevel"].get("genre_tzanetakis").get("all").get("met"),  # Genre Tzanetakis: Metal
        "pop_tzanetakis": highlevel_data["highlevel"].get("genre_tzanetakis").get("all").get("pop"),  # Genre Tzanetakis: Pop
        "reg_tzanetakis": highlevel_data["highlevel"].get("genre_tzanetakis").get("all").get("reg"),  # Genre Tzanetakis: Reggae
        "roc_tzanetakis": highlevel_data["highlevel"].get("genre_tzanetakis").get("all").get("roc"),  # Genre Tzanetakis: Rock

        # ISMIR04 Rhythm features
        "chachacha_ismir04": highlevel_data["highlevel"].get("ismir04_rhythm").get("all").get("ChaChaCha"),  # ISMIR04: ChaChaCha
        "jive_ismir04": highlevel_data["highlevel"].get("ismir04_rhythm").get("all").get("Jive"),  # ISMIR04: Jive
        "quickstep_ismir04": highlevel_data["highlevel"].get("ismir04_rhythm").get("all").get("Quickstep"),  # ISMIR04: Quickstep
        "rumba_american_ismir04": highlevel_data["highlevel"].get("ismir04_rhythm").get("all").get("Rumba-American"),  # ISMIR04: Rumba-American
        "rumba_international_ismir04": highlevel_data["highlevel"].get("ismir04_rhythm").get("all").get("Rumba-International"),  # ISMIR04: Rumba-International
        "rumba_misc_ismir04": highlevel_data["highlevel"].get("ismir04_rhythm").get("all").get("Rumba-Misc"),  # ISMIR04: Rumba-Misc
        "samba_ismir04": highlevel_data["highlevel"].get("ismir04_rhythm").get("all").get("Samba"),  # ISMIR04: Samba
        "tango_ismir04": highlevel_data["highlevel"].get("ismir04_rhythm").get("all").get("Tango"),  # ISMIR04: Tango
        "viennese_waltz_ismir04": highlevel_data["highlevel"].get("ismir04_rhythm").get("all").get("VienneseWaltz"),  # ISMIR04: Viennese Waltz
        "waltz_ismir04": highlevel_data["highlevel"].get("ismir04_rhythm").get("all").get("Waltz"),  # ISMIR04: Waltz

        # Mood features
        "acoustic_mood": highlevel_data['highlevel'].get('mood_acoustic').get('all').get('acoustic'),  # Mood: Acoustic
        "aggressive_mood": highlevel_data['highlevel'].get('mood_aggressive').get('all').get('aggressive'),  # Mood: Aggressive
        "electronic_mood": highlevel_data['highlevel'].get('mood_electronic').get('all').get('electronic'),  # Mood: Electronic
        "happy_mood": highlevel_data['highlevel'].get('mood_happy').get('all').get('happy'),  # Mood: Happy
        "party_mood": highlevel_data['highlevel'].get('mood_party').get('all').get('party'),  # Mood: Party
        "relaxed_mood": highlevel_data['highlevel'].get('mood_relaxed').get('all').get('relaxed'),  # Mood: Relaxed
        "sad_mood": highlevel_data['highlevel'].get('mood_sad').get('all').get('sad'),  # Mood: Sad

        # Mood Mirex features
        "cluster1_mirex": highlevel_data['highlevel'].get('moods_mirex').get('all').get('Cluster1'),  # Mood Mirex: Cluster 1
        "cluster2_mirex": highlevel_data['highlevel'].get('moods_mirex').get('all').get('Cluster2'),  # Mood Mirex: Cluster 2
        "cluster3_mirex": highlevel_data['highlevel'].get('moods_mirex').get('all').get('Cluster3'),  # Mood Mirex: Cluster 3
        "cluster4_mirex": highlevel_data['highlevel'].get('moods_mirex').get('all').get('Cluster4'),  # Mood Mirex: Cluster 4
        "cluster5_mirex": highlevel_data['highlevel'].get('moods_mirex').get('all').get('Cluster5')  # Mood Mirex: Cluster 5
    }
        flat_entry.update(high_level_features)
    
    features_list.append(flat_entry)

    df = pd.DataFrame(features_list)

    training_features = joblib.load('features.pkl')

    df = df.drop(columns=['mbid', 'genre'])   
    df = flatten_array_columns(df)
    
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)
    
    # Convert to numpy array
    df_array = df.to_numpy()

    # Apply scaling
    scaler = joblib.load('scaler.pkl')
    df_scaled = scaler.transform(df_array)

    return pd.DataFrame(df_scaled, columns=training_features)

def flatten_array_columns(df, length=10):
    for column in df.columns:
        if isinstance(df[column].iloc[0], (list, np.ndarray)):
            # Flatten each array to have 'length' number of elements (or pad with NaN)
            df[column] = df[column].apply(lambda x: x[:length] if isinstance(x, (list, np.ndarray)) else x)
            
            # If the array is shorter than the specified length, pad with NaN (or zeros if preferred)
            df[column] = df[column].apply(lambda x: x + [np.nan] * (length - len(x)) if len(x) < length else x)

            # Ensure the column is now a list of numbers with fixed length
            df[column] = df[column].apply(lambda x: np.array(x))
    
    return df

# Set up user agent for MusicBrainz API
mb.set_useragent("Project-Playlist", "1.0", "connorxspears@gmail.com")

songName = input("Enter the song name:")
songArtist = input(f"Enter the artist for {songName}:")

songData = getMBID(songName, songArtist)
print("songData before flatten_array_columns:", songData)

model = GenreNN(454, 512, 50)
model.load_state_dict(torch.load('GenreClassifier.pth'), strict=False)
model.eval()

json_file_path = os.path.abspath('../genre_data.json')
with open(json_file_path, 'r') as genre_json:
    genres = json.load(genre_json)
genre_names = [genre["name"] for genre in genres["genres"]]

with torch.no_grad():
    songTensor = torch.tensor(songData.values, dtype=torch.float32)
    print(f"songTensor: {songTensor}")

    output = model(songTensor)
    print(f"Logits before clamping: {output}")

    output = torch.clamp(output, min=-20, max=20)
    print(f"Logits after clamping: {output}")

    probabilities = torch.nn.functional.softmax(output, dim=1, dtype=torch.float32)
    print(f"Probabilities after softmax: {probabilities}")

    top3_indices = torch.topk(probabilities, 3).indices.squeeze().tolist()
    top3_genres = [(genre_names[i], probabilities[0, i].item()) for i in top3_indices]
    print(f"Top 3 predicted genres: {top3_genres}")