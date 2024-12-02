import musicbrainzngs
from sklearn.preprocessing import MinMaxScaler
import json
import pandas as pd

# Set up your user agent for MusicBrainz API
musicbrainzngs.set_useragent("Project-Playlist", "1.0", "connorxspears@gmail.com")

def parseData(): 
    # Load data
    with open('high_level_data.json', 'r') as high_level:
        highData = json.load(high_level)

    highlevel_features_list = []

    # Now process the songs
    for genre, songs in highData.items():
        for song in songs:
            song_id = song.get("mbid", None)
            highlevel_data = next((item["data"] for item in highData.get(genre, []) if item["mbid"] == song["mbid"]), {})

            flat_entry = {"mbid": song_id, "genre": genre}

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
                highlevel_features_list.append(flat_entry)
    
    highlevel_df = pd.DataFrame(highlevel_features_list)
    return highlevel_df

def flattenLowData():
    with open('low_level_data.json', 'r') as low_level:
        lowData = json.load(low_level)

    flatLowData = []

    for genre, songs in lowData.items():
        for song in songs:
            song_id = song.get("mbid", None)
            lowlevel_data = song.get("data", {}).get("lowlevel", {})

            flat_entry = {"mbid": song_id, "genre": genre}
            for key, value in lowlevel_data.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        flat_entry[f"{key}_{subkey}"] = subvalue
                else:
                    flat_entry[key] = value
            
            flatLowData.append(flat_entry)
    df = pd.DataFrame(flatLowData)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

lowlevel_df = flattenLowData()
highlevel_df = parseData()

print(lowlevel_df.columns)
print(highlevel_df.columns)

combined_df = pd.merge(lowlevel_df, highlevel_df, on=["mbid", "genre"], how="inner")

output_path = "combined_data.csv"
combined_df.to_csv(output_path, index=False)
print(f"Data combined and saved as {output_path}")
print(combined_df.head())