import musicbrainzngs as mb
import time
import os
import json
import requests

def fetch_acousticbrainz_data(mbid, high_quality=True):
    """Fetches AcousticBrainz data for a given MBID, either high-level or low-level."""
    url = f"https://acousticbrainz.org/{mbid}/high-level" if high_quality else f"https://acousticbrainz.org/{mbid}/low-level"
    
    try:
        response = requests.get(url)
        if response.status_code == 404:
            return None  # No data available
        response.raise_for_status()
        return response.json()  # Return the JSON data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {mbid}: {e}")
        return None  # Return None on error

def song_collection(genres, max_songs_per_genre=100, pause_duration=1):
    """Collects song MBIDs by genre and fetches their AcousticBrainz data."""
    genre_song_ids = {genre: [] for genre in genres}
    high_level_data = {}
    low_level_data = {}

    for genre in genres:
        high_level_data[genre] = []
        low_level_data[genre] = []
        offset = 0
        collected_songs = 0

        while collected_songs < max_songs_per_genre:
            try:
                # Query MusicBrainz for songs by genre
                result = mb.search_recordings(
                    query=f'tag:"{genre}"',
                    limit=max_songs_per_genre,
                    offset=offset
                )

                if not result['recording-list']:
                    print(f"No more songs found for genre '{genre}' at offset {offset}.")
                    break

                for recording in result['recording-list']:
                    mbid = recording['id']
                    
                    # Fetch high-level data
                    acoustic_data_high = fetch_acousticbrainz_data(mbid, high_quality=True)
                    if acoustic_data_high:
                        # Fetch low-level data only if high-level data is available
                        acoustic_data_low = fetch_acousticbrainz_data(mbid, high_quality=False)
                        if acoustic_data_low:
                            genre_song_ids[genre].append(mbid)
                            collected_songs += 1
                            
                            # Save data by genre, linked to MBID
                            high_level_data[genre].append({"mbid": mbid, "data": acoustic_data_high})
                            low_level_data[genre].append({"mbid": mbid, "data": acoustic_data_low})
                            print(f"Song {mbid} added successfully for genre '{genre}'!")
                            print(f"Songs collected in genre '{genre}': {collected_songs}")

                    if collected_songs == max_songs_per_genre:
                        break

                offset += max_songs_per_genre
                time.sleep(pause_duration)
            except Exception as e:
                print(f"Error during data collection for genre '{genre}': {e}")
                break

        print(f"Finished genre '{genre}' with {collected_songs} songs collected.")

    # Save high-level and low-level data to separate JSON files
    with open("high_level_data.json", "w") as f_high:
        json.dump(high_level_data, f_high, indent=4)
    with open("low_level_data.json", "w") as f_low:
        json.dump(low_level_data, f_low, indent=4)
    
    # Save genre and song ID mapping
    with open("genre_song_ids.json", "w") as f_ids:
        json.dump(genre_song_ids, f_ids, indent=4)

    print("Data collection and saving complete.")
    return genre_song_ids

# Set up your user agent for MusicBrainz API
mb.set_useragent("Project-Playlist", "1.0", "connorxspears@gmail.com")

# Load the generalized genres list from a JSON file
json_file_path = os.path.abspath('../genre_data.json')
with open(json_file_path, 'r') as genre_json:
    genres = json.load(genre_json)
genre_names = [genre["name"] for genre in genres["genres"]]

# Start data collection process
genre_song_ids = song_collection(genre_names)
print(f"Collected song IDs by genre: {genre_song_ids}")
