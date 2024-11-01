import musicbrainzngs as mb
import numpy as np
import time
import os
import json

def song_collection(genres, max_songs_per_genre=100, pause_duration=1):
    genre_song_ids = {genre: [] for genre in genres}

    for genre in genres:
        offset = 0
        collected_songs = 0

        while collected_songs < max_songs_per_genre:
            try:
                result = mb.search_recordings(
                    query=f'genre:"{genre}"',
                    limit=max_songs_per_genre,
                    offset=offset                          
                )

                if not result['recording-list']:
                    print(f"No more songs found for genre '{genre}' at offset {offset}.")
                    break

                for recording in result['recording-list']:
                    genre_song_ids[genre].append(recording['id'])
                    collected_songs += 1

                    if collected_songs == max_songs_per_genre: 
                        break
                
                offset += max_songs_per_genre
                time.sleep(pause_duration)
            except Exception as e:
                print(f"Error during data collection for genre '{genre}': {e}")
                break
        print(f"Finished Genre: '{genre}' with {len(genre_song_ids[genre])} songs collected.")
    return genre_song_ids

# Set up your user agent for MusicBrainz API
mb.set_useragent("Project-Playlist", "1.0", "connorxspears@gmail.com")

# Load the generalized genres list from a JSON file
json_file_path = os.path.abspath('../genre_data.json')
with open(json_file_path, 'r') as genre_json:
    genres = json.load(genre_json)
genre_names = [genre["name"] for genre in genres["genres"]]

genre_song_ids = song_collection(genre_names)
with open("genre_songs_ids.json", "w") as f:
    json.dump(genre_song_ids, f)
print(f"Collected song IDs by genre: {genre_song_ids}")