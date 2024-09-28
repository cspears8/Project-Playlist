import musicbrainzngs
from musicbrainzngs import WebServiceError

# Set up your user agent
musicbrainzngs.set_useragent("Project-Playlist", "1.0", "connorxspears@gmail.com")

try:
    # Example: Searching for an artist
    result = musicbrainzngs.search_artists(artist="The Beatles")
    for artist in result['artist-list']:
        print(f"Name: {artist['name']}, ID: {artist['id']}")
except WebServiceError as e:
    print(f"An error occurred while accessing the MusicBrainz web service: {e}")