# Project-Playlist
An app which constructs a basic multi-layer perceptron to predict the genre of an inputted song using data from AcousticBrainz.

## How To Run
1. Download the provided .csv file and put it into the Data folder.
2. Make sure to set up your environment with the libraries found in Requirements.txt.
3. Run startingNN.py to train the data, this will output multiple training statistics and charts using matplotlib.
4. Run app.py, you must be connected to the internet as this app uses a web-based API call to the Music/AcousticBrainz database.
5. The network will rerun the training process (this means you may not need to run startingNN.py before running app.py, but run it just to be safe).
6. Input a song name and the artist associated
7. Wait for your output!

## Requirements
combinedData.csv - https://drive.google.com/file/d/1bpqcktwniSO9RjLkUgC3AAGFHFaDbgvT/view?usp=sharing
