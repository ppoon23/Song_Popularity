import streamlit as st
import pickle
import numpy as np
import pandas as pd
import gzip
import plotly.graph_objects as go
import joblib

### Code adopted from https://github.com/dataprofessor/code/blob/master/streamlit/part2/iris-ml-app.py

st.write("""
# Song Popularity Prediction App
## This app can predict a song's popularity from 2010 - 2020!

The original dataset and user input features information can be found below.
""")

link = '[Kaggle - Spotify Dataset](https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks)'
spotify_api = '[Spotify Audio Features](https://developer.spotify.com/documentation/web-api/reference/#endpoint-get-audio-features)'
st.markdown(link, unsafe_allow_html=True)
st.markdown(spotify_api, unsafe_allow_html=True)

st.sidebar.header('User Input Parameters')


# Get user inputs
def user_input_features():
    acousticness = st.sidebar.slider('acousticness', 0.0, 1.0, 0.23)
    danceability = st.sidebar.slider('danceability', 0.0, 1.0, 0.7)
    duration_ms = st.sidebar.slider('duration_ms', 5000, 533800, 141000)
    energy = st.sidebar.slider('energy', 0.0, 1.0, 0.5)
    explicit = st.sidebar.slider('explicit', 0, 1, 1)
    instrumentalness = st.sidebar.slider('instrumentalness', 0.0, 1.0, 0.0)
    key = st.sidebar.slider('key', 0, 11, 3)
    liveness = st.sidebar.slider('liveness', 0.0, 1.0, 0.5)
    loudness = st.sidebar.slider('loudness', -54.0, 3.2, -4.0)
    mode = st.sidebar.slider('mode', 0, 1, 0)
    speechiness = st.sidebar.slider('speechiness', 0.0, 1.0, 0.03)
    tempo = st.sidebar.slider('tempo', 0.0, 220.0, 92.0)
    valence = st.sidebar.slider('valence', 0.0, 1.0, 0.7)
    year = st.sidebar.slider('year', 2010, 2020, 2015)
    data = {'acousticness': acousticness,
            'danceability': danceability,
            'duration_ms': duration_ms,
            'energy': energy,
            'explicit': explicit,
            'instrumentalness': instrumentalness,
            'key': key,
            'liveness': liveness,
            'loudness': loudness,
            'mode': mode,
            'speechiness': speechiness,
            'tempo': tempo,
            'valence': valence,
            'year': year,
            }
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

# Show user inputs
st.subheader('User Input parameters')
st.write(df)

# Create Plotly plot
columns = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness']
df_song_char = df.filter(items=columns)
y = df_song_char.values.tolist()[0]

fig = go.Figure(data=go.Bar(x=columns, y=y), layout_title_text='Audio Features from User Input')
st.plotly_chart(fig, use_container_width=True)

# load from pickle file
# with gzip.open('model_compressed.pkl.gz', 'rb') as f:
#     model_final_pipe = pickle.load(f)

model_final_pipe = joblib.load('model_compressed.pkl')

prediction = model_final_pipe.predict(df)

st.subheader('Predicted Song Popularity')
prediction = int(np.round(prediction, 0))
st.write(prediction)
