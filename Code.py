import numpy as np
import pandas as pd

songs_data=pd.read_csv("C:\\Users\\jangr\\OneDrive\\Desktop\\Machine learning End-to-end\\spotify_songs.csv")

print(songs_data)

print(songs_data.isnull().sum())

songs_data.dropna(inplace=True)

print(songs_data.info())

songs_data = songs_data[songs_data['track_popularity']>50]

print(songs_data)

songs_data = songs_data.groupby('track_name', group_keys=False).apply(lambda x: x.loc[x['track_popularity'].idxmax()])

print(songs_data.columns)

songs_data = songs_data[['track_artist','track_popularity','track_album_name','playlist_name','playlist_genre','playlist_subgenre']]

print(songs_data.reset_index(inplace=True))

print(songs_data)

print(songs_data['track_name'].unique().shape)

songs_data['track_artist_1'] = songs_data['track_artist'].apply(lambda x: x.replace(" ",""))
songs_data['track_artist_1'] = songs_data['track_artist_1'].apply(lambda x: x.split())

songs_data['track_album_name'] = songs_data['track_album_name'].apply(lambda x: x.replace(" ",""))
songs_data['track_album_name'] = songs_data['track_album_name'].apply(lambda x: x.split())

songs_data['playlist_subgenre'] = songs_data['playlist_subgenre'].apply(lambda x: x.replace(" ",""))
songs_data['playlist_subgenre'] = songs_data['playlist_subgenre'].apply(lambda x: x.split())

songs_data['playlist_name'] = songs_data['playlist_name'].apply(lambda x: x.replace(" ",""))
songs_data['playlist_name'] = songs_data['playlist_name'].apply(lambda x: x.split())

songs_data['playlist_genre'] = songs_data['playlist_genre'].apply(lambda x: x.replace(" ",""))
songs_data['playlist_genre'] = songs_data['playlist_genre'].apply(lambda x: x.split())

print(songs_data)

songs_data['track_popularity'] = songs_data['track_popularity'].apply(lambda x: [str(x)])

print(songs_data)

#Creating tags

songs_data['tags'] = songs_data['track_artist_1']+songs_data['track_popularity']+songs_data['track_album_name']+songs_data['playlist_name']+songs_data['playlist_genre']+songs_data['playlist_subgenre']

print(songs_data)

df = songs_data[['track_name','tags','track_artist']]

df['tags'] = df['tags'].apply(lambda x: " ".join(x))

df['tags'] = df['tags'].apply(lambda x: x.lower())

# from nltk.stem.porter import PorterStemmer
# ps = PorterStemmer()

# def stem(obj):
#     y=[]
#     for i in obj.split():
#         y.append(ps.stem(i))
#     return " ".join(y)
# df['tags'] = df['tags'].apply(stem)

print(df['tags'][0])

print(df)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=4000, stop_words='english')

vectors = cv.fit_transform(df['tags']).toarray()

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)

print(similarity[0])

sorted(list(enumerate(similarity[0])),reverse=True,key=(lambda x: x[1]))

def recommend(song):
    song_index = df[df['track_name']==song].index[0]
    distances = similarity[song_index]
    songs_list = sorted(list(enumerate(distances)),reverse=True,key=(lambda x: x[1]))[1:11]
    
    for i in songs_list:
        print(df.iloc[i[0]].track_name)

recommend('Without Me')

print(df.iloc[1319].track_artist)