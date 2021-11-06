import wave
import IPython.display as ipd

import streamlit as st
import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt
filename = 'notes1/fold1/S\'\'.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'music-carnatic/notes1/fold1/S\'.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'music-carnatic/notes1/fold2/R.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'music-carnatic/notes1/fold2/R\'.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'music-carnatic/notes1/fold2/R1.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'music-carnatic/notes1/fold2/R1\'.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'music-carnatic/notes1/fold3/G.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'music-carnatic/notes1/fold3/G\'.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'music-carnatic/notes1/fold3/G1.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'music-carnatic/notes1/fold3/G1\'.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'music-carnatic/notes1/fold4/M1,.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'music-carnatic/notes1/fold4/M1.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'music-carnatic/notes1/fold4/M1\'.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'music-carnatic/notes1/fold4/M2,.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'music-carnatic/notes1/fold4/M2.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'music-carnatic/notes1/fold4/M2\'.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'music-carnatic/notes1/fold5/P,.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'music-carnatic/notes1/fold5/P.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'music-carnatic/notes1/fold5/P\'.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'music-carnatic/notes1/fold6/D,.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'music-carnatic/notes1/fold6/D.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'music-carnatic/notes1/fold6/D\'.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'music-carnatic/notes1/fold6/D1,.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'music-carnatic/notes1/fold6/D1.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'music-carnatic/notes1/fold6/D1\'.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'music-carnatic/notes1/fold7/N,.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'music-carnatic/notes1/fold7/N.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'music-carnatic/notes1/fold7/N\'.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'music-carnatic/notes1/fold7/N1,.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'music-carnatic/notes1/fold7/N1.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
filename = 'music-carnatic/notes1/fold7/N1\'.wav'
data,sample_rate = librosa.load(filename)
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(filename)
import pandas as pd
metadata = pd.read_csv('music-carnatic/notes1/metadata.csv')
import struct

class WavFileHelper():
    
    def read_file_properties(self, filename):

        wave_file = open(filename,"rb")
        
        riff = wave_file.read(12)
        fmt = wave_file.read(36)
        
        num_channels_string = fmt[10:12]
        num_channels = struct.unpack('<H', num_channels_string)[0]

        sample_rate_string = fmt[12:16]
        sample_rate = struct.unpack("<I",sample_rate_string)[0]
        
        bit_depth_string = fmt[22:24]
        bit_depth = struct.unpack("<H",bit_depth_string)[0]

        return (num_channels, sample_rate, bit_depth)
import librosa 
from scipy.io import wavfile as wav
import numpy as np

filename = 'music-carnatic/notes1/fold7/N1.wav' 

librosa_audio, librosa_sample_rate = librosa.load(filename) 
scipy_sample_rate, scipy_audio = wav.read(filename) 
import matplotlib.pyplot as plt
mfccs = librosa.feature.mfcc(y=librosa_audio, sr=librosa_sample_rate, n_mfcc=40)
import librosa.display

import numpy as np
max_pad_len = 174

def extract_feature(file_name):
   
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None 
     
    return mfccs
# Load various imports 
import pandas as pd
import os
import librosa

# Set the path to the full UrbanSound dataset 
fulldatasetpath = 'music-carnatic/notes1/'

metadata = pd.read_csv('music-carnatic/notes1/metadata.csv')

features = []

# Iterate through each sound file and extract the features 
for index, row in metadata.iterrows():
    
    file_name = os.path.join(os.path.abspath(fulldatasetpath),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    
    class_label = row["class_name"]
    data = extract_feature(file_name)
    
    features.append([data, class_label])

# Convert into a Panda dataframe 
featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Convert features and corresponding classification labels into numpy arrays
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))
# split the dataset 
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)



import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 

num_rows = 40
num_columns = 174
num_channels = 1


x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

num_labels = yy.shape[1]
filter_size = 2

# Construct model 

# Construct model 
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))

model.add(Conv2D(filters=252, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))

model.add(Conv2D(filters=253, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))

model.add(Conv2D(filters=256, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))
model.add(GlobalAveragePooling2D())

model.add(Dense(num_labels, activation='softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# Display model architecture summary 

# Calculate pre-training accuracy 
score = model.evaluate(x_test, y_test, verbose=0)

from tensorflow.keras.callbacks import ModelCheckpoint 
from datetime import datetime 

num_epochs = 100
num_batch_size = 256

checkpointer = ModelCheckpoint(filepath='music-carnatic/notes1/saved_models/weights.best.basic_cnn.hdf5', 
                               verbose=0, save_best_only=True)
start = datetime.now()

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=0)


duration = datetime.now() - start


# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train, verbose=0)
import librosa 
import numpy as np 


def print_prediction(file_name):
    prediction_feature = extract_feature(file_name) 
    prediction_feature = prediction_feature.reshape(1, num_rows, num_columns, num_channels)
    prediction = model.predict(prediction_feature)
    predicted_class = le.inverse_transform(np.argmax(prediction, axis=1))
    st.write("Key: ", predicted_class[0],end='') 
    f = open('music-carnatic/notes1/log.txt', 'a')
    print(predicted_class[0], file = f,end=' ')
    
  
    return 0
   

# Import necessary libraries 
from pydub import AudioSegment 
import speech_recognition as sr 
  
# Input audio file to be sliced 
from glob import glob
st.title("WELCOME TO THE WORLD OF MUSIC!")
from PIL import Image

vid=open("example.mp4","rb")
st.video(vid)

st.markdown("<span style=“background-color:#121922”>",unsafe_allow_html=True)
def mus(audio):
    CHANNELS = 1
    swidth = 2
    Change_RATE = 2

    spf = wave.open(audio, 'rb')
    RATE=spf.getframerate()
    signal = spf.readframes(-1)

    wf = wave.open('changed2.wav', 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(swidth)
    wf.setframerate(RATE*Change_RATE)
    wf.writeframes(signal)
    wf.close()
    audio = 'changed2.wav'
    return (audio)

audio = st.file_uploader("Choose an audio...", type="wav")
if audio is not None:
    audio=mus(audio)

    audio = AudioSegment.from_wav(audio)
    n = len(audio) 
    counter = 1
    interval = 1 * 1000
    overlap = 0
    start = 0
    end = 0
    flag = 0
    for i in range(0,  n, interval): 

        if i == 0: 
            start = 0
            end = interval 
        else: 
            start = end  -overlap
            end = start + interval  
        if end >= n: 
            end = n 
            flag = 1

        # Storing audio file from the defined start to end 
        chunk = audio[start:end] 

        # Filename / Path to store the sliced audio 
        filename = 'music-carnatic/notes1/test2/chunk'+str(counter)+'.wav'
        chunk.export(filename, format ="wav") 
        # Print information about the current chunk 

        # Increment counter for the next chunk 
        counter = counter + 1
    for name in glob('music-carnatic/notes1/test2/*.wav'):
        filename = name
        print_prediction(filename)
from IPython.display import Audio
from ipywebrtc import CameraStream, AudioRecorder
st.sidebar.title("Duration")
duration = st.sidebar.slider("Recording duration", 0.0, 3600.0, 3.0)
import pyaudio
import wave
def record_and_predict(duration):
    

    # actually I found this hack in some js code
    # just pass mime type =)


    # the file name output you want to record into
    filename = "recorded.wav"
    # set the chunk size of 1024 samples
    chunk = 1024
    # sample format
    FORMAT = pyaudio.paInt16
    # mono, change to 2 if you want stereo
    channels = 1
    # 44100 samples per second
    sample_rate = 44100
    record_seconds = duration
    # initialize PyAudio object
    p = pyaudio.PyAudio()
    # open stream object as input & output
    stream = p.open(format=FORMAT,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)
    frames = []
    print("Recording...")
    for i in range(int(44100 / chunk * record_seconds)):
        data = stream.read(chunk)
        # if you want to hear your voice while recording
        # stream.write(data)
        frames.append(data)
    print("Finished recording.")
    # stop and close stream
    stream.stop_stream()
    stream.close()
    # terminate pyaudio object
    p.terminate()
    # save audio file
    # open the file in 'write bytes' mode
    wf = wave.open(filename, "wb")
    # set the channels
    wf.setnchannels(channels)
    # set the sample format
    wf.setsampwidth(p.get_sample_size(FORMAT))
    # set the sample rate
    wf.setframerate(sample_rate)
    # write the frames as bytes
    wf.writeframes(b"".join(frames))
    # close the file
    wf.close()

# convert wav to mp3                                                            
    audio="recorded.wav"
    if audio is not None:
        audio=mus(audio)
        audio = AudioSegment.from_wav(audio)
        n = len(audio) 
        counter = 1
        interval = 1 * 1000
        overlap = 0
        start = 0
        end = 0
        flag = 0
        for i in range(0,  n, interval): 

            if i == 0: 
                start = 0
                end = interval 
            else: 
                start = end  -overlap
                end = start + interval  
            if end >= n: 
                end = n 
                flag = 1

            # Storing audio file from the defined start to end 
            chunk = audio[start:end] 

            # Filename / Path to store the sliced audio 
            filename = 'music-carnatic/notes1/test2/chunk'+str(counter)+'.wav'
            chunk.export(filename, format ="wav") 
            # Print information about the current chunk 

            # Increment counter for the next chunk 
            counter = counter + 1
        for name in glob('music-carnatic/notes1/test2/*.wav'):
            filename = name
            print_prediction(filename)

if st.button("Start Recording"):
    with st.spinner("Recording..."):
        record_and_predict(duration)

