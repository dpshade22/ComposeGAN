import mido
import random
import os
import pandas as pd
import numpy as np

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import COMMASPACE
from email import encoders


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras import optimizers, regularizers
from keras.layers import GRU, LSTM, Dropout, Dense, BatchNormalization
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.optimizers import Adam

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
import tensorflow as tf

tf.get_logger().setLevel("ERROR")


MAX_TIME_STEPS = 1000
RealMidiFolder = "../assets/realMusic"
FakeMidiFolder = "../assets/fakeMusic"
RealCSVFolder = "../assets/csvs/realCSVs"
FakeCSVFolder = "../assets/csvs/fakeCSVs"

"""## Generate Random Music"""


def generateRandomMidiFiles(numFiles):
    # Generate random MIDI events
    for x in range(numFiles):
        outfile = f"{FakeMidiFolder}/randomMidi{x}.mid"
        mid = mido.MidiFile()

        # Create a new MIDI track
        track = mido.MidiTrack()
        mid.tracks.append(mido.MidiTrack())
        mid.tracks.append(track)

        for i in range(MAX_TIME_STEPS):
            # Generate a message based on the type

            message = mido.Message(
                "note_on",
                note=random.randint(0, 127),
                velocity=random.randint(0, 127),
                time=random.randint(0, 1200),
            )

            # Add the message to the track
            track.append(message)

        # Save the MIDI file
        mid.save(outfile)


# For row in realCSVs, add a new random number to each value
def addNoiseToCSVs(realCSVsFolder=RealCSVFolder, fakeCSVs=FakeCSVFolder):
    for csv in os.listdir(realCSVsFolder):
        try:
            # Read the CSV file into a dataframe
            df = pd.read_csv(f"{RealCSVFolder}/{csv}")
        except:
            os.remove(f"{RealCSVFolder}/{csv}")
            continue

        # Add random noise to each column
        df["track"] = df["track"] + np.random.randint(1, 2, len(df))
        df["note"] = df["note"] + np.random.randint(0, 11, len(df))
        df["velocity"] = df["velocity"]
        df["time"] = df["time"] + np.random.randint(0, 360, len(df))

        # Write the modified dataframe to a new CSV file
        df.to_csv(f'{FakeCSVFolder}/{csv.split(".")[0]}_modified.csv', index=False)


"""## MIDI processing functions"""


def getMidiDF(fileLoc: str):
    print(fileLoc)
    # Store the data from the MIDI file in a list of dictionaries
    mid = mido.MidiFile(fileLoc, clip=True)
    midi_data = []
    for i, track in enumerate(mid.tracks):
        addTime = 0
        for msg in track:
            if msg.type == "note_on":
                addTime += msg.time
                midi_data.append(
                    {
                        "track": i,
                        "note": msg.note,
                        "velocity": msg.velocity,
                        "time": msg.time,
                    }
                )

    # Convert the list of dictionaries to a Pandas DataFrame
    df = pd.DataFrame(midi_data)

    # Check if the DataFrame has fewer than MAX_TIME_STEP rows
    if df.shape[0] < MAX_TIME_STEPS:
        # Calculate the number of rows to pad with zeros
        num_rows_to_pad = MAX_TIME_STEPS - df.shape[0]

        # Create a new DataFrame with the necessary number of rows and columns
        padded_df = pd.DataFrame(
            np.zeros((num_rows_to_pad, df.shape[1])), columns=df.columns
        )

        # Concatenate the original DataFrame and the padded DataFrame
        df = pd.concat([df, padded_df])

    # Return the DataFrame with exactly MAX_TIME_STEPS rows
    return df[:MAX_TIME_STEPS]


def saveMidiFromDF(df: pd.DataFrame, midiFileName: str):
    mid = mido.MidiFile()

    tracks = [mido.MidiTrack() for i in df["track"].unique()]
    for track in tracks:
        mid.tracks.append(track)

    # Iterate through the DataFrame and add the messages to the tracks
    for i, row in df.iterrows():
        msg = mido.Message(
            type="note_on", note=row["note"], velocity=row["velocity"], time=row["time"]
        )
        tracks[0].append(msg)

    # Save the MIDI file
    mid.save(midiFileName)


# Load Midi file from CSV
def saveMidiFromCSV(csvFile: str, midiFileName: str):
    df = pd.read_csv(csvFile)
    try:
        df = df.drop(columns=["Unnamed: 0"])  # remove index column if present
    except:
        pass
    saveMidiFromDF(df, midiFileName)


"""## MIDI file import/export functions"""


def createCSVs(listOfDFs, folder):
    for i, df in enumerate(listOfDFs):
        df[:MAX_TIME_STEPS].to_csv(f"{folder}/{i}.csv", index=False)


def importMidisAndCreateCSVs(inputMidiFolder: str, outputCSVfolder: str):
    trainingDFs = []
    # Iterate over all subdirectories within the root folder
    for subdir, dirs, files in os.walk(inputMidiFolder):
        # Iterate over all files within the subdirectory
        for f in files:
            filePath = os.path.join(subdir, f)
            try:
                trainingDFs.append(getMidiDF(filePath))
            except:
                os.remove(filePath)
                continue

    createCSVs(trainingDFs, outputCSVfolder)


def loadCSVsToNumpy3DArray(folder):
    df_list = []
    for subdir, dirs, files in os.walk(folder):
        # Iterate over all files within the subdirectory
        for f in files:
            filePath = os.path.join(subdir, f)
            try:
                df_list.append(pd.read_csv(filePath))
            except:
                continue

    # Pad the arrays with zeros to make them all have the same number of time steps
    paddedDFs = [
        np.pad(df.values, ((0, MAX_TIME_STEPS - df.shape[0]), (0, 0)), "constant")
        for df in df_list
    ]

    # Stack the arrays along the first axis to create a 3D array
    kerasMidiData = np.stack(paddedDFs)
    return kerasMidiData


def csvToMidi(csvFilePath, midiFilePath):
    # Read the CSV file into a pandas dataframe
    df = pd.read_csv(csvFilePath)

    # Create a new MIDI file with one track
    midiFile = mido.MidiFile()
    track = mido.MidiTrack()
    midiFile.tracks.append(track)

    # Set the initial time and velocity values
    time = 0
    velocity = 64

    # Iterate through each row of the CSV file
    for index, row in df.iterrows():
        # Get the values from the row
        trackNumber = row["track"]
        note = row["note"]
        velocity = row["velocity"]
        deltaTime = row["time"]

        # Create a new message with the values from the row
        msg = mido.Message("note_on", note=note, velocity=velocity, time=deltaTime)

        # Add the message to the track
        track.append(msg)

    # Save the MIDI file
    midiFile.save(midiFilePath)


"""## Data preparation functions"""


def getNumpy3DArray(listOfTrainingDFs):
    # Find the maximum number of time steps

    # Pad the arrays with zeros to make them all have the same number of time steps
    paddedDFs = [
        np.pad(df, [(0, MAX_TIME_STEPS - df.shape[0]), (0, 0)], "constant")
        for df in listOfTrainingDFs
    ]

    # Stack the arrays along the first axis to create a 3D array
    kerasMidiData = np.stack(paddedDFs)
    return kerasMidiData


"""## Load and prepare the data"""


def createTrainingTestData(realData, fakeData, testSize=0.2, randomState=42):
    # Create labels for the real and fake data
    realLabels = np.ones((realData.shape[0], 1))
    fakeLabels = np.zeros((fakeData.shape[0], 1))

    # Combine the real and fake data and labels into a single array
    combinedData = np.concatenate((realData, fakeData), axis=0)
    combinedLabels = np.concatenate((realLabels, fakeLabels), axis=0)

    # Encode the labels using LabelEncoder from scikit-learn
    le = LabelEncoder()
    encodedLabels = le.fit_transform(combinedLabels.ravel())

    # Split the combined data and labels into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        combinedData, encodedLabels, test_size=testSize, random_state=randomState
    )

    return X_train, X_test, y_train, y_test


def getTrainAndTestFromCSVs(realCSVfolder=RealCSVFolder, fakeCSVfolder=FakeCSVFolder):
    realMusicFromCSVs = loadCSVsToNumpy3DArray(realCSVfolder)
    fakeMusicFromCSVs = loadCSVsToNumpy3DArray(fakeCSVfolder)
    X_train, X_test, y_train, y_test = createTrainingTestData(
        realMusicFromCSVs, fakeMusicFromCSVs
    )
    return X_train, X_test, y_train, y_test


def getTrainAndTestFromMidiFolders(
    realMidiFolder=RealMidiFolder,
    realCSVFolder=RealCSVFolder,
    fakeMidiFolder=FakeMidiFolder,
    fakeCSVFolder=FakeCSVFolder,
):
    importMidisAndCreateCSVs(realMidiFolder, realCSVFolder)
    importMidisAndCreateCSVs(fakeMidiFolder, fakeCSVFolder)
    X_train, X_test, y_train, y_test = getTrainAndTestFromCSVs()
    return X_train, X_test, y_train, y_test


"""## Model Training and Saving"""


def fitModel(X_train, X_test, y_train, y_test):
    # Define the model architecture
    model = Sequential()
    model.add(LSTM(units=128, input_shape=X_train.shape[1:], return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64))
    model.add(Dropout(0.2))
    model.add(Dense(units=64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation="sigmoid"))

    # Define the optimizer
    opt = optimizers.SGD(learning_rate=0.01)

    # Define callbacks
    early_stop = EarlyStopping(monitor="val_loss", patience=5)

    def scheduler(epoch, lr):
        if epoch < 10:
            return lr

        else:
            return lr * 0.1

    lr_scheduler = LearningRateScheduler(scheduler)

    # Compile and fit the model
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop, lr_scheduler],
    )

    # Evaluate the model on the testing data
    loss, accuracy = model.evaluate(X_test, y_test)
    print("Test loss:", loss)
    print("Test accuracy:", accuracy)

    # Save the trained model to a file
    model.save("model.h5")


def sendEmail(accuracy, loss):
    # Define the email content
    msg = MIMEMultipart()
    msg["Subject"] = f"Trained model - Accuracy: {accuracy:.2f} - Loss: {loss:.2f}"
    msg["From"] = "dpshadey22@gmail.com"
    msg["To"] = "dpshadey22@gmail.com"
    msg.attach(MIMEText("Please find attached the trained model."))

    # Add the model file as an attachment
    with open("model.h5", "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment", filename="model.h5")
        msg.attach(part)

    # Send the email
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login("dpshadey22@gmail.com", "eghrbpkjmywgkrsv")

    server.send_message(msg)
    server.quit()

    print("Email sent successfully.")


"""## Model Prediction"""


def predictMidiFile(midiFilePath, model):
    # Load the MIDI file into a Pandas DataFrame
    midi_df = getMidiDF(midiFilePath)

    # Add a new dimension at the beginning to represent the batch size
    midi_array = np.expand_dims(midi_df.values, axis=0)

    # Pad the array with zeros to match the expected shape of (x, MAX_TIME_STEPS, 4)

    if midi_array.shape[1] < MAX_TIME_STEPS:
        padding = ((0, 0), (0, MAX_TIME_STEPS - midi_array.shape[1]), (0, 0))
        midi_array = np.pad(midi_array, padding, mode="constant")

    # Use the model to predict the output
    predictions = model.predict(midi_array)

    return predictions


def predictCSV(csvDF, model):
    # Add a new dimension at the beginning to represent the batch size
    csv_array = np.expand_dims(csvDF.values, axis=0)

    # Pad the array with zeros to match the expected shape of (x, MAX_TIME_STEPS, 4)
    if csv_array.shape[1] < MAX_TIME_STEPS:
        padding = ((0, 0), (0, MAX_TIME_STEPS - csv_array.shape[1]), (0, 0))
        csv_array = np.pad(csv_array, padding, mode="constant")

    # Use the model to predict the output
    predictions = model.predict(csv_array)

    return predictions
