import streamlit as st
from keras.models import load_model
from helpers import predictNewMidi
import h5py
import os

# Define the main function to run the app
def main():
    # Set up the app title and description
    st.title("MIDI Generator")
    st.write("Upload a MIDI file and a pre-trained model to generate new music.")

    col1, col2 = st.columns(2)
    with col1:
        # Set up a file uploader to get the user's MIDI file
        uploadedMidi = st.file_uploader("Choose a MIDI file", type=["mid"])

    with col2:
        # Set up a file uploader to get the user's pre-trained model

        modelFile = st.file_uploader("Choose a pre-trained model file", type=["h5"])

    "---"
    if modelFile is not None:
        with h5py.File(modelFile, "r") as file:
        # Load the Keras model from the HDF5 file
            model = load_model(file)

    # Check if both a MIDI file and a model file have been uploaded
    if uploadedMidi is not None:
        # Load the pre-trained model from the file
        if modelFile is None:
            "*No model file was uploaded, using the default model.*"
            model = load_model("./models/model.h5")
            
        pred = predictNewMidi(uploadedMidi, model)

        # Display the output MIDI file to the user
        "## Prediction"
        f"*{uploadedMidi.name}*: **{round(pred[0][0] * 100, 2)}%**"

# Run the main function
if __name__ == "__main__":
    main()
