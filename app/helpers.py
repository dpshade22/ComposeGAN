import mido
import pandas as pd
import numpy as np
import io

MAX_TIME_STEPS = 1000

def getMidiDF(uploadedMidi):
    midiData = io.BytesIO(uploadedMidi.read())
    
    # Store the data from the MIDI file in a list of dictionaries
    mid = mido.MidiFile(file=midiData)
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


def predictNewMidi(uploadedMidi, model):
    # Load the MIDI file into a Pandas DataFrame
    midi_df = getMidiDF(uploadedMidi)

    # Add a new dimension at the beginning to represent the batch size
    midi_array = np.expand_dims(midi_df.values, axis=0)

    # Pad the array with zeros to match the expected shape of (x, MAX_TIME_STEPS, 4)

    if midi_array.shape[1] < MAX_TIME_STEPS:
        padding = ((0, 0), (0, MAX_TIME_STEPS - midi_array.shape[1]), (0, 0))
        midi_array = np.pad(midi_array, padding, mode="constant")

    # Use the model to predict the output
    predictions = model.predict(midi_array)

    return predictions