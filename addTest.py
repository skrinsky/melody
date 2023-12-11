import os
import pretty_midi
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed  # Add TimeDistributed

def is_midi_file(file_path):
    try:
        pretty_midi.PrettyMIDI(file_path)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def midi_to_sequences(midi_file):
    sequences = []

    if is_midi_file(midi_file):
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                sequences.append([note.pitch, note.start, note.end])

    return sequences

# Example usage:
midi_folder_path = '/Users/summerkrinsky/Documents/python/midi'
all_sequences = []  # Create a list to store sequences for all MIDI files

for midi_file in os.listdir(midi_folder_path):
    if not midi_file.endswith('.DS_Store'):  # Skip non-MIDI files
        midi_file_path = os.path.join(midi_folder_path, midi_file)
        sequences = midi_to_sequences(midi_file_path)
    
        # Append sequences for the current MIDI file to the list
        all_sequences.append(sequences)

# Assuming your sequences have a maximum length of 36
maxlen = 36

# Convert sequences to a list of NumPy arrays
sequences_list = [np.array(sequences) for sequences in all_sequences]

# Pad sequences to the specified maxlen
padded_sequences = pad_sequences(sequences_list, maxlen=maxlen, padding='post', truncating='post', dtype='float32')

# Convert the list of sequences to a NumPy array
X = np.array(sequences_list)


# Split the data into training and testing sets
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Reshape the data to be 3D (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# Define the LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(64, return_sequences=False))
    model.add(RepeatVector(input_shape[0]))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(TimeDistributed(Dense(3, activation='linear')))
    model.compile(optimizer='adam', loss='mse')
    return model

# Build the LSTM model
input_shape = (maxlen, 3)

# Print the shapes of your input data
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)

# Print the expected input shape for the model
print("Expected input shape for the model:", input_shape)

model = build_lstm_model(input_shape)

# Train the model on the training set
epochs = 10
batch_size = 4
model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size)

# Evaluate the model on the test set
test_loss = model.evaluate(X_test, X_test, batch_size=batch_size)
print(f'Test loss: {test_loss}')


def generate_midi_melody(model, input_sequence, sequence_length=36):
    # Initialize an empty sequence
    generated_sequence = []

    # Generate the first note to seed the sequence
    initial_input = np.zeros((1, sequence_length, input_sequence.shape[-1]))
    initial_input[:, :input_sequence.shape[1], :] = input_sequence

    # Append the provided input_sequence to the generated_sequence
    generated_sequence.extend(input_sequence[0])

    # Generate the rest of the sequence
    for _ in range(sequence_length - input_sequence.shape[1]):
        # Use the last few notes from the generated sequence as input
        actual_input_data = np.array([generated_sequence[-input_sequence.shape[1]:]])
        model_input = np.array([actual_input_data])
        
        # Use the model to predict the next note with temperature
        next_prediction = model.predict(model_input)[0]
        generated_sequence.append(next_prediction)



    # Create a PrettyMIDI object to convert the sequence to MIDI
    midi_data = pretty_midi.PrettyMIDI()

    # Create an instrument (e.g., piano)
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'))
   

    # Convert the sequence to MIDI notes
    for note_params in generated_sequence:
        note_number, note_on, note_off = note_params
        note = pretty_midi.Note(
            velocity=64,  # Adjust the velocity as needed
            pitch=int(note_number*88),
            start=(note_on * 15),
            end=(note_off *15)
        )
        instrument.notes.append(note)

    # Add the instrument to the MIDI data
    midi_data.instruments.append(instrument)

    # Save the generated MIDI file
    midi_data.write('generated_melody.mid')


# Assuming a sequence of length 36 and 3 features for each time step
input_sequence = np.random.rand(36, 3)
input_sequence = np.expand_dims(input_sequence, axis=0)


# Call the function to predict meldoy with trained model
generate_midi_melody(model, input_sequence)

print(model.summary())