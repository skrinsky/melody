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
                sequences.append([note.pitch, note.start, note.end, note.velocity])

    return sequences

# Implement
midi_folder_path = '/Users/summerkrinsky/Documents/python/midi'
all_sequences = []  # Create a list to store sequences for all MIDI files

for midi_file in os.listdir(midi_folder_path):
    if not midi_file.endswith('.DS_Store'):  # Skip non-MIDI files
        midi_file_path = os.path.join(midi_folder_path, midi_file)
        sequences = midi_to_sequences(midi_file_path)
    
        # Append sequences for the current MIDI file to the list
        all_sequences.append(sequences)

# Assuming your sequences have a maximum length of 36
maxlen = 95

# Convert sequences to a list of NumPy arrays
sequences_list = [np.array(sequences) for sequences in all_sequences]

#for sequence in sequences_list:
   #print(np.array(sequence).shape)


# Pad sequences to the specified maxlen
padded_sequences = pad_sequences(sequences_list, maxlen=maxlen, padding='post', truncating='post', dtype='float32')

# Convert the list of sequences to a NumPy array
X = np.array(padded_sequences)


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
    model.add(TimeDistributed(Dense(4, activation='linear')))
    model.compile(optimizer='adam', loss='mse')
    return model

# Build the LSTM model
input_shape = (maxlen, 4)

# Print the shapes of your input data
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)

# Print the expected input shape for the model
print("Expected input shape for the model:", input_shape)

model = build_lstm_model(input_shape)


# Train the model on the training set
epochs = 80
batch_size = 64
model.fit(X_train, X_train,  epochs=epochs, batch_size=batch_size)

# Evaluate the model on the test set
test_loss = model.evaluate(X_test, X_test, batch_size=batch_size)
print(f'Test loss: {test_loss}')

def sample(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions) / temperature
    exp_predictions = np.exp(predictions)
    predictions = exp_predictions / np.sum(exp_predictions)
    probabilities = np.random.multinomial(1, predictions, 1)
    return np.argmax(probabilities)

def generate_melody(model, seed, sequence_length, temperature=.5):
    generated_sequence = seed.copy()

    # Predict the entire sequence at once
    input_sequence = seed.reshape(1, seed.shape[0], seed.shape[1])
    predicted_steps = model.predict(input_sequence)

    # Update the generated sequence step by step
    for i in range(sequence_length):
        predicted_step = predicted_steps[0, i]

        # Use the sample function with the specified temperature
        sampled_step = sample(predicted_step, temperature)

        # Update the generated sequence with the new prediction
        generated_sequence[i] = sampled_step

    return generated_sequence

sequence_length = 95
# Generate a new sequence
seed = np.random.rand(sequence_length, 4)
generated_sequence = generate_melody(model, seed, sequence_length)

# Convert the generated melody to PrettyMIDI format
midi_data = pretty_midi.PrettyMIDI()

# Add an instrument (assuming piano, change the program number as needed)
instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'))
midi_data.instruments.append(instrument)

# Assuming your generated melody has pitch, velocity, start_time, and end_time in each row
for note_info in generated_sequence:
    pitch = int(note_info[0])
    velocity = int(note_info[1])
    start_time = float(note_info[2])
    end_time = float(note_info[3])

    # Create a Note instance and add it to the PrettyMIDI instrument
    midi_note = pretty_midi.Note(
        velocity=velocity,
        pitch=pitch,
        start=start_time,
        end=end_time
    )
    instrument.notes.append(midi_note)

# Save the generated melody to a MIDI file
midi_data.write('generated_melody.mid')
