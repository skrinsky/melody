import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Embedding, Concatenate
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import math
seql = 50

def train_network():
    """ Train a Neural Network to generate music """
    notes, velocities, durations = get_notes()

    # Get unique pitch, velocity, and duration values
    pitch_names = sorted(set(notes))
    velocity_names = sorted(set(velocities))
    duration_values = sorted(set(durations))

    pitch_vocab_size = len(pitch_names)
    velocity_vocab_size = len(velocity_names)
    duration_vocab_size = len(duration_values)
    print("Pitch size: ", pitch_vocab_size, "Velocity size: ", velocity_vocab_size, "Duration size: ", duration_vocab_size)

    network_input_pitch, network_input_velocity, network_input_duration, network_output_pitch, network_output_velocity, network_output_duration = prepare_sequences(
        notes, velocities, durations, pitch_vocab_size, velocity_vocab_size, duration_vocab_size)

    model = create_network(pitch_vocab_size, velocity_vocab_size, duration_vocab_size)

    train(model, network_input_pitch, network_input_velocity, network_input_duration, network_output_pitch, network_output_velocity, network_output_duration)

def get_notes(velocity_quantization=10):
    """ Get all the notes, velocities, and durations from the MIDI files """
    notes = []
    velocities = []
    durations = []

    for file in glob.glob("midi_songsKEYCHANGE/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try:  # File has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:  # File has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                quantized_velocity = math.floor(element.volume.velocity / velocity_quantization) * velocity_quantization
                notes.append(str(element.pitch))
                velocities.append(quantized_velocity)
                durations.append(element.duration.quarterLength)
                print("Note,", str(element.pitch), quantized_velocity, element.duration.quarterLength)
            elif isinstance(element, chord.Chord):
                quantized_velocity = math.floor(element.notes[0].volume.velocity / velocity_quantization) * velocity_quantization
                chord_notes = [str(n.pitch) for n in element.notes]
                for i, n in enumerate(chord_notes):
                    notes.append(n)
                    velocities.append(quantized_velocity)
                    if i == 0:
                        durations.append(element.duration.quarterLength)  # Set the actual duration for the first note
                        print("Chord note,", n, quantized_velocity, element.duration.quarterLength)
                    else:
                        durations.append(0)  # Set duration to 0 for the other notes
                        print("Chord note,", n, quantized_velocity, 0)

    # Save notes, velocities, and durations to files
    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)
    with open('data/velocities', 'wb') as filepath:
        pickle.dump(velocities, filepath)
    with open('data/durations', 'wb') as filepath:
        pickle.dump(durations, filepath)

    return notes, velocities, durations


def prepare_sequences(notes, velocities, durations, pitch_vocab_size, velocity_vocab_size, duration_vocab_size):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = seql

    pitch_to_int = dict((note, number) for number, note in enumerate(sorted(set(notes))))
    velocity_to_int = dict((velocity, number) for number, velocity in enumerate(sorted(set(velocities))))
    duration_to_int = dict((duration, number) for number, duration in enumerate(sorted(set(durations))))

    network_input_pitch = []
    network_input_velocity = []
    network_input_duration = []
    network_output_pitch = []
    network_output_velocity = []
    network_output_duration = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in_pitch = notes[i:i + sequence_length]
        sequence_in_velocity = velocities[i:i + sequence_length]
        sequence_in_duration = durations[i:i + sequence_length]

        sequence_out_pitch = notes[i + sequence_length]
        sequence_out_velocity = velocities[i + sequence_length]
        sequence_out_duration = durations[i + sequence_length]

        network_input_pitch.append([pitch_to_int[char] for char in sequence_in_pitch])
        network_input_velocity.append([velocity_to_int[char] for char in sequence_in_velocity])
        network_input_duration.append([duration_to_int[char] for char in sequence_in_duration])

        network_output_pitch.append(pitch_to_int[sequence_out_pitch])
        network_output_velocity.append(velocity_to_int[sequence_out_velocity])
        network_output_duration.append(duration_to_int[sequence_out_duration])

    n_patterns = len(network_input_pitch)

    network_input_pitch = np.reshape(network_input_pitch, (n_patterns, sequence_length))
    network_input_velocity = np.reshape(network_input_velocity, (n_patterns, sequence_length))
    network_input_duration = np.reshape(network_input_duration, (n_patterns, sequence_length))

    network_output_pitch = to_categorical(network_output_pitch, num_classes=pitch_vocab_size)
    network_output_velocity = to_categorical(network_output_velocity, num_classes=velocity_vocab_size)
    network_output_duration = to_categorical(network_output_duration, num_classes=duration_vocab_size)

    return network_input_pitch, network_input_velocity, network_input_duration, network_output_pitch, network_output_velocity, network_output_duration

def create_network(pitch_vocab_size, velocity_vocab_size, duration_vocab_size):
    """ Create the structure of the neural network """
    input_pitch = Input(shape=(seql,))
    input_velocity = Input(shape=(seql,))
    input_duration = Input(shape=(seql,))

    embedding_pitch = Embedding(pitch_vocab_size, seql)(input_pitch)
    embedding_velocity = Embedding(velocity_vocab_size, seql)(input_velocity)
    embedding_duration = Embedding(duration_vocab_size, seql)(input_duration)

    merged = Concatenate()([embedding_pitch, embedding_velocity, embedding_duration])

    lstm = LSTM(512, return_sequences=True)(merged)
    lstm = Dropout(0.3)(lstm)
    lstm = LSTM(512, return_sequences=True)(lstm)
    lstm = Dropout(0.3)(lstm)
    lstm = LSTM(512)(lstm)

    dense_pitch = Dense(256)(lstm)
    dense_pitch = Dropout(0.3)(dense_pitch)
    dense_velocity = Dense(256)(lstm)
    dense_velocity = Dropout(0.3)(dense_velocity)
    dense_duration = Dense(256)(lstm)
    dense_duration = Dropout(0.3)(dense_duration)

    final_concat = Concatenate()([dense_pitch, dense_velocity, dense_duration])

    output_pitch = Dense(pitch_vocab_size, activation='softmax', name='pitch_output')(final_concat)
    output_velocity = Dense(velocity_vocab_size, activation='softmax', name='velocity_output')(final_concat)
    output_duration = Dense(duration_vocab_size, activation='softmax', name='duration_output')(final_concat)

    model = Model(inputs=[input_pitch, input_velocity, input_duration], outputs=[output_pitch, output_velocity, output_duration])

    model.compile(loss={'pitch_output': 'categorical_crossentropy', 
                        'velocity_output': 'categorical_crossentropy', 
                        'duration_output': 'categorical_crossentropy'},
                  optimizer=RMSprop(),
                  metrics={"pitch_output": "accuracy", 
                           "velocity_output": "accuracy", 
                           "duration_output": "accuracy"})

    model.summary()
    return model


def train(model, network_input_pitch, network_input_velocity, network_input_duration, network_output_pitch, network_output_velocity, network_output_duration):
    """ Train the neural network and plot loss and accuracy """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')

    early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)

    callbacks_list = [checkpoint, early_stopping]

    history = model.fit(
        [network_input_pitch, network_input_velocity, network_input_duration],
        {'pitch_output': network_output_pitch, 
         'velocity_output': network_output_velocity, 
         'duration_output': network_output_duration},
        epochs=200,
        batch_size=128,
        callbacks=callbacks_list
    )

    plot_loss_accuracy(history)

def plot_loss_accuracy(history):
    """ Plot the loss and accuracy during training """
    plt.figure(figsize=(18, 15))  # Adjust figure size as needed

    # Plot Combined Loss
    plt.subplot(4, 2, 4)
    plt.plot(history.history['loss'], 'b', label='Combined Training loss')
    plt.title('Combined Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Pitch Loss
    plt.subplot(4, 2, 1)
    plt.plot(history.history['pitch_output_loss'], 'r', label='Pitch Training Loss')
    plt.title('Pitch Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Velocity Loss
    plt.subplot(4, 2, 2)
    plt.plot(history.history['velocity_output_loss'], 'g', label='Velocity Training Loss')
    plt.title('Velocity Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Duration Loss
    plt.subplot(4, 2, 3)
    plt.plot(history.history['duration_output_loss'], 'y', label='Duration Training Loss')
    plt.title('Duration Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Pitch Accuracy
    plt.subplot(4, 2, 5)
    plt.plot(history.history['pitch_output_accuracy'], 'r', label='Pitch Training Accuracy')
    plt.title('Pitch Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Velocity Accuracy
    plt.subplot(4, 2, 6)
    plt.plot(history.history['velocity_output_accuracy'], 'g', label='Velocity Training Accuracy')
    plt.title('Velocity Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Duration Accuracy
    plt.subplot(4, 2, 7)
    plt.plot(history.history['duration_output_accuracy'], 'y', label='Duration Training Accuracy')
    plt.title('Duration Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    train_network()
