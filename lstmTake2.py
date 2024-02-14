#more info here https://wordpress.com/view/summerliketheseasoncom.wordpress.com
#listen to generated sounds https://soundcloud.com/summer-554845429/sets
""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras import callbacks
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop


def train_network():
    """ Train a Neural Network to generate music """
    notes = get_notes()

    # get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)

def get_notes(velocity_quantization=20):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    for file in glob.glob("midi_songsKEYCHANGE/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                # Include pitch and normalized velocity (quantized)
                quantized_velocity = round(element.volume.velocity / velocity_quantization) * velocity_quantization
                note_str = f"{str(element.pitch)}_{quantized_velocity}"
                notes.append(note_str)
                print("Notes,", note_str)
            elif isinstance(element, chord.Chord):
                # Include chord representation with normalized velocity (quantized)
                quantized_velocities = [round(n.volume.velocity / velocity_quantization) * velocity_quantization for n in element]
                chord_str = '.'.join(f"{str(pitch)}_{n}" for pitch, n in zip(element.pitches, quantized_velocities))
                notes.append(chord_str)
                print("Chords,", chord_str)

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all unique pitch and velocity combinations
    pitch_velocity_names = sorted(set(item for item in notes))

    # create a dictionary to map pitch and velocity to integers
    pitch_velocity_to_int = dict((note, number) for number, note in enumerate(pitch_velocity_names))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([pitch_velocity_to_int[char] for char in sequence_in])
        network_output.append(pitch_velocity_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = to_categorical(network_output)
    
    return (network_input, network_output)

def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    
    """Create a simpler version of the neural network"""
    
    model = Sequential()
    model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))  # Reduce the number of units in the dense layer
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    # Change learning rate in RMSprop optimizer
    model.compile(loss='categorical_crossentropy', optimizer= 'rmsprop', metrics=["accuracy"])

    model.summary()
    return model



def train(model, network_input, network_output):
    """ Train the neural network and plot loss and accuracy """

    # Define filepath for saving the best weights
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-{accuracy:.4f}-bigger.hdf5"
    checkpoint = callbacks.ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    
    # Define callback for plotting
    class PlotLossAccuracyCallback(callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.accuracies = []

        def on_epoch_end(self, epoch, logs={}):
            self.losses.append(logs.get('loss'))
            self.accuracies.append(logs.get('accuracy'))

    plot_callback = PlotLossAccuracyCallback()

    callbacks_list = [checkpoint, plot_callback]

    # Train the model
    history = model.fit(network_input, network_output, epochs=200, batch_size=256, callbacks=callbacks_list)

    # Plot loss and accuracy after training
    plot_loss_accuracy(plot_callback.losses, plot_callback.accuracies)

def plot_loss_accuracy(losses, accuracies):
    """ Plot the loss and accuracy during training """
    epochs = range(1, len(losses) + 1)

    # Plot Loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, 'b', label='Training loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, 'r', label='Training accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
    

if __name__ == '__main__':
    train_network()
    
