import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LSTM, Activation, BatchNormalization, Input
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt

def train_network():
    """ Train a Neural Network to generate music """
    notes, velocities = get_notes()

    # get amount of pitch names
    pitch_names = sorted(set(notes))
    velocity_names = sorted(set(velocities))

    pitch_vocab_size = len(pitch_names)
    velocity_vocab_size = len(velocity_names)

    network_input, network_output_pitch, network_output_velocity = prepare_sequences(notes, velocities, pitch_vocab_size)

    model = create_network(network_input, pitch_vocab_size, velocity_vocab_size)

    train(model, network_input, network_output_pitch, network_output_velocity)

def get_notes(velocity_quantization=20):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []
    velocities = []

    for file in glob.glob("midi_songsKEYCHANGE/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try:  # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:  # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                quantized_velocity = round(element.volume.velocity / velocity_quantization) * velocity_quantization
                notes.append(str(element.pitch))
                velocities.append(quantized_velocity)
                print("Notes,", str(element.pitch), quantized_velocity)
            elif isinstance(element, chord.Chord):
                # Simplify chord representation to a unique integer
                chord_name = '.'.join(str(pitch) for pitch in element.pitches)
                quantized_velocity = round(element.notes[0].volume.velocity / velocity_quantization) * velocity_quantization
                notes.append(chord_name)
                velocities.append(quantized_velocity)
                print("Chords,", chord_name, quantized_velocity)

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)
    with open('data/velocities', 'wb') as filepath:
        pickle.dump(velocities, filepath)

    return notes, velocities

def prepare_sequences(notes, velocities, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    pitch_names = sorted(set(notes))
    velocity_names = sorted(set(velocities))

    pitch_to_int = dict((note, number) for number, note in enumerate(pitch_names))
    velocity_to_int = dict((velocity, number) for number, velocity in enumerate(velocity_names))

    network_input = []
    network_output_pitch = []
    network_output_velocity = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        velocity_in = velocities[i:i + sequence_length]
        velocity_out = velocities[i + sequence_length]

        network_input.append([[pitch_to_int[char], velocity_to_int[vel]] for char, vel in zip(sequence_in, velocity_in)])
        network_output_pitch.append(pitch_to_int[sequence_out])
        network_output_velocity.append(velocity_to_int[velocity_out])

    n_patterns = len(network_input)

    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 2))

    pitch_vocab_size = len(pitch_to_int)
    velocity_vocab_size = len(velocity_to_int)

    network_input_pitch = network_input[:, :, 0] / float(pitch_vocab_size)
    network_input_velocity = network_input[:, :, 1] / float(velocity_vocab_size)

    network_input = numpy.stack((network_input_pitch, network_input_velocity), axis=2)

    network_output_pitch = to_categorical(network_output_pitch, num_classes=pitch_vocab_size)
    network_output_velocity = to_categorical(network_output_velocity, num_classes=velocity_vocab_size)

    return network_input, network_output_pitch, network_output_velocity

def create_network(network_input, pitch_vocab_size, velocity_vocab_size):
    """ Create the structure of the neural network """
    input_layer = Input(shape=(network_input.shape[1], network_input.shape[2]))

    lstm = LSTM(512, return_sequences=True)(input_layer)
    lstm = Dropout(0.3)(lstm)
    lstm = LSTM(512, return_sequences=True)(lstm)
    lstm = Dropout(0.3)(lstm)
    lstm = LSTM(512)(lstm)

    dense_pitch = Dense(256)(lstm)
    dense_pitch = Dropout(0.3)(dense_pitch)
    output_pitch = Dense(pitch_vocab_size, activation='softmax', name='pitch_output')(dense_pitch)

    dense_velocity = Dense(256)(lstm)
    dense_velocity = Dropout(0.3)(dense_velocity)
    output_velocity = Dense(velocity_vocab_size, activation='softmax', name='velocity_output')(dense_velocity)

    model = Model(inputs=input_layer, outputs=[output_pitch, output_velocity])

    model.compile(loss={'pitch_output': 'categorical_crossentropy', 'velocity_output': 'categorical_crossentropy'},
                  optimizer=RMSprop(),
                  metrics={"pitch_output": "accuracy", "velocity_output": "accuracy"})

    model.summary()
    return model

def train(model, network_input, network_output_pitch, network_output_velocity):
    """ Train the neural network and plot loss and accuracy """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-{pitch_output_loss:.4f}-{pitch_output_accuracy:.4f}-{velocity_output_loss:.4f}-{velocity_output_accuracy:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')

    early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)

    class PlotLossAccuracyCallback(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.pitch_losses = []
            self.velocity_losses = []
            self.pitch_accuracies = []
            self.velocity_accuracies = []

        def on_epoch_end(self, epoch, logs={}):
            self.losses.append(logs.get('loss'))
            self.pitch_losses.append(logs.get('pitch_output_loss'))
            self.velocity_losses.append(logs.get('velocity_output_loss'))
            self.pitch_accuracies.append(logs.get('pitch_output_accuracy'))
            self.velocity_accuracies.append(logs.get('velocity_output_accuracy'))

    plot_callback = PlotLossAccuracyCallback()

    callbacks_list = [checkpoint, plot_callback, early_stopping]

    history = model.fit(network_input,
                        {'pitch_output': network_output_pitch, 'velocity_output': network_output_velocity},
                        epochs=200,
                        batch_size=128,
                        callbacks=callbacks_list)

    plot_loss_accuracy(plot_callback.losses, plot_callback.pitch_losses, plot_callback.velocity_losses,
                       plot_callback.pitch_accuracies, plot_callback.velocity_accuracies)

def plot_loss_accuracy(losses, pitch_losses, velocity_losses, pitch_accuracies, velocity_accuracies):
    """ Plot the loss and accuracy during training """
    epochs = range(1, len(losses) + 1)

    plt.figure(figsize=(18, 9))

    # Plot Combined Loss
    plt.subplot(2, 3, 3)
    plt.plot(epochs, losses, 'b', label='Combined Training loss')
    plt.title('Combined Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Pitch Loss
    plt.subplot(2, 3, 1)
    plt.plot(epochs, pitch_losses, 'r', label='Pitch Training Loss')
    plt.title('Pitch Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Velocity Loss
    plt.subplot(2, 3, 2)
    plt.plot(epochs, velocity_losses, 'g', label='Velocity Training Loss')
    plt.title('Velocity Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Pitch Accuracy
    plt.subplot(2, 3, 4)
    plt.plot(epochs, pitch_accuracies, 'r', label='Pitch Training Accuracy')
    plt.title('Pitch Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Velocity Accuracy
    plt.subplot(2, 3, 5)
    plt.plot(epochs, velocity_accuracies, 'g', label='Velocity Training Accuracy')
    plt.title('Velocity Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    train_network()
