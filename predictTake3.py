import numpy as np
import pickle
from music21 import instrument, note, chord, stream
from keras.models import load_model, Model
from keras.layers import Dense, Dropout, LSTM, Input
from keras.utils import to_categorical

def prepare_sequences(notes, velocities, n_vocab_pitch, n_vocab_velocity):
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

    network_input = np.reshape(network_input, (n_patterns, sequence_length, 2))

    network_input_pitch = network_input[:, :, 0] / float(n_vocab_pitch)
    network_input_velocity = network_input[:, :, 1] / float(n_vocab_velocity)

    network_input = np.stack((network_input_pitch, network_input_velocity), axis=2)

    network_output_pitch = to_categorical(network_output_pitch, num_classes=n_vocab_pitch)
    network_output_velocity = to_categorical(network_output_velocity, num_classes=n_vocab_velocity)

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
                  optimizer='rmsprop',
                  metrics={"pitch_output": "accuracy", "velocity_output": "accuracy"})

    return model

def generate_notes(model, network_input, pitch_names, velocity_names, n_vocab_pitch, n_vocab_velocity):
    """ Generate notes from the neural network based on a sequence of notes """
    start = np.random.randint(0, len(network_input)-1)

    int_to_pitch = dict((number, note) for number, note in enumerate(pitch_names))
    int_to_velocity = dict((number, velocity) for number, velocity in enumerate(velocity_names))

    pattern = network_input[start]
    prediction_output = []

    print("Generating notes...")

    # generate 500 notes
    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 2))
        prediction_input = np.stack((prediction_input[:,:,0] / float(n_vocab_pitch), 
                                     prediction_input[:,:,1] / float(n_vocab_velocity)), axis=2)
        
        prediction = model.predict(prediction_input, verbose=0)
        index_pitch = np.argmax(prediction[0])
        index_velocity = np.argmax(prediction[1])

        result_pitch = int_to_pitch[index_pitch]
        result_velocity = int_to_velocity[index_velocity]

        print(f"Prediction {note_index+1}: Pitch - {result_pitch}, Velocity - {result_velocity}")

        prediction_output.append((result_pitch, result_velocity))

        pattern = np.append(pattern, [[index_pitch, index_velocity]], axis=0)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def create_midi(prediction_output, output_filename='test_output.mid', velocity_quantization=20):
    """ Convert the output from the prediction to notes and create a midi file from the notes """
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        pitch, velocity = pattern
        velocity = int(velocity)

        if ('.' in pitch) or pitch.isdigit():
            notes_in_chord = pitch.split('.')
            notes = []
            for current_note in notes_in_chord:
                if current_note.isdigit():
                    new_note = note.Note(int(current_note))
                else:
                    new_note = note.Note(current_note)
                new_note.storedInstrument = instrument.Piano()
                new_note.volume.velocity = velocity
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pitch)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            new_note.volume.velocity = velocity
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    
    try:
        midi_stream.write('midi', fp=output_filename)
        print(f"MIDI file created successfully: {output_filename}")
    except Exception as e:
        print(f"Error while creating MIDI file: {e}")

def generate_midi(weights_file, notes_file, velocities_file, output_filename='test_outputnow.mid'):
    """ Generate a MIDI file using the trained model """
    with open(notes_file, 'rb') as filepath:
        notes = pickle.load(filepath)
    with open(velocities_file, 'rb') as filepath:
        velocities = pickle.load(filepath)

    pitch_names = sorted(set(notes))
    velocity_names = sorted(set(velocities))

    n_vocab_pitch = len(pitch_names)
    n_vocab_velocity = len(velocity_names)

    network_input, _, _ = prepare_sequences(notes, velocities, n_vocab_pitch, n_vocab_velocity)

    model = create_network(network_input, n_vocab_pitch, n_vocab_velocity)
    model.load_weights(weights_file)

    prediction_output = generate_notes(model, network_input, pitch_names, velocity_names, n_vocab_pitch, n_vocab_velocity)
    create_midi(prediction_output, output_filename)

if __name__ == '__main__':
    generate_midi('weights-improvement-79-1.3697-1.3009-0.7213-0.0688-0.9769-bigger.hdf5', 'data/notes', 'data/velocities')
