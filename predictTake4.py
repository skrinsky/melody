import numpy as np
import pickle
from music21 import instrument, note, chord, stream
from keras.models import load_model, Model
from keras.layers import Dense, Dropout, LSTM, Input
from keras.utils import to_categorical

def prepare_sequences(notes, velocities, durations, n_vocab_pitch, n_vocab_velocity, n_vocab_duration):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    pitch_names = sorted(set(notes))
    velocity_names = sorted(set(velocities))
    duration_values = sorted(set(durations))

    pitch_to_int = dict((note, number) for number, note in enumerate(pitch_names))
    velocity_to_int = dict((velocity, number) for number, velocity in enumerate(velocity_names))
    duration_to_int = dict((duration, number) for number, duration in enumerate(duration_values))

    network_input = []
    network_output_pitch = []
    network_output_velocity = []
    network_output_duration = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out_pitch = notes[i + sequence_length]
        sequence_out_velocity = velocities[i + sequence_length]
        sequence_out_duration = durations[i + sequence_length]

        network_input.append([[pitch_to_int[char], velocity_to_int[vel], duration_to_int[dur]] 
                              for char, vel, dur in zip(sequence_in, velocities[i:i + sequence_length], durations[i:i + sequence_length])])
        
        network_output_pitch.append(pitch_to_int[sequence_out_pitch])
        network_output_velocity.append(velocity_to_int[sequence_out_velocity])
        network_output_duration.append(duration_to_int[sequence_out_duration])

    n_patterns = len(network_input)

    network_input = np.reshape(network_input, (n_patterns, sequence_length, 3))

    network_input_pitch = network_input[:, :, 0] / float(n_vocab_pitch)
    network_input_velocity = network_input[:, :, 1] / float(n_vocab_velocity)
    network_input_duration = network_input[:, :, 2] / float(n_vocab_duration)

    network_input = np.stack((network_input_pitch, network_input_velocity, network_input_duration), axis=2)

    network_output_pitch = to_categorical(network_output_pitch, num_classes=n_vocab_pitch)
    network_output_velocity = to_categorical(network_output_velocity, num_classes=n_vocab_velocity)
    network_output_duration = to_categorical(network_output_duration, num_classes=n_vocab_duration)

    return network_input, network_output_pitch, network_output_velocity, network_output_duration

def create_network(network_input, pitch_vocab_size, velocity_vocab_size, duration_vocab_size):
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

    dense_duration = Dense(256)(lstm)
    dense_duration = Dropout(0.3)(dense_duration)
    output_duration = Dense(duration_vocab_size, activation='softmax', name='duration_output')(dense_duration)

    model = Model(inputs=input_layer, outputs=[output_pitch, output_velocity, output_duration])

    model.compile(loss={'pitch_output': 'categorical_crossentropy', 
                        'velocity_output': 'categorical_crossentropy', 
                        'duration_output': 'categorical_crossentropy'},
                  optimizer='rmsprop',
                  metrics={"pitch_output": "accuracy", 
                           "velocity_output": "accuracy", 
                           "duration_output": "accuracy"})

    return model

def generate_notes(model, network_input, pitch_names, velocity_names, duration_values, n_vocab_pitch, n_vocab_velocity, n_vocab_duration):
    """ Generate notes from the neural network based on a sequence of notes """
    start = np.random.randint(0, len(network_input)-1)

    int_to_pitch = dict((number, note) for number, note in enumerate(pitch_names))
    int_to_velocity = dict((number, velocity) for number, velocity in enumerate(velocity_names))
    int_to_duration = dict((number, duration) for number, duration in enumerate(duration_values))

    pattern = network_input[start]
    prediction_output = []

    print("Generating notes...")

    # generate 500 notes
    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 3))
        prediction_input_pitch = prediction_input[:,:,0] / float(n_vocab_pitch)
        prediction_input_velocity = prediction_input[:,:,1] / float(n_vocab_velocity)
        prediction_input_duration = prediction_input[:,:,2] / float(n_vocab_duration)
        
        prediction_input = np.stack((prediction_input_pitch, prediction_input_velocity, prediction_input_duration), axis=2)
        
        prediction = model.predict(prediction_input, verbose=0)
        index_pitch = np.argmax(prediction[0])
        index_velocity = np.argmax(prediction[1])
        index_duration = np.argmax(prediction[2])

        result_pitch = int_to_pitch[index_pitch]
        result_velocity = int_to_velocity[index_velocity]
        result_duration = int_to_duration[index_duration]

        print(f"Prediction {note_index+1}: Pitch - {result_pitch}, Velocity - {result_velocity}, Duration - {result_duration}")

        prediction_output.append((result_pitch, result_velocity, result_duration))

        pattern = np.append(pattern, [[index_pitch, index_velocity, index_duration]], axis=0)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def create_midi(prediction_output, output_filename='test_output.mid', velocity_quantization=20):
    """ Convert the output from the prediction to notes and create a midi file from the notes """
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        pitch, velocity, duration = pattern
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
                new_note.quarterLength = float(duration)
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pitch)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            new_note.volume.velocity = velocity
            new_note.quarterLength = float(duration)
            output_notes.append(new_note)

        offset += float(duration)

    midi_stream = stream.Stream(output_notes)
    
    try:
        midi_stream.write('midi', fp=output_filename)
        print(f"MIDI file created successfully: {output_filename}")
    except Exception as e:
        print(f"Error while creating MIDI file: {e}")

def generate_midi(weights_file, notes_file, velocities_file, durations_file, output_filename='test_outputnow.mid'):
    """ Generate a MIDI file using the trained model """
    with open(notes_file, 'rb') as filepath:
        notes = pickle.load(filepath)
    with open(velocities_file, 'rb') as filepath:
        velocities = pickle.load(filepath)
    with open(durations_file, 'rb') as filepath:
        durations = pickle.load(filepath)

    pitch_names = sorted(set(notes))
    velocity_names = sorted(set(velocities))
    duration_values = sorted(set(durations))

    n_vocab_pitch = len(pitch_names)
    n_vocab_velocity = len(velocity_names)
    n_vocab_duration = len(duration_values)

    network_input, _, _, _ = prepare_sequences(notes, velocities, durations, n_vocab_pitch, n_vocab_velocity, n_vocab_duration)

    model = create_network(network_input, n_vocab_pitch, n_vocab_velocity, n_vocab_duration)
    model.load_weights(weights_file)

    prediction_output = generate_notes(model, network_input, pitch_names, velocity_names, duration_values, n_vocab_pitch, n_vocab_velocity, n_vocab_duration)
    create_midi(prediction_output, output_filename)

if __name__ == '__main__':
    generate_midi('weights-improvement-91-1.5381-1.2598-0.7363-0.1112-0.9596-0.1671-0.9427-bigger.hdf5', 'data/notes', 'data/velocities', 'data/durations')
