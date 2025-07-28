Encoder-Decoder LSTM Model Implementation
This module implements the core encoder-decoder model as specified in the assignment.
"""

import numpy as np
from numpy import array, argmax, array_equal
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

def define_models(n_input, n_output, n_units):
    """
    Define encoder-decoder recurrent neural network models.
    
    This function creates three models as specified in the assignment:
    1. Training model: Takes source and shifted target sequences
    2. Inference encoder: Encodes source sequence into hidden states  
    3. Inference decoder: Generates target sequence step by step
    
    Args:
        n_input (int): The cardinality of the input sequence (number of features, words, or characters per time step)
        n_output (int): The cardinality of the output sequence (number of features, words, or characters per time step)
        n_units (int): The number of cells to create in the encoder and decoder models (e.g., 128 or 256)
    
    Returns:
        tuple: (train_model, inference_encoder, inference_decoder)
            - train: Model that can be trained given source, target, and shifted target sequences
            - inference_encoder: Encoder model used when making predictions for new source sequences
            - inference_decoder: Decoder model used when making predictions for new source sequences
    """
    
    # Define training encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    
    # Define training decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Create training model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    # Define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    
    # Define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    
    # Return all models
    return model, encoder_model, decoder_model


def predict_sequence(infenc, infdec, source, n_steps, cardinality):
    """
    Generate target sequence given source sequence using trained encoder-decoder models.
    
    This function uses the inference encoder and decoder models to generate predictions
    step by step, as specified in the assignment.
    
    Args:
        infenc: Encoder model used when making a prediction for a new source sequence
        infdec: Decoder model used when making a prediction for a new source sequence  
        source: Encoded source sequence (one-hot encoded)
        n_steps (int): Number of time steps in the target sequence
        cardinality (int): The cardinality of the output sequence (number of features per time step)
    
    Returns:
        numpy.ndarray: Generated target sequence (one-hot encoded)
    """
    
    # Encode the source sequence
    state = infenc.predict(source, verbose=0)
    
    # Start of sequence input (all zeros except for padding token)
    target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
    
    # Collect predictions
    output = list()
    
    # Generate sequence step by step
    for t in range(n_steps):
        # Predict next character/token
        yhat, h, c = infdec.predict([target_seq] + state, verbose=0)
        
        # Store prediction
        output.append(yhat[0, 0, :])
        
        # Update state
        state = [h, c]
        
        # Update target sequence (use prediction as next input)
        target_seq = yhat
    
    return array(output)
