"""
Sequence Generation and Processing Utilities
This module implements sequence generation, dataset creation, and encoding utilities as specified in the assignment.
"""

import numpy as np
from numpy import array, argmax
from numpy.random import randint
from tensorflow.keras.utils import to_categorical

def generate_sequence(length, n_unique):
    """
    Generate a sequence of random integers.
    
    As specified in the assignment, this function generates random integers
    in the range [1, n_unique-1] to avoid using 0 (reserved for padding).
    
    Args:
        length (int): Length of the sequence to generate
        n_unique (int): Number of unique values (cardinality), 0 is reserved for padding
    
    Returns:
        list: Sequence of random integers
    """
    return [randint(1, n_unique-1) for _ in range(length)]


def get_dataset(n_in, n_out, cardinality, n_samples):
    """
    Prepare training data for the LSTM model.
    
    This function generates the specified number of sequence pairs for training,
    where each target sequence is the first n_out elements of the source sequence in reverse order.
    
    Args:
        n_in (int): Length of input/source sequences
        n_out (int): Length of output/target sequences  
        cardinality (int): Number of unique values in sequences (including padding)
        n_samples (int): Number of training samples to generate
    
    Returns:
        tuple: (X1, X2, y) where:
            - X1: Source sequences (one-hot encoded)
            - X2: Shifted target sequences for training (one-hot encoded)  
            - y: Target sequences (one-hot encoded)
    """
    
    X1, X2, y = list(), list(), list()
    
    for _ in range(n_samples):
        # Generate source sequence
        source = generate_sequence(n_in, cardinality)
        
        # Define target sequence (first n_out elements reversed)
        target = source[:n_out]
        target.reverse()
        
        # Create padded input target sequence (shifted by one with start token)
        target_in = [0] + target[:-1]
        
        # One-hot encode all sequences
        src_encoded = to_categorical([source], num_classes=cardinality)
        tar_encoded = to_categorical([target], num_classes=cardinality)
        tar2_encoded = to_categorical([target_in], num_classes=cardinality)
        
        # Store encoded sequences
        X1.append(src_encoded)
        X2.append(tar2_encoded)
        y.append(tar_encoded)
    
    return array(X1), array(X2), array(y)


def one_hot_decode(encoded_seq):
    """
    Decode a one-hot encoded sequence back to integer sequence.
    
    This function converts one-hot encoded vectors back to their integer representations
    for easy reading and comparison of results.
    
    Args:
        encoded_seq: One-hot encoded sequence (2D array)
    
    Returns:
        list: Decoded integer sequence
    """
    return [argmax(vector) for vector in encoded_seq]


def evaluate_model_accuracy(encoder_model, decoder_model, n_steps_in, n_steps_out, n_features, n_samples=100):
    """
    Evaluate model accuracy on randomly generated test sequences.
    
    Args:
        encoder_model: Trained inference encoder model
        decoder_model: Trained inference decoder model
        n_steps_in (int): Length of input sequences
        n_steps_out (int): Length of output sequences
        n_features (int): Number of features (cardinality)
        n_samples (int): Number of test samples to evaluate
    
    Returns:
        tuple: (accuracy_percentage, sample_results) where sample_results contains prediction examples
    """
    from encoder_decoder import predict_sequence
    
    total, correct = n_samples, 0
    sample_results = []
    
    for i in range(total):
        # Generate test sequence
        X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
        
        # Reshape for prediction
        X1_reshaped = X1.reshape(1, n_steps_in, n_features)
        
        # Make prediction
        target = predict_sequence(encoder_model, decoder_model, X1_reshaped, n_steps_out, n_features)
        
        # Decode sequences for comparison
        source_decoded = one_hot_decode(X1[0].reshape(n_steps_in, n_features))
        expected_decoded = one_hot_decode(y[0].reshape(n_steps_out, n_features))
        predicted_decoded = one_hot_decode(target)
        
        # Check if prediction is correct
        is_correct = np.array_equal(expected_decoded, predicted_decoded)
        if is_correct:
            correct += 1
        
        # Store sample results for first few predictions
        if i < 20:
            sample_results.append({
                'source': source_decoded,
                'expected': expected_decoded,
                'predicted': predicted_decoded,
                'correct': is_correct
            })
    
    accuracy = (correct / total) * 100
    return accuracy, sample_results
