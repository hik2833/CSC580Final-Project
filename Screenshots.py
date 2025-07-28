"""
Screenshot and Results Saving Utilities
This module provides functionality for saving results and preparing data for assignment submission.
"""

import pandas as pd
from datetime import datetime
import json

def save_results_to_file(accuracy, model_config, predictions_df):
    """
    Save training results and model configuration to a formatted text file.
    
    Args:
        accuracy (float): Model accuracy percentage
        model_config (dict): Model configuration parameters
        predictions_df (pd.DataFrame): Sample predictions dataframe
    
    Returns:
        str: Formatted results content for download
    """
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    content = f"""
ENCODER-DECODER LSTM PORTFOLIO PROJECT RESULTS
============================================

Generated on: {timestamp}

MODEL CONFIGURATION
==================
Feature Cardinality: {model_config['n_features']} (including padding)
Input Sequence Length: {model_config['n_steps_in']}
Output Sequence Length: {model_config['n_steps_out']}
LSTM Units: {model_config['n_units']}
Training Epochs: {model_config['epochs']}
Training Samples: {model_config['n_samples']:,}

PERFORMANCE RESULTS
==================
Model Accuracy: {accuracy:.2f}%
Target Accuracy: >90% (Assignment Requirement)
Status: {'✅ PASSED' if accuracy >= 90 else '❌ NEEDS IMPROVEMENT'}

SAMPLE PREDICTIONS
==================
{predictions_df.to_string(index=False)}

IMPLEMENTATION SUMMARY
=====================
✅ Part 1: Encoder-Decoder Model in Keras - COMPLETED
   - define_models() function implemented
   - Training, inference encoder, and inference decoder models created
   - LSTM layers with configurable units
   - Proper model compilation

✅ Part 2: Scalable Sequence-to-Sequence Problem - COMPLETED  
   - generate_sequence() function implemented
   - get_dataset() function for training data generation
   - One-hot encoding and decoding utilities
   - Configurable problem parameters

✅ Part 3: Encoder-Decoder LSTM for Sequence Prediction - COMPLETED
   - predict_sequence() function implemented
   - Model training with progress tracking
   - Accuracy evaluation with test samples
   - Performance visualization

ASSIGNMENT DELIVERABLES
=======================
✅ Complete source code implementation
✅ Interactive Streamlit application
✅ Model architecture documentation
✅ Training progress visualization  
✅ Results summary and screenshots
✅ Performance evaluation metrics

TECHNICAL SPECIFICATIONS
========================
Framework: TensorFlow/Keras
Interface: Streamlit Web Application
Languages: Python 3.x
Libraries: NumPy, Pandas, Matplotlib, Plotly
Model Type: Sequence-to-Sequence LSTM Encoder-Decoder

For assignment submission, include:
1. Screenshots of this results page
2. Screenshots of model architecture  
3. Screenshots of training progress
4. This results summary file
5. Complete source code files

END OF REPORT
============
"""
    
    return content


def format_model_summary_for_display(model):
    """
    Format Keras model summary for better display in Streamlit.
    
    Args:
        model: Keras model object
    
    Returns:
        str: Formatted model summary
    """
    import io
    
    string_buffer = io.StringIO()
    model.summary(print_fn=lambda x: string_buffer.write(x + '\n'))
    return string_buffer.getvalue()


def create_results_json(accuracy, model_config, sample_predictions):
    """
    Create a JSON representation of results for structured data export.
    
    Args:
        accuracy (float): Model accuracy
        model_config (dict): Model configuration
        sample_predictions (list): List of sample prediction results
    
    Returns:
        str: JSON formatted results
    """
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "project": "Encoder-Decoder LSTM Portfolio Project",
        "performance": {
            "accuracy_percentage": accuracy,
            "target_accuracy": 90,
            "status": "PASSED" if accuracy >= 90 else "NEEDS_IMPROVEMENT"
        },
        "model_configuration": model_config,
        "sample_predictions": sample_predictions,
        "implementation_status": {
            "encoder_decoder_model": "COMPLETED",
            "scalable_problem": "COMPLETED", 
            "lstm_prediction": "COMPLETED",
            "evaluation": "COMPLETED"
        }
    }
    
    return json.dumps(results, indent=2)


def capture_screenshot_instructions():
    """
    Return instructions for manual screenshot capture.
    
    Returns:
        str: Screenshot capture instructions
    """
    
    instructions = """
    SCREENSHOT CAPTURE INSTRUCTIONS FOR ASSIGNMENT SUBMISSION
    ========================================================
    
    1. MODEL ARCHITECTURE SCREENSHOT:
       - Navigate to 'Model Architecture' tab
       - Capture the model summaries (Training, Encoder, Decoder)
       - Include the architecture flow diagram
    
    2. TRAINING RESULTS SCREENSHOT:
       - Navigate to 'Training & Evaluation' tab
       - Capture the training progress charts
       - Include the sample predictions table
       - Show the final accuracy metric
    
    3. RESULTS SUMMARY SCREENSHOT:
       - Navigate to 'Results & Screenshots' tab
       - Capture the performance gauge and metrics
       - Include the configuration JSON displays
       - Show the final results table
    
    4. CODE IMPLEMENTATION SCREENSHOT:
       - Capture the source code files (app.py, encoder_decoder.py, etc.)
       - Include function definitions and comments
       - Show the complete implementation
    
    5. ASSIGNMENT CHECKLIST SCREENSHOT:
       - Navigate to 'Assignment Summary' tab
       - Capture the completion checklist
       - Include the implementation summary
    
    BROWSER SCREENSHOT METHODS:
    - Windows: Win + Shift + S (Snipping Tool)
    - Mac: Cmd + Shift + 4 (Screenshot selection)
    - Chrome: F12 → Device toolbar → Capture screenshot
    - Firefox: F12 → Take a screenshot
    
    Save all screenshots in high resolution for clear documentation.
    """
    
    return instructions
