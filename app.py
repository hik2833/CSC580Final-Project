# CSC580Final-Project
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io
import base64
from PIL import Image
import time

# Import our custom modules
from encoder_decoder import define_models, predict_sequence
from sequence_utils import generate_sequence, get_dataset, one_hot_decode
from screenshot_utils import capture_screenshot, save_results_to_file

# Configure page
st.set_page_config(
    page_title="Encoder-Decoder LSTM Portfolio Project",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🧠 Encoder-Decoder LSTM for Sequence-to-Sequence Prediction")
    st.markdown("### Portfolio Project Implementation")
    st.markdown("This application implements an encoder-decoder model for sequence-to-sequence prediction using Keras/TensorFlow.")
    
    # Sidebar for configuration
    st.sidebar.header("📊 Model Configuration")
    
    # Problem configuration
    st.sidebar.subheader("Problem Parameters")
    n_features_base = st.sidebar.slider("Feature Cardinality (base)", 10, 100, 50, help="Number of unique features (excluding padding)")
    n_features = n_features_base + 1  # +1 for padding/start-of-sequence
    
    n_steps_in = st.sidebar.slider("Input Sequence Length", 3, 10, 6, help="Length of source sequence")
    n_steps_out = st.sidebar.slider("Output Sequence Length", 2, 6, 3, help="Length of target sequence")
    
    # Model configuration
    st.sidebar.subheader("Model Parameters")
    n_units = st.sidebar.selectbox("LSTM Units", [64, 128, 256], index=1, help="Number of LSTM units in encoder/decoder")
    epochs = st.sidebar.slider("Training Epochs", 10, 100, 50, help="Number of training epochs")
    n_samples = st.sidebar.slider("Training Samples", 1000, 10000, 5000, help="Number of training samples")
    
    # Evaluation parameters
    st.sidebar.subheader("Evaluation Parameters")
    eval_samples = st.sidebar.slider("Evaluation Samples", 50, 200, 100, help="Number of samples for accuracy evaluation")
    
    # Random seed for reproducibility
    random_seed = st.sidebar.number_input("Random Seed", 0, 9999, 42, help="For reproducible results")
    np.random.seed(random_seed)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Problem Overview", "🏗️ Model Architecture", "🔄 Training & Evaluation", "📊 Results & Screenshots", "📋 Assignment Summary"])
    
    with tab1:
        st.header("Problem Overview")
        st.markdown("""
        ### Sequence-to-Sequence Problem Definition
        
        **Source Sequence**: Random integers (e.g., [20, 36, 40, 10, 34, 28])
        
        **Target Sequence**: First n elements reversed (e.g., [40, 36, 20])
        
        **Task**: Train an encoder-decoder LSTM to learn this transformation.
        """)
        
        # Display current configuration
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"""
            **Current Configuration:**
            - Feature Cardinality: {n_features} (including padding)
            - Input Length: {n_steps_in}
            - Output Length: {n_steps_out}
            - LSTM Units: {n_units}
            """)
        
        with col2:
            st.info(f"""
            **Training Setup:**
            - Epochs: {epochs}
            - Training Samples: {n_samples}
            - Evaluation Samples: {eval_samples}
            - Random Seed: {random_seed}
            """)
        
        # Example demonstration
        st.subheader("Example Sequences")
        if st.button("Generate Example Sequences"):
            examples_data = []
            for i in range(5):
                source = generate_sequence(n_steps_in, n_features)
                target = source[:n_steps_out]
                target.reverse()
                examples_data.append({
                    "Example": i+1,
                    "Source Sequence": str(source),
                    "Target Sequence": str(target)
                })
            
            examples_df = pd.DataFrame(examples_data)
            st.dataframe(examples_df, use_container_width=True)
    
    with tab2:
        st.header("Model Architecture")
        st.markdown("""
        ### Encoder-Decoder LSTM Architecture
        
        The model consists of three components:
        1. **Training Model**: Takes source and shifted target sequences
        2. **Inference Encoder**: Encodes source sequence into hidden states
        3. **Inference Decoder**: Generates target sequence step by step
        """)
        
        # Model definition and summary
        if st.button("Define and Display Model Architecture"):
            with st.spinner("Creating model architecture..."):
                try:
                    # Import TensorFlow/Keras here to avoid initial loading delay
                    import tensorflow as tf
                    from tensorflow.keras.models import Model
                    from tensorflow.keras.layers import Input, LSTM, Dense
                    
                    # Set TensorFlow to suppress warnings
                    tf.get_logger().setLevel('ERROR')
                    
                    train_model, encoder_model, decoder_model = define_models(n_features, n_features, n_units)
                    
                    st.success("✅ Model architecture created successfully!")
                    
                    # Display model summaries
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("Training Model")
                        # Capture model summary
                        string_buffer = io.StringIO()
                        train_model.summary(print_fn=lambda x: string_buffer.write(x + '\n'))
                        model_summary = string_buffer.getvalue()
                        st.text(model_summary)
                    
                    with col2:
                        st.subheader("Encoder Model")
                        string_buffer = io.StringIO()
                        encoder_model.summary(print_fn=lambda x: string_buffer.write(x + '\n'))
                        encoder_summary = string_buffer.getvalue()
                        st.text(encoder_summary)
                    
                    with col3:
                        st.subheader("Decoder Model")
                        string_buffer = io.StringIO()
                        decoder_model.summary(print_fn=lambda x: string_buffer.write(x + '\n'))
                        decoder_summary = string_buffer.getvalue()
                        st.text(decoder_summary)
                    
                    # Store models in session state for later use
                    st.session_state['train_model'] = train_model
                    st.session_state['encoder_model'] = encoder_model
                    st.session_state['decoder_model'] = decoder_model
                    st.session_state['model_created'] = True
                    
                except Exception as e:
                    st.error(f"Error creating model: {str(e)}")
        
        # Model architecture diagram description
        st.subheader("Architecture Flow")
        st.markdown("""
        
