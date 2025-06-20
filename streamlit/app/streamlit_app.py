# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import spacy
from pathlib import Path
from sklearn.tree import plot_tree
import tensorflow as tf
# from tech_analyzer import TechAnalyzer


# Custom CSS for styling
st.set_page_config(
    page_title="AI Toolkit Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- STYLES ----
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("assets/styles.css")

# ---- MODEL LOADING ----
@st.cache_resource
def load_iris_model():
    with open(Path(r"models/iris_model.pkl"), 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_mnist_model():
    return tf.keras.models.load_model(Path(r"models/mnist_cnn.h5"))

import joblib

# Define TechAnalyzer first
class TechAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self._setup_tech_brands()
        self._setup_matchers()
    
    def _setup_tech_brands(self):
        self.tech_brands = ["TECNO", "M-Pesa", "Jumia", "Mara Phone", "Flutterwave",
                            "Safaricom", "MTN", "Airtel", "Konga", "Paystack"]
    
    def _setup_matchers(self):
        pass  # You can define the actual matchers here if needed

# Load the joblib-saved model
@st.cache_resource
def load_nlp_model():
    analyzer = joblib.load(Path("models/tech_analyzer.joblib"))
    return analyzer


# ---- HEADER ----
st.title("✨ AI Toolkit Explorer")
st.markdown("""
<div class="header">
    <p>Explore three powerful AI applications in one dashboard</p>
</div>
""", unsafe_allow_html=True)

# ---- SIDEBAR ----
st.sidebar.image("assets/images/logo.jpg", width=200)
app_mode = st.sidebar.radio(
    "Select Application",
    ["🏠 Home", "🌸 Iris Classifier", "✍️ Digit Recognizer", "🌍 Tech Analyzer"],
    index=0
)

# ---- HOME PAGE ----
if app_mode == "🏠 Home":
    col1, col2 = st.columns([1, 2])
    with col1:
        st.video("assets/ai_demo.mp4")
    with col2:
        st.markdown("""
        ## Welcome to Our AI Playground!
        
        This interactive dashboard showcases:
        
        - **Flower Classification** with Decision Trees  
        - **Handwritten Digit Recognition** using CNNs  
        - **African Tech Sentiment Analysis** with NLP  
        
        Select a demo from the sidebar to begin!
        """)
    
    st.markdown("---")
    st.subheader("🛠️ Our Toolkit")
    cols = st.columns(3)
    with cols[0]:
        st.image("assets/images/scikit.jpg", width=100)
        st.markdown("**scikit-learn**  \nClassical ML models")
    with cols[1]:
        st.image("assets/images/tensorflow.jpg", width=100)
        st.markdown("**TensorFlow**  \nDeep learning power")
    with cols[2]:
        st.image("assets/images/spacy.png", width=100)
        st.markdown("**spaCy**  \nIndustrial-strength NLP")

# ---- IRIS CLASSIFIER ----
elif app_mode == "🌸 Iris Classifier":
    st.header("🌸 Iris Flower Classifier")
    
    with st.expander("ℹ️ About this model"):
        st.markdown("""
        - **Model Type**: Decision Tree
        - **Accuracy**: 98% on test data
        - **Classes**: Setosa, Versicolor, Virginica
        """)
    
    # Input sliders
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
    with col2:
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)
    
    if st.button("🌻 Predict Species", type="primary"):
        try:
            model = load_iris_model()
            input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
            
            # Get prediction (handles both string and numeric outputs)
            prediction = model.predict(input_data)[0]
            
            # Create mapping for all possible outputs
            species_map = {
                'Iris-setosa': 'Setosa',
                'Iris-versicolor': 'Versicolor',
                'Iris-virginica': 'Virginica',
                0: 'Setosa',
                1: 'Versicolor',
                2: 'Virginica'
            }
            
            # Get the display name (works with both string and numeric predictions)
            species = species_map.get(prediction, "Unknown")
            
            st.success(f"Predicted Species: **{species}**")
                
            # Visualize decision tree
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_tree(model, 
                     feature_names=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
                     class_names=["Setosa", "Versicolor", "Virginica"],
                     filled=True, ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# ---- MNIST RECOGNIZER ----
elif app_mode == "✍️ Digit Recognizer":
    st.header("✍️ Handwritten Digit Recognizer")
    st.markdown("CNN model with >95% accuracy")
    
    upload_col, preview_col = st.columns([2, 1])
    with upload_col:
        uploaded_file = st.file_uploader("Upload digit image (28x28px)", 
                                       type=["png", "jpg", "jpeg"],
                                       accept_multiple_files=False)
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('L')
        image = image.resize((28, 28))
        
        with preview_col:
            st.image(image, caption="Your Digit", width=150)
        
        if st.button("🔍 Analyze Digit", type="primary"):
            img_array = np.array(image).reshape(1, 28, 28, 1) / 255.0
            model = load_mnist_model()
            prediction = model.predict(img_array)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction)
            
            st.success(f"Prediction: **{predicted_digit}** (Confidence: {confidence:.1%})")
            
            # Show prediction distribution
            st.subheader("Prediction Confidence")
            fig, ax = plt.subplots()
            ax.bar(range(10), prediction[0], color='#FF6B6B')
            ax.set_facecolor('#F7FFF7')
            ax.set_title("Model Confidence", pad=20)
            st.pyplot(fig)

# ---- TECH ANALYZER ----
elif app_mode == "🌍 Tech Analyzer":
    st.header("🌍 Tech Sentiment Analyzer")
    st.markdown("Analyze reviews of African tech products")
    
    review = st.text_area("Enter a review about an tech product/service:", 
                         "M-Pesa's mobile money service is revolutionary for Kenya!")
    
    if st.button("📊 Analyze Sentiment", type="primary"):
        analyzer = load_nlp_model()
        analysis = analyzer.analyze(review)
        
        # Display results in tabs
        tab1, tab2, tab3 = st.tabs(["Results", "Entities", "Sentiment"])
        
        with tab1:
            st.metric("Sentiment Score", 
                     f"{analysis['sentiment_score']:.2f}", 
                     "Positive" if analysis['sentiment_score'] >=0 else "Negative",
                     delta_color="off")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Positive Terms**")
                pos_terms = [t[0] for t in analysis['sentiment_indicators'] if t[1] == "positive"]
                st.write(pos_terms if pos_terms else "None found")
            with col2:
                st.write("**Negative Terms**")
                neg_terms = [t[0] for t in analysis['sentiment_indicators'] if t[1] == "negative"]
                st.write(neg_terms if neg_terms else "None found")
        
        with tab2:
            st.write("**Detected Tech Brands**")
            st.write([e[0] for e in analysis['entities']] or "None")
            
            st.write("**Mentioned Products**")
            st.write([p[0] for p in analysis['products']] or "None")
        
        with tab3:
            fig, ax = plt.subplots()
            pos_count = len([t for t in analysis['sentiment_indicators'] if t[1] == "positive"])
            neg_count = len([t for t in analysis['sentiment_indicators'] if t[1] == "negative"])
            
            ax.pie([pos_count, neg_count], 
                  labels=["Positive", "Negative"],
                  colors=["#4ECDC4", "#FF6B6B"],
                  autopct="%1.1f%%")
            ax.set_title("Sentiment Distribution")
            st.pyplot(fig)

# ---- FOOTER ----
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    text-align: center;
    padding: 10px;
    background-color: #0E1117;
    color: white;
}
</style>
<div class="footer">
    <p>© 2025 AI Toolkit Project</p>
</div>
""", unsafe_allow_html=True)