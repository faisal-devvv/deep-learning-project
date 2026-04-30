import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import resnet18
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define CIFAR-10 classes
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Page config for better UI
st.set_page_config(page_title="CIFAR-10 Classifier", page_icon="🖼️", layout="centered")

@st.cache_resource
def load_model():
    # Initialize the model structure
    model = resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    
    # Load weights
    try:
        model.load_state_dict(torch.load("model/classifier.pth", map_location=torch.device('cpu'), weights_only=True))
        model.eval()
        return model
    except FileNotFoundError:
        return None

def preprocess_image(image):
    # Preprocess as requested: resize to 32x32, convert to tensor, normalize
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def main():
    # Sidebar
    st.sidebar.title("🧠 Model Info")
    st.sidebar.markdown("---")
    st.sidebar.info("**Architecture:** ResNet18")
    st.sidebar.info("**Dataset:** CIFAR-10")
    st.sidebar.info("**Accuracy:** ~73.6%")
    st.sidebar.markdown("---")
    st.sidebar.write("Built with Streamlit & PyTorch")

    st.title("🖼️ CIFAR-10 Image Classifier")
    st.write("Upload an image to see the model's top 3 predictions across the 10 CIFAR-10 classes.")

    model = load_model()
    if model is None:
        st.warning("Model weights not found. Please wait for `train.py` to finish and generate `model/classifier.pth`.")
        st.stop()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption='Uploaded Image', use_container_width=True)
            
        with col2:
            st.markdown("### Top 3 Predictions")
            with st.spinner("Classifying..."):
                # Preprocess and predict
                input_tensor = preprocess_image(image)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = F.softmax(output[0], dim=0)
                    
                    # Get top 3 predictions
                    top3_prob, top3_idx = torch.topk(probabilities, 3)
                
                top3_classes = [CLASSES[idx.item()] for idx in top3_idx]
                top3_probs = [prob.item() * 100 for prob in top3_prob]
                
                # Display best result
                st.success(f"🥇 **{top3_classes[0].capitalize()}** ({top3_probs[0]:.2f}%)")
                
                # Matplotlib Bar Chart
                fig, ax = plt.subplots(figsize=(5, 3))
                colors = ['#4CAF50', '#2196F3', '#FFC107']
                bars = ax.barh(top3_classes[::-1], top3_probs[::-1], color=colors[::-1])
                ax.set_xlabel('Confidence (%)')
                ax.set_xlim(0, 100)
                
                # Add labels to bars
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                            va='center', ha='left', fontsize=10)
                
                # Remove borders
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                st.pyplot(fig)

if __name__ == "__main__":
    main()
