# CIFAR-10 Image Classifier

This is a PyTorch and Streamlit-based web application that uses a fine-tuned ResNet18 model to classify images into one of the 10 CIFAR-10 categories.

## Features
- Custom PyTorch training script (`train.py`) utilizing Transfer Learning with ResNet18.
- Sleek Streamlit UI (`app.py`) for uploading images and running inference.
- Displays the top 3 class predictions with a beautiful matplotlib bar chart.

## How to Run Locally

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model:**
   ```bash
   python train.py
   ```
   This will train the model and save the weights to `model/classifier.pth`.

3. **Run the Web App:**
   ```bash
   streamlit run app.py
   ```
