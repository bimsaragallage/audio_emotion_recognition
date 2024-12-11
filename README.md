## Introduction

### Purpose
This project presents a novel approach to audio emotion recognition using Variational Autoencoders. Audio emotion recognition is a significant task in speech processing, with applications in areas such as human-computer interaction, mental health monitoring, and sentiment analysis.

### Dataset
The model is trained and evaluated on the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset, which contains emotional speech recordings. The dataset can be accessed using the following link:  
[RAVDESS Dataset on Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)  

### Features
To extract meaningful information from audio signals, the following features were used:  
- **MFCC (Mel Frequency Cepstral Coefficients)**: Captures the short-term power spectrum of audio signals, commonly used in speech and audio processing.  
- **Pitch**: Extracted as the highest fundamental frequency in each frame, providing insights into the emotional tone of speech.  
- **Band Energy Ratio (BER)**: Measures energy distribution across frequency bands to identify speech dynamics.  
- **Spectral Centroid**: Represents the "center of mass" of the audio spectrum, related to the perceived brightness of the sound.

### Approach
The Variational Autoencoder architecture leverages advanced encoding techniques to capture the underlying emotional features from audio signals. Unlike traditional methods, this approach ensures better representation of vibrational patterns in speech, leading to improved recognition accuracy.  

This project showcases the potential of Variational-based encoding in emotion recognition, offering an innovative perspective to advance the field.

## Installation

### Requirements
Ensure you have Python 3.11 installed. The following libraries are required for this project:  
- `numpy`  
- `pandas`  
- `matplotlib`  
- `seaborn`  
- `scikit-learn`  
- `librosa`  
- `tensorflow`  
- `keras`  

### Steps
1. **Clone the repository**  
   Download the project repository using the following command:  
   ```bash
   git clone <repository_url>
   cd <repository_name>

## Project Description  

This project explores a novel approach to **Audio Emotion Recognition** using **Variational Autoencoders (VAEs)**. The main objective is to classify emotions in audio samples while leveraging the generative capabilities of VAEs to improve feature representation and model performance.  

### Variational Autoencoders for Audio Emotion Recognition  

**Variational Autoencoders (VAEs)** are generative models that learn to encode data into a lower-dimensional latent space and decode it back to the original space. They are particularly effective for tasks involving feature learning, data compression, and anomaly detection.  

#### Key Components of VAEs:  

1. **Encoder (Recognition Model)**:  
   The encoder maps input audio features (e.g., MFCC, pitch, spectral centroid) into a latent space, where each input is represented as a distribution (mean and variance) rather than a single point.  

2. **Latent Space (Sampling)**:  
   Using the reparameterization trick, the model samples latent variables from a standard normal distribution, ensuring the latent space is smooth and enables gradient descent optimization.  

3. **Decoder (Generative Model)**:  
   The decoder maps the latent variables back to reconstruct the input audio features, helping to maintain meaningful and robust feature representations.  

4. **Loss Function**:  
   The VAE loss comprises two components:  
   - **Reconstruction Loss**: Ensures the reconstructed data is close to the input.  
   - **KL Divergence Loss**: Regularizes the latent space by aligning the learned distribution with a standard normal distribution.  

### Novelty of Approach  

- Incorporates VAEs to create a more robust latent representation of audio features, capturing both the temporal and frequency characteristics of emotional speech.  
- Combines traditional audio features (MFCC, pitch, spectral centroid) with latent embeddings from VAEs for improved emotion classification.  
- Focuses on lightweight models suitable for real-time applications in mobile and embedded systems.  

### Methodology Overview  

1. **Feature Extraction**: Extracted features include:  
   - **MFCC**: Captures the power spectrum of speech.  
   - **Pitch**: Highlights the fundamental frequency in audio frames.  
   - **Band Energy Ratio**: Represents the energy distribution across frequency bands.  
   - **Spectral Centroid**: Identifies the "center of mass" of the spectrum.  

2. **Model Design**:  
   - The VAE architecture comprises an encoder, a latent space sampler, and a decoder.  
   - Features from the VAE's latent space are combined with extracted features for emotion classification.  

3. **Evaluation Metrics**:  
   - Classification accuracy for emotional states.  
   - Reconstruction quality using VAE loss.  

4. **Dataset**:  
   - **RAVDESS** dataset for emotional speech audio, which contains 24 professional actors performing scripted sentences with various emotions.  

This project demonstrates the potential of VAEs in improving audio emotion recognition accuracy and robustness by creating meaningful latent representations of speech features.

## Workflow, Latent Representation, and Results  

### 1. Workflow  
The workflow for the Audio Emotion Recognition model is illustrated below:  

![diagram](https://github.com/user-attachments/assets/cfd904a9-4874-49ef-b21f-9f32639a6f78)

This diagram outlines the end-to-end pipeline, from data preprocessing and feature extraction to VAE training and emotion classification.  

---

### 2. Latent Representation  
The latent space representation learned by the Variational Autoencoder captures the intricate patterns in the audio features. The following visualization demonstrates the clustering of emotional states within the latent space:  

<img width="373" alt="Screenshot 2024-12-04 120848" src="https://github.com/user-attachments/assets/f9822384-26be-4ee2-916b-5346735190d7">

- The latent space demonstrates meaningful separation of emotional states, enabling the model to effectively capture the underlying structure of the data.  

---

### 3. Confusion Matrix  
The final classification performance is summarized in the confusion matrix:  

![output](https://github.com/user-attachments/assets/5db83584-1e79-446b-bc89-d7abdf5f50f6)

- The confusion matrix highlights the model's accuracy across different emotional categories, showcasing its strengths and areas for improvement.  

---

### Key Observations  
- **Workflow**: The end-to-end process integrates feature extraction, latent representation learning, and emotion classification seamlessly.  
- **Latent Representation**: The clustering in the latent space confirms the effectiveness of the VAE in learning distinct features for different emotions.  
- **Confusion Matrix**: reasonable accuracy is observed in most categories, though some overlap exists in closely related emotions (e.g., calm vs. neutral).  



