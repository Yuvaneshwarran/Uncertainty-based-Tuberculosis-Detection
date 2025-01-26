# Tuberculosis Image Classification using Bayesian CNN

## Overview
This project involves building a deep learning model for classifying tuberculosis (TB) images using a Bayesian Convolutional Neural Network (CNN). The model leverages **Monte Carlo Dropout** for uncertainty estimation and **SMOTE (Synthetic Minority Over-sampling Technique)** to address class imbalance. The project is implemented using **Streamlit** for the user interface and integrates key tools and techniques for medical image classification.

---

## Features
- **Bayesian CNN with Monte Carlo Dropout**: Provides uncertainty estimation for model predictions.
- **Class Imbalance Handling**: Uses SMOTE for oversampling minority classes, improving the model's robustness.
- **Streamlit UI**: Intuitive and interactive interface for users to upload and classify TB images.
- **Preprocessing and Visualization**: Includes image preprocessing and visualization for better interpretability.
- **Metrics and Evaluation**: Supports performance evaluation using metrics such as accuracy, Dice Score, and IoU (Intersection over Union).

---

## Project Workflow
1. **Data Preprocessing**:
   - Images are resized, normalized, and augmented.
   - SMOTE is applied to address class imbalance.

2. **Model Architecture**:
   - Bayesian CNN is used to incorporate Monte Carlo Dropout during inference for uncertainty estimation.
   - TensorFlow is used for model implementation.

3. **Loss Functions**:
   - Dice Loss
   - Binary Cross-Entropy
   - Intersection over Union (IoU)

4. **User Interface**:
   - Built using Streamlit for easy image upload and classification.
   - Displays predictions along with confidence scores and uncertainty estimates.

5. **Evaluation**:
   - Metrics include accuracy, Dice Score, IoU, and F1-score.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/tuberculosis-bayesian-cnn.git
   cd tuberculosis-bayesian-cnn
   ```

2. Install dependencies:
   Create a virtual environment and activate it (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate   # For Windows
   ```
   Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## Requirements
Key dependencies for the project include:
- `streamlit==1.24.1`
- `tensorflow>=2.12.0`
- `opencv-python==4.8.0.76`
- `pandas==1.5.3`
- `scikit-learn==1.2.2`
- `matplotlib==3.6.3`
- `protobuf==3.20.3`

For a full list, see `requirements.txt`.

---

## Usage
1. Upload a chest X-ray image using the Streamlit interface.
2. The model processes the image and predicts the likelihood of tuberculosis.
3. View the classification result along with the confidence score and uncertainty estimate.
4. Analyze the performance metrics and uncertainty visualization.

---

## Project Structure
```
.
├── app.py                  # Streamlit UI implementation
├── model.py                # Bayesian CNN model definition
├── preprocessing.py        # Data preprocessing and augmentation
├── utils.py                # Utility functions for metrics and visualization
├── requirements.txt        # List of dependencies
├── README.md               # Project documentation
└── datasets/               # Contains training and testing data
```

---

## Future Improvements
- **Integration of Additional Models**: Explore architectures like U-Net or Mask R-CNN for enhanced segmentation capabilities.
- **Automated Hyperparameter Tuning**: Incorporate tools like Optuna or Ray Tune.
- **Deployment**: Deploy the application on platforms like AWS, Heroku, or Google Cloud for broader accessibility.
- **Advanced Visualizations**: Add Grad-CAM or saliency map visualizations to improve interpretability.

---

## Contributions
Contributions are welcome! If you'd like to contribute, please:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments
- TensorFlow for the deep learning framework.
- Streamlit for providing a seamless user interface.
- Scikit-learn for data preprocessing and evaluation metrics.
- Open-source resources and datasets for tuberculosis classification.

