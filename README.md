# AI-101 Semester Project: Image Classification Using CNN

By Marko Ivan Marojević

This project will be focused on classifying images of animals using deep learning techniques, specifically Convolutional Neural Networks (CNNs). The model is trained to classify images from several categories, such as dogs, cats, birds, and more, using the TensorFlow framework.

The project is intended to finish until the end of October. 

## Project Overview

### Objective:

The main objective of this project is to build an image classification model that can accurately classify images into their respective animal categories. We aim to achieve a high accuracy score using a CNN architecture with minimal overfitting and efficient processing time.

### Key Features:
- **Custom Convolutional Neural Network** built using TensorFlow and Keras
- **Image Data Augmentation** to improve model generalization
- **Confusion Matrix** and **Accuracy Graphs** for performance visualization
- **Real-time prediction API** for live image classification

### Technologies Used:
- **Python 3.8+**
- **TensorFlow** (for deep learning and CNN model building)
- **Keras** (high-level neural network API)
- **Numpy, Pandas** (for data manipulation)
- **Matplotlib, Seaborn** (for plotting graphs)
- **Flask** (for building a simple web interface for model predictions)

## Getting Started

### Prerequisites:
Before you begin, ensure you have the following installed on your local machine:
- Python 3.8 or later
- TensorFlow 2.x
- Numpy, Pandas, Matplotlib
- Flask (for running the web app)

### Installation:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ai-101-semester-project.git
   cd ai-101-semester-project
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset:
- The dataset used for this project is a collection of animal images from the [Kaggle Animals Dataset](https://www.kaggle.com/datasets).
- Ensure the dataset is downloaded and placed in the `data/` directory, or provide the path in the configuration file located at `config.yaml`.

## Running the Project

1. Preprocess the data:
   ```bash
   python src/preprocess.py
   ```
   This script handles image resizing, normalization, and data augmentation.

2. Train the model:
   ```bash
   python src/train.py
   ```
   The model will begin training, and progress will be logged in the console.

3. Evaluate the model:
   ```bash
   python src/evaluate.py
   ```
   This script will output the model's accuracy, confusion matrix, and other metrics.

4. Run the prediction API:
   ```bash
   python src/app.py
   ```
   A local Flask web app will be started where you can upload an image to see the predicted animal category.

## Project Structure

```bash
ai-101-semester-project/
│
├── data/                 # Folder for storing datasets
├── models/               # Saved models and weights
├── notebooks/            # Jupyter notebooks for experimentation
├── src/                  # Source code for data processing, model training, etc.
│   ├── preprocess.py     # Data preprocessing script
│   ├── train.py          # Model training script
│   ├── evaluate.py       # Model evaluation script
│   └── app.py            # Script for the Flask web app
├── config.yaml           # Configuration file for data paths and hyperparameters
├── requirements.txt      # List of required dependencies
└── README.md             # Project documentation (this file)
```

## Results
- **Model Accuracy**: will be shared. 
- **Loss**: will be shared.

## Future Improvements
- **Increase the Dataset**: Adding more categories and training on a larger dataset could improve the model's robustness.
- **Fine-tuning Hyperparameters**: Adjusting the learning rate and model architecture might improve accuracy further.
- **Deploy the API**: Deploy the Flask app to a cloud service to allow real-time classification through a web interface.

## Contact

For any questions or concerns, please contact me at [ivan.marojevic@prvabankacg.com]

---

Let me know if you'd like any more adjustments!
