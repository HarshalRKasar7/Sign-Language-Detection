# Hand Sign Detection Models

This repository contains two models for hand sign detection: one based on a Random Forest algorithm using MediaPipe for hand coordinates, and the other based on a CNN Inception v3 model. Each model is organized in its respective folder and includes the necessary scripts and notebooks for data collection, training, and deployment.

## Model 1: Random Forest with MediaPipe

### Description
This model uses the Random Forest algorithm to detect hand signs in real-time. It leverages MediaPipe to capture hand coordinates.

### Folder Structure
- `collect_img.py`: Script to collect images for the dataset and save them in the `data` folder.
- `create_dataset.py`: Script to find hand sign coordinates from the collected images and save them using pickle.
- `train_classifier.py`: Script to train the model using the coordinate files.
- `interface.py`: Script to run the model and detect hand signs live.
- `requirements.txt`: List of required Python packages for Python 3.11.

### Requirements
- Python 3.11
- Install the required packages:
  ```bash
  pip install -r requirements

### Usage

1. **Collect Images:**
   ```bash
   python collect_img.py

2. **Create Dataset:**
   ```bash
   python collect_img.py

3. **Train Model:**
   ```bash
   python collect_img.py

4. **Run Interface:**
    ```bash
    python interface.py

## Model 2: CNN Inception v3

### Description
This model uses the CNN Inception v3 architecture to detect hand signs. It is built and deployed using Streamlit.

### Folder Structure
- `Organizing-Dataset.ipynb`: Jupyter notebook to organize the dataset.
- `Build-Model.ipynb`: Jupyter notebook to train the model.
- `app.py`: Streamlit application to run the model.
- `requirements`: List of required Python packages for Python 3.11.

### Requirements
- Python 3.11
- Install the required packages:
  ```bash
  pip install -r requirements

### Usage

1. **Organize Dataset:**
    Open and run the `Organizing-Dataset.ipynb` notebook to prepare the dataset.

2. **Train Model:**
   Open and run the `Build-Model.ipynb` notebook to train the model.

3. **Run Application:**
   ```bash
   streamlit run app.py
