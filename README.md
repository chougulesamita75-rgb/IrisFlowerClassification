# 🌸 Flower Iris Classification

A beginner-friendly machine learning project that predicts the species of an Iris flower from four measurements.

## Dataset
The included `Iris.csv` contains 150 samples and these columns:
- `SepalLengthCm`
- `SepalWidthCm`
- `PetalLengthCm`
- `PetalWidthCm`
- `Species` (target)

The `Id` column is not used for prediction.

## Machine learning algorithm
This project uses a **Random Forest Classifier** inside a scikit-learn pipeline. The data is split into 80% training data and 20% testing data. The program prints accuracy, a classification report, and a confusion matrix.

## 1. Open the project
Extract the ZIP file and open the `Flower_Iris_Classification` folder in VS Code.

## 2. Create and activate a virtual environment (recommended)

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install libraries
```bash
pip install -r requirements.txt
```

## 4. Train the model
```bash
python train_model.py
```
This creates `iris_model.pkl` and displays evaluation results in the terminal.

## 5. Run the web app
```bash
streamlit run app.py
```
A browser tab opens. Enter the four flower measurements in the sidebar and click **Predict species**.

## Project structure
```text
Flower_Iris_Classification/
├── Iris.csv
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

## Important
- Measurements must be in centimetres.
- The model predicts one of the species labels present in the dataset.
- Accuracy depends on the train/test split; do not assume a fixed result without running the training script.
