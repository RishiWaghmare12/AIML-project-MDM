# Heart Disease Prediction Project

**Team Members:**
- Aditya Kotkar (PRN: 202301040009)
- Krishna Tolani (PRN: 202301040073)
- Rishi Waghmare (PRN: 202301040014)

**Institution:** MIT Academy of Engineering, Alandi, Pune  
**Academic Year:** 2025-2026

## 1. Project Overview

This project aims to build and evaluate a series of machine learning models to predict the presence of heart disease in a patient.

The workflow involves loading and cleaning the `heart.csv` dataset, performing exploratory data analysis, and engineering a new "Patient Profile" feature using K-Means clustering.

Several base classification models (like KNN, SVM, and Logistic Regression) are trained and evaluated. Their performance is then improved by implementing advanced ensemble models (Voting, AdaBoost, and Stacking). Finally, the best-performing model, the AdaBoost Ensemble, is saved and deployed as an interactive web application using Streamlit.

## 2. Project Structure

```
AIML-project-MDM/
├── aiml_pr7.ipynb              # JupyterLab notebook with full ML pipeline
├── heart.csv                   # Dataset (303 patient records)
├── app.py                      # Streamlit web application
├── pyproject.toml              # Project configuration and dependencies
├── uv.lock                     # Locked dependency versions
├── .gitignore                  # Git ignore rules
├── LICENSE.md                  # MIT License
└── README.md                   # This file
```

### Generated Files (not in repository)

These files are created when you run the notebook:

- `heart_disease_model.pkl`: Trained AdaBoost ensemble model
- `heart_disease_scaler.pkl`: Fitted StandardScaler for feature preprocessing
- `kmeans_model.pkl`: Trained K-Means clustering model for Patient_Profile feature
- `kmeans_scaler.pkl`: Fitted StandardScaler for K-Means clustering
- `.venv/`: Virtual environment (created by `uv sync`)
- `.ipynb_checkpoints/`: JupyterLab checkpoint files

## 3. Getting Started

### Prerequisites

- Python 3.9+
- `uv` (a fast Python package installer and virtual environment manager)

If you don't have `uv` installed, you can install it using one of the following commands:

**On macOS or Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**On Windows (using PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Installation

1. Clone this repository and navigate to the project directory:
   ```bash
   git clone https://github.com/RishiWaghmare12/AIML-project-MDM.git
   cd AIML-project-MDM
   ```

2. Sync the project dependencies (this automatically creates a virtual environment and installs all dependencies):
   ```bash
   uv sync
   ```

That's it! All dependencies including scikit-learn, streamlit, jupyterlab, matplotlib, and seaborn are now installed.

## 4. Running the Project

### Step 1: Train the Model (Required First Time)

Before running the web app, you need to train the model and generate the pickle files.

Launch JupyterLab with the notebook:

```bash
uv run jupyter lab aiml_pr7.ipynb
```

This will open JupyterLab in your browser with the notebook already loaded.

**Important:** Run all cells in the notebook to:
- Perform data analysis and feature engineering
- Train all models and compare their performance
- Generate 4 pickle files:
  - `heart_disease_model.pkl` (AdaBoost model)
  - `heart_disease_scaler.pkl` (feature scaler)
  - `kmeans_model.pkl` (K-Means clustering model)
  - `kmeans_scaler.pkl` (clustering scaler)

These files are required for the Streamlit app to work.

Once you've finished running all cells, close JupyterLab by pressing `Ctrl+C` in the terminal where it's running.

### Step 2: Run the Streamlit Web App

Once the model files are generated and JupyterLab is closed, launch the interactive prediction application:

```bash
uv run streamlit run app.py
```

Your web browser will automatically open to `http://localhost:8501` with the prediction interface.

## 5. Model Performance

The AdaBoost Ensemble was the clear top performer, achieving the highest accuracy and F1-Score.

| Model                  | Accuracy | F1-Score |
|------------------------|----------|----------|
| AdaBoost Ensemble      | 0.8852   | 0.8727   |
| Stacking Ensemble      | 0.8689   | 0.8571   |
| K-Nearest Neighbors    | 0.8688   | 0.8519   |
| Voting Ensemble        | 0.8525   | 0.8364   |
| Logistic Regression    | 0.8361   | 0.8214   |
| Support Vector Machine | 0.8361   | 0.8077   |
| Decision Tree          | 0.6393   | 0.6207   |

## 6. Technologies Used

- **Python 3.9+**: Programming language
- **scikit-learn**: Machine learning models and preprocessing
- **pandas & numpy**: Data manipulation and analysis
- **matplotlib & seaborn**: Data visualization
- **Streamlit**: Interactive web application framework
- **JupyterLab**: Modern interactive notebook environment
- **uv**: Fast Python package manager

## 7. Features

### JupyterLab Notebook (`aiml_pr7.ipynb`)
- Exploratory Data Analysis (EDA) with visualizations
- K-Means clustering for feature engineering (Patient Profile)
- Training and evaluation of 7 different models
- Model comparison and performance metrics
- Saves best model (AdaBoost) and scaler as pickle files

### Streamlit Web App (`app.py`)
- Interactive user interface for predictions
- Real-time heart disease risk assessment
- Input validation and preprocessing
- Probability scores for predictions
- Clean, responsive design

## 8. Dataset

The `heart.csv` dataset contains 303 patient records with 14 attributes:

- **Age**: Patient age in years
- **Sex**: Gender (1 = male, 0 = female)
- **ChestPain**: Type of chest pain (typical, asymptomatic, nonanginal, nontypical)
- **RestBP**: Resting blood pressure (mm Hg)
- **Chol**: Serum cholesterol (mg/dl)
- **Fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **RestECG**: Resting electrocardiographic results (0, 1, 2)
- **MaxHR**: Maximum heart rate achieved
- **ExAng**: Exercise induced angina (1 = yes, 0 = no)
- **Oldpeak**: ST depression induced by exercise
- **Slope**: Slope of peak exercise ST segment (1, 2, 3)
- **Ca**: Number of major vessels colored by fluoroscopy (0-3)
- **Thal**: Thalassemia (normal, fixed, reversable)
- **Target**: Heart disease diagnosis (1 = disease, 0 = no disease)

## 9. License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
