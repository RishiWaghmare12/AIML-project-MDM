# Customer Churn Prediction Project

**Team Members:**
- Aditya Kotkar (PRN: 202301040009)
- Krishna Tolani (PRN: 202301040073)
- Rishi Waghmare (PRN: 202301040014)

**Institution:** MIT Academy of Engineering, Alandi, Pune  
**Academic Year:** 2025-2026

## 1. Project Overview

This project aims to build and evaluate a series of machine learning models to predict customer churn in a telecommunications company, following the methodology from the research paper **"Prediction of Customer Churn Using Machine Learning"** by Yash Singh et al. (2022), published in the International Research Journal of Modernization in Engineering Technology and Science (IRJMETS).

**Paper Reference:** [https://www.academia.edu/download/105452759/fin_irjmets1649599690.pdf](https://www.academia.edu/download/105452759/fin_irjmets1649599690.pdf)

The workflow involves loading and cleaning the Telco Customer Churn dataset, performing exploratory data analysis, and engineering a new "Customer Segment" feature using K-Means clustering.

Several base classification models (including Naive Bayes, KNN, SVM, Decision Tree, and Logistic Regression) are trained and evaluated. Their performance is then improved by implementing advanced ensemble models (Voting, AdaBoost, and Stacking). Finally, the best-performing model (Stacking Ensemble) is saved and deployed as an interactive web application using Streamlit.

## 2. Project Structure

```
AIML-project-MDM/
├── aiml_pr7.ipynb                           # JupyterLab notebook with full ML pipeline
├── WA_Fn-UseC_-Telco-Customer-Churn.csv     # Dataset (7043 customer records)
├── app.py                                   # Streamlit web application
├── pyproject.toml                           # Project configuration and dependencies
├── uv.lock                                  # Locked dependency versions
├── .gitignore                               # Git ignore rules
├── LICENSE.md                               # MIT License
└── README.md                                # This file
```

### Generated Files (not in repository)

These files are created when you run the notebook:

- `churn_model.pkl`: Trained Stacking Ensemble model (best performing)
- `churn_scaler.pkl`: Fitted StandardScaler for feature preprocessing
- `kmeans_model.pkl`: Trained K-Means clustering model for Customer_Segment feature
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
- Train all models (including Naive Bayes) and compare their performance
- Generate 4 pickle files:
  - `churn_model.pkl` (Stacking Ensemble model - best performer)
  - `churn_scaler.pkl` (feature scaler)
  - `kmeans_model.pkl` (K-Means clustering model)
  - `kmeans_scaler.pkl` (clustering scaler)

These files are required for the Streamlit app to work.

Once you've finished running all cells, close JupyterLab by pressing `Ctrl+C` in the terminal where it's running.

### Step 2: Run the Streamlit Web App

Once the model files are generated and JupyterLab is closed, launch the interactive churn prediction application:

```bash
uv run streamlit run app.py
```

Your web browser will automatically open to `http://localhost:8501` with the customer churn prediction interface.

## 5. Model Performance

The **Stacking Ensemble** achieved the best overall performance, demonstrating the power of combining multiple models. This aligns with the findings from the research paper by Yash Singh et al. (2022), which emphasizes the effectiveness of ensemble methods for churn prediction.

| Model                  | Accuracy | F1-Score |
|------------------------|----------|----------|
| Stacking Ensemble      | 0.8142   | 0.6245   |
| Logistic Regression    | 0.8053   | 0.6108   |
| AdaBoost Ensemble      | 0.7982   | 0.5896   |
| Support Vector Machine | 0.7889   | 0.5561   |
| Voting Ensemble        | 0.7868   | 0.5522   |
| Naive Bayes            | 0.7654   | 0.5512   |
| K-Nearest Neighbors    | 0.7598   | 0.5469   |
| Decision Tree          | 0.7221   | 0.5082   |

**Key Findings:**
- Stacking Ensemble outperforms all individual models and other ensemble methods
- The combination of diverse base learners (Logistic Regression, KNN, Decision Tree, SVM) with a meta-learner provides superior predictive power
- Naive Bayes provides competitive performance with fast training time

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
- K-Means clustering for feature engineering (Customer Segment)
- Training and evaluation of 8 different models
- Implementation of ensemble methods (Voting, AdaBoost, Stacking)
- Model comparison and performance metrics
- Saves best model (Stacking Ensemble) and scaler as pickle files

### Streamlit Web App (`app.py`)
- Interactive user interface for churn predictions
- Real-time customer churn risk assessment
- Input validation and preprocessing
- Probability scores for churn predictions
- Clean, responsive design

## 8. Dataset

The Telco Customer Churn dataset contains 7,043 customer records with 21 attributes:

**Customer Demographics:**
- **customerID**: Unique customer identifier
- **gender**: Customer gender (Male/Female)
- **SeniorCitizen**: Whether customer is a senior citizen (1 = yes, 0 = no)
- **Partner**: Whether customer has a partner (Yes/No)
- **Dependents**: Whether customer has dependents (Yes/No)

**Services:**
- **tenure**: Number of months the customer has stayed with the company
- **PhoneService**: Whether customer has phone service (Yes/No)
- **MultipleLines**: Whether customer has multiple lines (Yes/No/No phone service)
- **InternetService**: Type of internet service (DSL/Fiber optic/No)
- **OnlineSecurity**: Whether customer has online security (Yes/No/No internet service)
- **OnlineBackup**: Whether customer has online backup (Yes/No/No internet service)
- **DeviceProtection**: Whether customer has device protection (Yes/No/No internet service)
- **TechSupport**: Whether customer has tech support (Yes/No/No internet service)
- **StreamingTV**: Whether customer has streaming TV (Yes/No/No internet service)
- **StreamingMovies**: Whether customer has streaming movies (Yes/No/No internet service)

**Account Information:**
- **Contract**: Contract term (Month-to-month/One year/Two year)
- **PaperlessBilling**: Whether customer has paperless billing (Yes/No)
- **PaymentMethod**: Payment method (Electronic check/Mailed check/Bank transfer/Credit card)
- **MonthlyCharges**: Monthly charge amount
- **TotalCharges**: Total amount charged to the customer

**Target:**
- **Churn**: Whether the customer churned (Yes/No)

## 9. Research Paper Reference

This project follows the methodology outlined in:

**"Prediction of Customer Churn Using Machine Learning"**  
Authors: Yash Singh, Yash Pandit, Neil Joshi, Prof. Vedika Avhad  
Published in: International Research Journal of Modernization in Engineering Technology and Science (IRJMETS)  
Volume: 04, Issue: 04, April 2022  
e-ISSN: 2582-5208  
Impact Factor: 6.752

**Paper Link:** [https://www.academia.edu/download/105452759/fin_irjmets1649599690.pdf](https://www.academia.edu/download/105452759/fin_irjmets1649599690.pdf)

The paper emphasizes the importance of:
- Using multiple machine learning algorithms for comparison
- Implementing ensemble methods for improved accuracy
- Feature engineering through clustering techniques
- Comprehensive evaluation metrics (Accuracy, F1-Score, Confusion Matrix)

Our implementation extends the paper's approach by:
- Adding K-Means clustering for customer segmentation
- Implementing Stacking Ensemble as the meta-learner
- Creating an interactive web application for real-time predictions
- Achieving superior performance through advanced ensemble techniques

## 10. License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
