# 🩺 Breast Cancer Detection System (ANN & XAI)

An end-to-end medical Artificial Intelligence project designed to detect and classify breast cancer tumors (Benign vs. Malignant) with high precision. This system utilizes a Deep Learning **Artificial Neural Network (ANN)** built with PyTorch, an Explainable AI (XAI) pipeline, and a RESTful API deployed via FastAPI.

## 🚀 Key Features

* **Deep Learning Model:** A 5-layer Artificial Neural Network (ANN) achieving **98%+ accuracy**.
* **Explainable AI (XAI):** Integration of **SHAP** and **LIME** to provide medical transparency, explaining *why* the model made a specific prediction based on clinical features.
* **Robust Data Processing:** Implemented `StandardScaler` for feature normalization and addressed potential vanishing gradients using optimized activation functions (ReLU) and Adam optimizer.
* **REST API Backend:** A lightning-fast API built with **FastAPI** and **Uvicorn** to serve the PyTorch model.
* **Interactive UI:** A clean, browser-based frontend for doctors to input patient data and receive real-time predictions with confidence scores.

## 📁 Project Structure

The repository is modularly designed for clean architecture:

- `ai_model/`: Contains the core Machine Learning workflow.
  - Data preprocessing and EDA.
  - Model architecture (`M5` structure) and training loop using PyTorch.
  - XAI implementation (SHAP/LIME) and Latent Space Analysis (PCA/t-SNE).
- `backend/`: Contains the deployment and server files.
  - `main.py`: The FastAPI server script.
  - `index.html`: The frontend user interface.
  - Saved model weights (`.pth`) and scaler (`.pkl`).

## 🛠️ Tech Stack

* **Machine Learning:** PyTorch, Scikit-learn, Pandas, NumPy
* **Explainable AI:** SHAP, LIME
* **Backend:** Python, FastAPI, Uvicorn
* **Frontend:** HTML5, CSS3, JavaScript

## ⚙️ How to Run the Project Locally

**1. Clone the repository:**
```bash
git clone [https://github.com/abdallah-sobaih/Meme-Kanseri-Teshisi-ANN.git](https://github.com/abdallah-sobaih/Meme-Kanseri-Teshisi-ANN.git)
cd Meme-Kanseri-Teshisi-ANN
```
2. Create a virtual environment and activate it:
```
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate
```
3. Install dependencies:
``` 
   cd backend
pip install -r requirements.txt
 ```
4. Start the FastAPI Server:
``` 
   python -m uvicorn main:app --reload
  ```
The application will be available at http://127.0.0.1:8000
👨‍💻 Author
Abdallah Sobaih Software Developer & Computer Engineer
