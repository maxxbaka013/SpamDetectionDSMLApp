# 📩 Spam Detection — Naive Bayes

A lightweight **Streamlit** web app for classifying SMS/email messages as **Spam** or **Ham** using a **Multinomial Naive Bayes** model with `CountVectorizer`.
[Live Demo → Spam Detector](https://spamdetectiondsmlapp-k7vgjeapyzkhw4cxkzwqkx.streamlit.app/)

---

## 📂 Repository Structure

```
├── app.py               # Streamlit app source code
├── code.ipynb           # Jupyter notebook (EDA, model experiments)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 🚀 Features

* Train a Naive Bayes spam classifier directly in the app
* Upload a custom dataset (CSV with `Category` & `Message` columns)
* View metrics: Accuracy, Precision, Recall, F1-score
* Visualize Confusion Matrix
* Save & download trained pipeline as `.pkl`
* Load pre-trained model for instant predictions
* Classify single messages with probability scores

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/spam-detection-streamlit.git
cd spam-detection-streamlit
```

### 2️⃣ Create virtual environment & install dependencies

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> If `requirements.txt` is missing, create it with:

```bash
pip freeze > requirements.txt
```

### 3️⃣ Run locally

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📊 Usage

1. **Load Saved Model (.pkl)** or **Train New Model** from the sidebar
2. If training:

   * Upload CSV with columns:

     * `Category` → labels (`spam`, `ham`, etc.)
     * `Message` → text messages
   * Adjust `test_size` and `random_state`
   * Click **Train** to build & evaluate the model
3. Try your own text in the **Try It** section

---

## 📑 Dataset Format

Example CSV:

```csv
Category,Message
ham,Hey there! Are you free for lunch?
spam,WIN a $1000 prize by clicking here!
```

---

## 💾 Model Export

* After training, save the pipeline as `.pkl`
* Load later for fast predictions without retraining

---

## 📌 Requirements

* Python 3.9+
* [Streamlit](https://streamlit.io)
* pandas, numpy, scikit-learn, matplotlib

---

## 🧪 Development Notes

* Notebook `code.ipynb` contains exploratory analysis & testing
* `app.py` mirrors notebook pipeline (CountVectorizer → MultinomialNB)
* Uses `train_test_split` for validation
* Displays weighted metrics & confusion matrix

---

## 🌐 Deployment

Deployed via **Streamlit Community Cloud**
[Live App](https://spamdetectiondsmlapp-k7vgjeapyzkhw4cxkzwqkx.streamlit.app/)

---

## 📜 License

MIT License — feel free to use, improve, and share.

---
