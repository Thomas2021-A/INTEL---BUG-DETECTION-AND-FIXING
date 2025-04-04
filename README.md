# Bug Detection and Automated Fixing

## 📌 Project Description
This project automates the process of detecting and fixing bugs in Python code using machine learning techniques. It utilizes a combination of traditional ML (TF-IDF + Random Forest) and deep learning (CodeT5 Transformer) to not only classify buggy code but also suggest context-aware fixes.

---

## 📂 Dataset
The buggy and fixed code samples were collected from the [BugsInPy Repository](https://github.com/soarsmu/BugsInPy), a rich source of real-world Python bugs and their corresponding patches. We structured this data into a CSV file using a custom Python script.

---

## 🛠 Technologies Used
- Python
- Pandas & Scikit-learn
- Hugging Face Transformers (`CodeT5`)
- TF-IDF Vectorizer
- Random Forest Classifier
- Cosine Similarity

---

## 🧪 Models Implemented

### 🔹 Model 1: TF-IDF + Random Forest
- Converts buggy code to numerical features using TF-IDF.
- Classifies the project type using `RandomForestClassifier`.
- Retrieves relevant fixes based on `cosine similarity`.

### 🔹 Model 2: CodeT5 Transformer
- Fine-tuned to understand code semantics.
- Generates context-aware, fixed code using sequence-to-sequence learning.

---

## 📊 Results
- **Model 1**: Accurate in classification and retrieval-based fixing.
- **Model 2**: Provided high-quality, semantic bug fixes for unseen code samples.

---

## 🖼️ Output Screenshots

Here’s an example of the output:

![Output Screenshot](./Screenshot%202025-04-04%20125141.png)

You can find all output screenshots in this folder:  
`./Bug Detection and Fixing/screenshots/`

---


## 👨‍💻 Authors
- Thomas A  
- Soorya K  
**Karunya Institute of Technology and Sciences**

---

## 📚 References
- [BugsInPy Repository](https://github.com/soarsmu/BugsInPy)


