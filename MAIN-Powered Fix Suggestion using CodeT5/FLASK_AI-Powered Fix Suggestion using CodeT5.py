from flask import Flask, request, jsonify, render_template
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from transformers import RobertaTokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Load dataset
df = pd.read_csv("balanced_bug_dataset.csv").dropna()

# Train a classifier (Project Classification)
X = df['Buggy_Code']
y = df['Project']

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_tfidf, y)

# Load CodeT5 model
tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')

def suggest_fix(buggy_code):
    max_input_length = 512
    max_output_length = 128

    # Preserve newlines explicitly
    buggy_code = buggy_code.replace("\n", " <NEWLINE> ")

    inputs = tokenizer(
        f"fix: {buggy_code}",
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_input_length
    )

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_output_length
        )

    # Decode and restore newlines
    suggested_fix = tokenizer.decode(outputs[0], skip_special_tokens=True)
    suggested_fix = suggested_fix.replace("<NEWLINE>", "\n")

    return suggested_fix

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    buggy_code = data.get("buggy_code", "").strip()  # Preserve multiline input

    if not buggy_code:
        return jsonify({"error": "No code provided"}), 400

    # Predict the project type
    project_pred = clf.predict(vectorizer.transform([buggy_code]))[0]

    # Suggest a fix
    fix = suggest_fix(buggy_code)

    return jsonify({
        "project_type": project_pred,
        "suggested_fix": fix
    })

if __name__ == '__main__':
    app.run(debug=True)
