import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

# Load dataset
df = pd.read_csv("balanced_bug_dataset.csv")

# Fill any missing values
df = df.dropna()

# Preprocessing: Normalize and Tokenize (example normalization)
df['Buggy_Code'] = df['Buggy_Code'].str.strip()
df['Fixed_Code'] = df['Fixed_Code'].str.strip()

# Split data into training and testing for classification
X_train, X_test, y_train, y_test = train_test_split(
    df['Buggy_Code'], df['Project'], test_size=0.2, random_state=42)

# Convert code into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_tfidf, y_train)

# Evaluate model performance
y_pred = clf.predict(X_test_tfidf)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Load CodeT5 model for fix suggestion
tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')

# Function to suggest a fix using CodeT5
def suggest_fix(buggy_code):
    # Set a maximum length for the input sequences
    max_input_length = 512
    max_output_length = 128

    # Tokenize the input buggy code
    inputs = tokenizer(
        f"fix: {buggy_code}",
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_input_length
    )

    # Generate a fix using the model
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_output_length
        )

    # Decode the generated fix
    suggested_fix = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return suggested_fix

# Optional: Fine-tune CodeT5 on your dataset
def fine_tune_codet5(df):
    from datasets import Dataset

    # Tokenize the dataset
    def preprocess_data(examples):
        inputs = [f"fix: {bug}" for bug in examples['Buggy_Code']]
        targets = [fix for fix in examples['Fixed_Code']]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Convert DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(preprocess_data, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./fine-tuned-codet5",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        save_steps=500,
        logging_dir="./logs",
        logging_steps=10,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("./fine-tuned-codet5")
    tokenizer.save_pretrained("./fine-tuned-codet5")

# Uncomment the following line to fine-tune the model
# fine_tune_codet5(df)

# Get user input
user_buggy_code = input("Enter your buggy code: ")
print("\nBuggy Code:\n", user_buggy_code)
print("Suggested Fix:\n", suggest_fix(user_buggy_code))