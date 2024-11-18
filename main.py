import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import nltk

# Download required NLTK data
nltk.download('stopwords')

# Load dataset (replace with the path to your dataset)
data = pd.read_excel('camera_movements.xlsx')

# Preprocess text
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

data['Processed_Command'] = data['Command'].apply(preprocess_text)

# Vectorize commands
commands = data['Processed_Command']
responses = data['Response']
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(commands)

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, responses, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Generate predictions for the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Prediction function
def predict_command(command):
    command_preprocessed = preprocess_text(command)
    command_transformed = vectorizer.transform([command_preprocessed])
    prediction = model.predict(command_transformed)
    return prediction[0]

# Main loop for user interaction
print("\nCamera Movement Text Classifier")
print("Enter a command for the camera (type 'exit' to quit):")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Exiting. Goodbye!")
        break
    try:
        response = predict_command(user_input)
        print("Response:", response)
    except Exception as e:
        print("Error processing command:", e)
