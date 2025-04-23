import joblib

def predict_category(text):
    model = joblib.load("email_classifier.pkl")
    return model.predict([text])[0]
