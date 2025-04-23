import uvicorn
import os
from train_model import train_model

if not os.path.exists("email_classifier.pkl"):
    print("Training model from dataset...")
    train_model()

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
