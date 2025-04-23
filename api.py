from fastapi import FastAPI
from pydantic import BaseModel
from models import predict_category
from utils import mask_text

app = FastAPI()

class EmailRequest(BaseModel):
    input_email_body: str

@app.post("/")
async def classify_email(request: EmailRequest):
    email = request.input_email_body
    masked_email, entities = mask_text(email)
    category = predict_category(masked_email)

    return {
        "input_email_body": email,
        "list_of_masked_entities": entities,
        "masked_email": masked_email,
        "category_of_the_email": category
    }
