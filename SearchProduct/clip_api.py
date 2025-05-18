from fastapi import FastAPI, Query, UploadFile, File
import open_clip
import torch
from PIL import Image
import requests
import io
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline,  AutoModelForSequenceClassification, AutoTokenizer
app = FastAPI()

# Cho phép frontend truy cập (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load mô hình CLIP
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

@app.get("/")
def root():
    return {"message": "CLIP API is running"}

@app.post("/image-embedding-from-url/")
def get_image_embedding(image_url: str = Query(...)):
    try:
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        image_input = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            embedding = image_features[0].tolist()

        return {"embedding": embedding} 
    except Exception as e:
        return {"error": str(e)}

@app.post("/image-embedding-from-file/")
async def get_image_embedding_from_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_input = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            embedding = image_features[0].tolist()

        return {"embedding": embedding}
    except Exception as e:
        return {"error": str(e)}
sentiment_model_name = "wonrax/phobert-base-vietnamese-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

    
sentiment_classifier = pipeline(
    "text-classification",
    model=sentiment_model,
    tokenizer=sentiment_tokenizer,
    device=0 if torch.cuda.is_available() else -1
)
@app.post("/analyze-sentiment")
async def analyze_sentiment(text: str = Query(..., min_length=1)):
 
    try:
        result = sentiment_classifier(text)[0]
        return {
            "text": text,
            "sentiment": result["label"],
            "confidence": round(result["score"], 4),
            "model": sentiment_model_name
        }
    except Exception as e:
        return {"error": str(e)}