import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from pymongo import MongoClient
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import httpx
import logging
import asyncio
import uvicorn
from motor.motor_asyncio import AsyncIOMotorClient


# Enable logging for FastAPI app
logging.basicConfig(level=logging.DEBUG)

# MongoDB Atlas Connection String (Replace with your actual connection string)
MONGO_URI = "mongodb+srv://ashlynnGrace:SkyBlue132@cluster0.uvycp.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = AsyncIOMotorClient(MONGO_URI)
db = client["Cluster0"]
collection = db["testCollection"]

app = FastAPI()


class UserInput(BaseModel):
    url: str
    number_of_keywords: int
    keywords: list[str]  # A list of strings for the keywords


async def extract_links(url):
    """Asynchronously extract links from the URL using httpx and BeautifulSoup"""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        links = []
        for a in soup.find_all('a', href=True):
            full_url = urljoin(url, a['href'])
            if full_url.startswith("http"):
                anchor_text = a.get_text(strip=True)
                title_attr = a.get("title", "")
                file_extension = full_url.split('.')[-1] if '.' in full_url.split('/')[-1] else "HTML"
                rel_attr = a.get("rel", "")
                links.append((full_url, anchor_text, title_attr, file_extension.lower(), rel_attr))
        return links


async def classify_link_hybrid(url, anchor, title, rel):
    """Classify a link based on its metadata"""
    # Load the trained model (ensure it's loaded only once and not during every call)
    with open("hybrid_link_classifier.pkl", "rb") as f:
        vectorizer, model = pickle.load(f)

    # Combine URL and metadata for classification
    input_text = f"{url} {anchor} {title} {rel}"
    input_tfidf = vectorizer.transform([input_text])

    # Predict relevance
    prediction = model.predict(input_tfidf)
    return "Relevant" if prediction[0] == 1 else "Not Relevant"


async def run_script(url, keywords):
    """Run the script to extract links, classify them, and rank them"""
    links_data = await extract_links(url)

    # Rank the relevant links based on keywords
    ranked_links = rank_relevant_links(links_data, keywords)

    # Store the ranked links in MongoDB
    for url, anchor, title, rel, file_type, score in ranked_links:
        collection.insert_one({
            "url": url,
            "type": file_type,
            "relevance_score": score,
            "keywords": keywords
        })

    return ranked_links


def rank_relevant_links(links, keywords):
    """Ranks relevant links based on keyword priority scoring."""
    ranked_links = []

    for url, anchor, title, file_type, rel in links:
        if classify_link_hybrid(url, anchor, title, rel) == "Relevant":  # Check if relevant
            # Combine metadata for ranking
            combined_text = f"{url} {anchor} {title} {rel}".lower()

            # Compute keyword score with explicit priority weights
            score = sum((len(keywords) - i) * combined_text.count(keyword) for i, keyword in enumerate(keywords))

            # Debugging: print keyword matches
            print(f"🔹 {url} | Score: {score} | Matches: {[keyword for keyword in keywords if keyword in combined_text]}")

            ranked_links.append((url, anchor, title, rel, file_type, score))

    # Sort by score (higher score = higher ranking)
    ranked_links.sort(key=lambda x: x[5], reverse=True)

    return ranked_links


@app.post("/process")
async def process_data(data: UserInput):
    try:
        if not data.url or not data.keywords:
            raise HTTPException(status_code=400, detail="Both URL and keywords are required")
        
        # Call the async function to run the script
        ranked_links = await run_script(data.url, data.keywords)

        return {"message": "Processing successful", "ranked_links": ranked_links}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

class DataItem(BaseModel):
    name: str
    value: int

@app.get("/get_data")
async def get_data():
    try:
        # Fetch all documents in the collection, excluding the MongoDB ObjectId
        data = await collection.find({}, {"_id": 0}).to_list(None)
        
        # If no data found, return a message indicating that
        if not data:
            return {"message": "No data found in the collection"}
        
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve data: {str(e)}")

# Route to delete all documents from the collection
@app.delete("/delete_all/")
async def delete_all_data():
    try:
        result = await collection.delete_many({})  # Deletes all documents in the collection
        return {"message": f"Deleted {result.deleted_count} documents"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete documents: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
