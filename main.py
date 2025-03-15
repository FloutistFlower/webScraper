import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from pymongo import MongoClient
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from bson import ObjectId
import os
from fastapi.responses import HTMLResponse
import uvicorn

import logging
import asyncio


# Enable logging for FastAPI app
logging.basicConfig(level=logging.DEBUG)

#uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# Prompt the user for a number

'''
url = input("Enter the target url: ")
number_of_keywords = input("Enter the number of keywords: ")
keywords = []
erase_db = input("Would you like to clear the database before searching? (Y/N): ")
#full_search = input("Would you like a more thorough search? (Y/N)")
# Convert the input to an integer
try:
    number = int(number_of_keywords)
    for i in range(number):
        keyword = input("Enter keyword " + str(i+1) + ": ")
        keywords.append(keyword)
except ValueError:
    print("Invalid input! Please enter a valid integer.")
'''

url = "https://www.a2gov.org/"  #target URL
keywords = ["budget", "finance", "acfr"]


app = FastAPI()

# MongoDB Atlas Connection String (Replace with your actual connection string)
MONGO_URI = "mongodb+srv://ashlynnGrace:SkyBlue132@cluster0.uvycp.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"


# Connect to MongoDB Atlas
client = MongoClient(MONGO_URI)
db = client["Cluster0"] 
collection = db["testCollection"] 

class URLInput(BaseModel):
    url: str

async def process_data(data: dict):
    await asyncio.sleep(2)  # Simulate async work
    return {"message": "Processed successfully", "input": data}

@app.post("/run")
async def run_script(data: dict):
    return await process_data(data)

class UserInput(BaseModel):
    user_input: str
    url: str  # Ensure 'url' is explicitly required

@app.post("/process")
async def process_data(data: UserInput):
    try:
        # Simulate processing (replace this with your logic)
        if not data.user_input or not data.url:
            raise HTTPException(status_code=400, detail="Both user_input and url are required")
        
        processed_result = f"You entered: {data.user_input}, URL: {data.url}"
        return {"message": "Success", "result": processed_result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/hello/{name}")
def read_item(name: str):
    return {"message": f"Hello, {name}!"}


# Endpoint to insert a document
@app.post("/insert/")
async def insert_data(data: dict):
    result = collection.insert_one(data)
    return {"inserted_id": str(result.inserted_id)}


#app = FastAPI()

class DataItem(BaseModel):
    name: str
    value: int

@app.post("/add_data")
def add_data(item: DataItem):
    return {"message": "Data added successfully", "item": item}

# Endpoint to fetch all documents
@app.get("/get_data")
def get_data():
    # Retrieve all documents from MongoDB
    data = list(collection.find({}, {"_id": 0}))  # Exclude MongoDB ObjectId
    return {"data": data}


@app.delete("/delete_all/")
def delete_all_data():
    result = collection.delete_many({})  # Deletes all documents
    return {"message": f"Deleted {result.deleted_count} documents"}

for doc in collection.find():
    print(doc)



def extract_links(url):
    response = requests.get(url)
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


response = requests.get(url)
html_content = response.text
soup = BeautifulSoup(html_content, "html.parser")
links = [a['href'] for a in soup.find_all('a', href=True)]


links_data = extract_links(url)
print("Extracted links data:", links_data)

'''
for url, anchor, title, file_type, rel in links_data:
    if (file_type == "html"):
        print("testing")
        extra_links_data = extract_links(url)
for url, anchor, title, filetype, rel in extra_links_data:
    links_data.append((url, anchor, title, file_type, rel))
'''

# Sample training data (URLs + metadata)
x_train = [
    # Highly relevant financial reports and documents
    "Budget",
    "Finance",
    "acfr",
    "Budget Report",
    "Finance Director",
    "CFO",
    "treasury",
    "administrative",
    "administration",
    "director",
    "monetary",
    "board",
    "CEO",
    "CTO",
    "Trustees",
    "Leadership",
    "https://example.com/reports/annual-budget-2024.pdf",
    "https://example.com/finance/fiscal-year-2023-summary",
    "https://citygov.org/documents/acfr-2023.pdf",
    "https://citygov.org/finance-reports/tax-summary-2024",
    "https://govdata.com/spending/2023-budget",
    "https://example.com/gov/audit-reports-2024",
    "https://example.com/finance/debt-policy",
    "https://citygov.org/treasury/financial-outlook-2025",
    "https://city.gov/documents/financial-strategy-2023",
    "https://stategov.com/revenue-expenditure-2024",
    "finance-and-administrative-services",

    # Irrelevant general government links
    "Volunteering",
    "Deferment",
    "Property Tax",
    "Translate",
    "Service",
    "https://city.gov/parks-and-recreation",
    "https://govportal.com/contact-us",
    "https://example.com/mayor-office",
    "https://govdata.com/housing-development",
    "https://example.com/legal-department",
    "https://city.gov/public-safety/police",
    "LinkedIn",
    "Instagram",
    "facebook",
    "Twitter",
    "Contact Us",
    "Report an Issue",
    "Nextdoor",
    "Youtube",
    "site",
    "visitors",
    "government"
    "Notifications"
    "Compost",
    "services",
    "judicial",
    "court",
    "3661e8349526",
    "parking",
    "apply",
    "volunteer",
    "request",
    "report",
    "repair",
    "broken",
    "Engage Ann Arbor",
    "Trash",
    "Holiday",
    "Demographics",
    "Sports"
]

y_train = [
    # 1 = Relevant, 0 = Not Relevant
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, #relevant
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 #irrelevant

]


# Convert text into numerical features
vectorizer = TfidfVectorizer()
x_train_tfidf = vectorizer.fit_transform(x_train)

# Train the classifier
model = LogisticRegression()
model.fit(x_train_tfidf, y_train)

# Save the model for future use
with open("hybrid_link_classifier.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)

print("Model trained and saved!")


def classify_link_hybrid(url, anchor, title, rel):
    # Load the trained model
    with open("hybrid_link_classifier.pkl", "rb") as f:
        vectorizer, model = pickle.load(f)

    # Combine URL and metadata for classification
    input_text = f"{url} {anchor} {title} {rel}"
    input_tfidf = vectorizer.transform([input_text])

    # Predict relevance
    prediction = model.predict(input_tfidf)
    return "Relevant" if prediction[0] == 1 else "Not Relevant"

# Use extract_links() to get URL + metadata
links_data = extract_links(url)

for url, anchor, title, file_type, rel in links_data:
    print(url, classify_link_hybrid(url, anchor,title, rel))

def rank_relevant_links(links):
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
    ranked_links.sort(key=lambda x: x[4], reverse=True)

    return ranked_links


ranked_links = rank_relevant_links(links_data)

# Print ranked links
print("\n🔹 Ranked Relevant Links:\n")
for rank, (url, anchor, title, rel, file_type, score) in enumerate(ranked_links, start=1):
    print(f"{rank}. {url} | Score: {score}")
    print(f"   Anchor: {anchor}")
    print(f"   Title: {title}")
    print(f"   type: {file_type}")
    print(f"   Rel: {rel}\n")
    data = {
        "url": url,
        "type": file_type,
        "relevance_score": score,
        "keywords": keywords
    }
    collection.insert_one(data)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


