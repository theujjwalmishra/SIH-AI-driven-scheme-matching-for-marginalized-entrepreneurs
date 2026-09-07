from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import numpy as np

app = FastAPI(
    title="Marginalized Entrepreneur Scheme Matcher",
    description="SIH Prototype: Rule-based eligibility + Semantic AI retrieval",
    version="1.0.0"
)

# Load a lightweight, fast semantic model
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------------------------------------
# Scheme Dataset (Can be expanded or backed by MongoDB/Postgres)
# -------------------------------------------------------------
SCHEMES_DB = [
    {
        "id": "SCH001",
        "name": "Stand-Up India Scheme",
        "ministry": "Ministry of Finance",
        "target_categories": ["SC", "ST", "Women"],
        "min_age": 18,
        "max_turnover": 50000000,
        "description": "Facilitates bank loans between 10 lakh and 1 crore to at least one SC/ST borrower and one woman borrower per bank branch for greenfield enterprises in manufacturing, services, or trading.",
        "benefits": "Loan up to ₹1 Crore with composite subsidy",
    },
    {
        "id": "SCH002",
        "name": "PM Mudra Yojana (Shishu/Kishore)",
        "ministry": "Ministry of Finance",
        "target_categories": ["General", "OBC", "SC", "ST", "Women", "PwD"],
        "min_age": 18,
        "max_turnover": 1000000,
        "description": "Provides collateral-free micro loans up to 10 lakhs to small artisans, street vendors, small retail shops, and village entrepreneurs.",
        "benefits": "Loans up to ₹50,000 (Shishu) and up to ₹5 Lakh (Kishore)",
    },
    {
        "id": "SCH003",
        "name": "PMEGP (Prime Minister Employment Generation Programme)",
        "ministry": "Ministry of MSME",
        "target_categories": ["General", "SC", "ST", "OBC", "Women", "PwD"],
        "min_age": 18,
        "max_turnover": 2500000,
        "description": "Credit-linked subsidy programme to generate self-employment ventures in rural and urban areas for micro enterprises and traditional artisans.",
        "benefits": "15% to 35% margin money subsidy on project cost",
    },
    {
        "id": "SCH004",
        "name": "PM Vishwakarma Scheme",
        "ministry": "Ministry of MSME",
        "target_categories": ["General", "SC", "ST", "OBC"],
        "min_age": 18,
        "max_turnover": 500000,
        "description": "Financial aid, skill upgrades, toolkits, and low-interest credit for traditional craftsmen like potters, blacksmiths, carpenters, and weavers.",
        "benefits": "₹15,000 toolkit incentive + collateral-free loans at 5% interest",
    }
]

# Precompute embeddings for all scheme descriptions at startup
SCHEME_VECTORS = model.encode([s["description"] for s in SCHEMES_DB], normalize_embeddings=True)

# -------------------------------------------------------------
# Data Models
# -------------------------------------------------------------
class EntrepreneurProfile(BaseModel):
    name: str
    age: int
    gender: str = Field(..., example="Female")
    category: str = Field(..., example="SC")  # SC, ST, OBC, General, PwD
    annual_turnover: float = Field(..., example=120000.0)
    business_description: str = Field(..., example="Handmade bamboo baskets and rural clay pottery shop")

class SchemeMatchResponse(BaseModel):
    scheme_id: str
    scheme_name: str
    ministry: str
    semantic_score: float
    benefits: str
    eligibility_status: str

# -------------------------------------------------------------
# Matching Logic
# -------------------------------------------------------------
def check_hard_eligibility(profile: EntrepreneurProfile, scheme: dict) -> bool:
    """Deterministic validation on demographic and income guardrails."""
    if profile.age < scheme["min_age"]:
        return False
    if profile.annual_turnover > scheme["max_turnover"]:
        return False
    
    # Check category intersection
    valid_categories = [c.upper() for c in scheme["target_categories"]]
    if profile.category.upper() not in valid_categories and profile.gender.upper() not in valid_categories:
        return False
        
    return True

@app.post("/api/match-schemes", response_model=List[SchemeMatchResponse])
def match_schemes(profile: EntrepreneurProfile):
    # 1. Embed entrepreneur's business description
    user_vec = model.encode([profile.business_description], normalize_embeddings=True)[0]
    
    # 2. Calculate cosine similarities (dot product with normalized vectors)
    scores = np.dot(SCHEME_VECTORS, user_vec)
    
    matched_results = []
    for idx, scheme in enumerate(SCHEMES_DB):
        is_eligible = check_hard_eligibility(profile, scheme)
        
        if is_eligible:
            matched_results.append({
                "scheme_id": scheme["id"],
                "scheme_name": scheme["name"],
                "ministry": scheme["ministry"],
                "semantic_score": round(float(scores[idx]), 3),
                "benefits": scheme["benefits"],
                "eligibility_status": "Eligible"
            })
            
    # Sort results by best semantic contextual fit
    matched_results.sort(key=lambda x: x["semantic_score"], reverse=True)
    return matched_results

@app.get("/health")
def health_check():
    return {"status": "active", "schemes_loaded": len(SCHEMES_DB)}
