# app.py - FastAPI + MongoDB RCM Pipeline Server

import easyocr
import json
import pandas as pd
import difflib
import shutil
import os
from groq import Groq
from pymongo import MongoClient
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ==========================================
# 1. INITIALIZATION
# ==========================================
print("⚙️ Initializing System...")

# Groq Client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # Replace with your key

# EasyOCR
reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have GPU

# MongoDB Connection
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["insurance_db"]
patients_col = db["patients"]
claims_col = db["claims"]

# Load patients from MongoDB into DataFrame
def load_patients():
    records = list(patients_col.find({}, {"_id": 0}))
    df = pd.DataFrame(records)
    all_ids = df['insurance_id'].astype(str).tolist()
    return df, all_ids

# FastAPI App
app = FastAPI(title="RCM Insurance Pipeline", version="1.0")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["http://localhost:3000"],
    allow_origins=["*"],  # allow all origins for local file testing
    allow_methods=["*"],
    allow_headers=["*"]
)

# ==========================================
# 2. AGENT DEFINITIONS
# ==========================================

def registration_agent(file_path, database_df, id_list):
    """Phase 1: Verify Patient Identity and Insurance ID"""
    print(f"\n🔍 [AGENT 1] Processing Registration: {file_path}")
    results = reader.readtext(file_path, detail=0)
    raw_text = " | ".join(results)

    system_prompt = (
        "You are a Medical Data Parser. Extract: Name, Age, Gender, Blood Type, Insurance ID, and Provider. "
        "OCR often confuses '0' with 'O' and '7' with 'Z'. Provide the most likely ID. "
        "Return a JSON: {'Patient': {...}, 'Insurance': {'Provider': '...', 'ID': '...'}}"
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_text}
        ],
        response_format={"type": "json_object"}
    )
    data = json.loads(response.choices[0].message.content)

    # Fuzzy Matching
    extracted_id = str(data.get("Insurance", {}).get("ID", "")).strip().upper()
    matches = difflib.get_close_matches(extracted_id, id_list, n=1, cutoff=0.8)

    if matches:
        data["Insurance"]["ID"] = matches[0]
        data["access_granted"] = True
        data["verification_status"] = "Verified (Fuzzy Match)"
        print(f"✅ Patient Verified: {matches[0]}")
    else:
        data["access_granted"] = False
        data["verification_status"] = "Invalid/Not Found"
        print(f"❌ Access Denied for ID: {extracted_id}")

    return data


def coding_agent(image_path):
    """Phase 2: Convert Doctor's Notes to ICD-10/CPT Codes"""
    print(f"\n🧠 [AGENT 2] Processing Medical Coding: {image_path}")
    results = reader.readtext(image_path, detail=0)
    raw_notes = " | ".join(results)

    system_prompt = (
        "You are a Certified Medical Coder. Convert clinical notes into codes. "
        "ALSO extract the Insurance ID if present in the notes. "
        "MAPPING: Cancer->C80.1, Diabetes->E11.9, Arthritis->M19.90, Asthma->J45.909, Obesity->E66.9, Hypertension->I10. "
        "VISITS: Urgent->99214, Emergency->99284, Elective->99213. "
        "Return JSON: {'insurance_id': '...or null', 'medical_condition': '...', 'icd_10': '...', 'admission_type': '...', 'cpt_code': '...'}"
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_notes}
        ],
        response_format={"type": "json_object"}
    )
    coding_data = json.loads(response.choices[0].message.content)
    print(f"✅ Coding Complete: {coding_data['icd_10']} / {coding_data['cpt_code']}")
    return coding_data


def adjudication_agent(registration_data, coding_data, database_df):
    """Phase 3: Final Claim Decision"""
    print("\n⚖️ [AGENT 3] Starting Final Adjudication...")

    payer_contracts = {
        "Blue Cross": 0.80, "Aetna": 0.75, "Medicare": 0.90,
        "UnitedHealthcare": 0.70, "Cigna": 0.85, "Default": 0.50
    }

    target_id = registration_data.get("Insurance", {}).get("ID")
    patient_record = database_df[database_df['insurance_id'] == target_id]

    if patient_record.empty:
        return {"status": "Rejected", "reason": "Patient ID not found in database."}

    record = patient_record.iloc[0]
    db_condition = record['Medical Condition']
    total_bill = float(record['Billing Amount'])
    provider = record['Insurance Provider']

    if coding_data['medical_condition'].lower() == db_condition.lower():
        coverage_rate = payer_contracts.get(provider, payer_contracts["Default"])
        status = "Approved"
        reason = "Diagnosis matches authorized record. Insurance coverage applied."
    else:
        coverage_rate = 0.0
        status = "Denied"
        reason = (
            f"Mismatch! Doctor treated '{coding_data['medical_condition']}', "
            f"but record expected '{db_condition}'. Claim denied."
        )

    insurance_pays = total_bill * coverage_rate
    patient_owes = total_bill - insurance_pays

    return {
        "status": status,
        "denial_reason": reason if status == "Denied" else "N/A",
        "billing_details": {
            "total_charge": round(total_bill, 2),
            "insurance_coverage_percent": f"{int(coverage_rate * 100)}%",
            "insurance_payment": round(insurance_pays, 2),
            "patient_total_due": round(patient_owes, 2)
        }
    }

# ==========================================
# 3. API ROUTES
# ==========================================

@app.get("/")
def root():
    return {"message": "RCM Pipeline API is running ✅"}


@app.get("/patients")
def get_all_patients():
    """Fetch all patients from MongoDB"""
    records = list(patients_col.find({}, {"_id": 0}))
    return {"total": len(records), "patients": records}


@app.get("/claims")
def get_all_claims():
    """Fetch all past claims from MongoDB"""
    records = list(claims_col.find({}, {"_id": 0}))
    return {"total": len(records), "claims": records}


@app.post("/process-claim")
async def process_claim(
    registration: UploadFile = File(...),
    notes: UploadFile = File(...)
):
    reg_path   = f"temp_{registration.filename}"
    notes_path = f"temp_{notes.filename}"

    with open(reg_path, "wb") as f:
        shutil.copyfileobj(registration.file, f)
    with open(notes_path, "wb") as f:
        shutil.copyfileobj(notes.file, f)

    try:
        df, all_ids = load_patients()

        # Agent 1: Registration
        reg_result = registration_agent(reg_path, df, all_ids)

        if not reg_result["access_granted"]:
            return {
                "status": "Rejected",
                "reason": "Patient verification failed. Insurance ID not found."
            }

        # Agent 2: Medical Coding
        code_result = coding_agent(notes_path)

        # =============================================
        # 🔒 CROSS-DOCUMENT VALIDATION
        # Check if notes image belongs to same patient
        # =============================================
        reg_id   = reg_result.get("Insurance", {}).get("ID", "").strip().upper()
        notes_id = str(code_result.get("insurance_id") or "").strip().upper()

        # Fuzzy match notes ID against database too
        notes_id_matched = difflib.get_close_matches(notes_id, all_ids, n=1, cutoff=0.8)
        notes_id_resolved = notes_id_matched[0] if notes_id_matched else notes_id

        if notes_id_resolved and notes_id_resolved != reg_id:
            print(f"🚨 Document Mismatch! Registration ID: {reg_id} | Notes ID: {notes_id_resolved}")
            return {
                "status": "Rejected",
                "reason": f"Document mismatch detected! Registration belongs to '{reg_id}' but doctor's notes belong to '{notes_id_resolved}'. Claim rejected immediately.",
                "billing_details": None
            }

        print(f"✅ Document Cross-Check Passed: Both documents belong to {reg_id}")
        # =============================================

        # Agent 3: Adjudication
        final_report = adjudication_agent(reg_result, code_result, df)

        # Save to MongoDB
        claim_record = {
            "patient_id":   reg_id,
            "patient_info": reg_result.get("Patient", {}),
            "coding":       code_result,
            "report":       final_report,
            "cross_check":  "Passed"
        }
        claims_col.insert_one(claim_record)
        print("✅ Claim saved to MongoDB")

        return final_report

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(reg_path):   os.remove(reg_path)
        if os.path.exists(notes_path): os.remove(notes_path)