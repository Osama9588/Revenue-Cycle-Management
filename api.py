# app.py - FastAPI + MongoDB Atlas AI RCM Pipeline
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
# 1. INITIALIZATION & CLOUD CONNECTION
# ==========================================
print("⚙️ Initializing Cloud-Powered RCM System...")

# Groq Client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

# EasyOCR (Set gpu=True if you have a CUDA-enabled GPU)
reader = easyocr.Reader(['en'], gpu=False)

# MongoDB Atlas Connection
ATLAS_URI = "mongodb+srv://osama:Mehtab123@cluster0.mah8vuk.mongodb.net/"
mongo_client = MongoClient(ATLAS_URI)
db = mongo_client["insurance_db"]
patients_col = db["patients"]
claims_col = db["claims"]

# --- GLOBAL DATA CACHE ---
# We cache the 55k rows in memory for instant Fuzzy Matching
cached_df = pd.DataFrame()
cached_ids = []

def refresh_patient_cache():
    global cached_df, cached_ids
    print("🔄 Downloading latest patient registry from MongoDB Atlas...")
    records = list(patients_col.find({}, {"_id": 0}))
    if records:
        cached_df = pd.DataFrame(records)
        # Ensure column name matches your 'insurance_id' requirement
        cached_ids = cached_df['insurance_id'].astype(str).tolist()
        print(f"✅ Cache Ready: {len(cached_ids)} patients loaded.")
    else:
        print("⚠️ Warning: No records found in 'insurance_db.patients'")

# Perform initial cache load
refresh_patient_cache()

# FastAPI App
app = FastAPI(title="AI Revenue Cycle Management", version="2.5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ==========================================
# 2. THE AI AGENTS
# ==========================================

def registration_agent(file_path, database_df, id_list):
    """Phase 1: OCR + LLM to verify Patient Identity"""
    print(f"🔍 [AGENT 1] Reading Registration Document...")
    results = reader.readtext(file_path, detail=0)
    raw_text = " | ".join(results)

    system_prompt = (
        "You are a Medical Data Parser. Extract: Name, Age, Gender, Blood Type, Insurance ID, and Provider. "
        "OCR often confuses '0' with 'O' and '7' with 'Z'. Provide the most likely ID. "
        "Return a JSON: {'Patient': {...}, 'Insurance': {'Provider': '...', 'ID': '...'}}"
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": raw_text}],
        response_format={"type": "json_object"}
    )
    data = json.loads(response.choices[0].message.content)

    # Fuzzy Matching for ID correction
    extracted_id = str(data.get("Insurance", {}).get("ID", "")).strip().upper()
    matches = difflib.get_close_matches(extracted_id, id_list, n=1, cutoff=0.8)

    if matches:
        data["Insurance"]["ID"] = matches[0]
        data["access_granted"] = True
        data["verification_status"] = "Verified (Fuzzy Match)"
    else:
        data["access_granted"] = False
        data["verification_status"] = "Invalid/Not Found"

    return data


def coding_agent(image_path):
    """Phase 2: OCR + LLM to generate ICD/CPT Codes"""
    print(f"🧠 [AGENT 2] Extracting Clinical Codes...")
    results = reader.readtext(image_path, detail=0)
    raw_notes = " | ".join(results)

    system_prompt = (
        "You are a Certified Medical Coder. Convert clinical notes into codes. "
        "Extract the Insurance ID if present. "
        "MAPPING: Cancer->C80.1, Diabetes->E11.9, Arthritis->M19.90, Asthma->J45.909, Obesity->E66.9, Hypertension->I10. "
        "VISITS: Urgent->99214, Emergency->99284, Elective->99213. "
        "Return JSON: {'insurance_id': '...', 'medical_condition': '...', 'icd_10': '...', 'admission_type': '...', 'cpt_code': '...'}"
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": raw_notes}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


def adjudication_agent(registration_data, coding_data, database_df):
    """Phase 3: Financial Logic (Keys updated for UI compatibility)"""
    print("⚖️ [AGENT 3] Adjudicating Claim...")

    payer_contracts = {
        "Blue Cross": 0.80, "Aetna": 0.75, "Medicare": 0.90,
        "UnitedHealthcare": 0.70, "Cigna": 0.85, "Default": 0.50
    }

    target_id = registration_data.get("Insurance", {}).get("ID")
    patient_record = database_df[database_df['insurance_id'] == target_id]

    if patient_record.empty:
        return {"status": "Rejected", "reason": "Patient ID not in Atlas database."}

    record = patient_record.iloc[0]
    db_condition = record['Medical Condition']
    total_bill = float(record['Billing Amount'])
    provider = record['Insurance Provider']

    if coding_data['medical_condition'].lower() == db_condition.lower():
        coverage_rate = payer_contracts.get(provider, 0.50)
        status = "Approved"
        reason = "Diagnosis matches database record."
    else:
        coverage_rate = 0.0 
        status = "Denied"
        reason = f"Mismatch! Note says '{coding_data['medical_condition']}' but DB says '{db_condition}'."

    # --- MATCHING YOUR UI EXPECTATIONS ---
    return {
        "status": status,
        "adjudication_reason": reason,
        "billing_details": {  # Changed from 'financials'
            "total_charge": round(total_bill, 2),
            "insurance_payment": round(total_bill * coverage_rate, 2),
            "patient_total_due": round(total_bill * (1 - coverage_rate), 2) # Changed from 'patient_balance'
        }
    }
# ==========================================
# 3. API ENDPOINTS
# ==========================================

@app.get("/")
def health_check():
    return {"status": "Active", "database": "Connected to MongoDB Atlas"}

@app.get("/patients")
def list_patients():
    """Fetch first 100 patients for dashboard preview"""
    records = list(patients_col.find({}, {"_id": 0}).limit(100))
    return {"count": len(records), "data": records}

@app.post("/process-claim")
async def process_claim(registration: UploadFile = File(...), notes: UploadFile = File(...)):
    # Create unique temp file paths
    reg_path = f"temp_reg_{registration.filename}"
    notes_path = f"temp_notes_{notes.filename}"

    with open(reg_path, "wb") as f: shutil.copyfileobj(registration.file, f)
    with open(notes_path, "wb") as f: shutil.copyfileobj(notes.file, f)

    try:
        # Step 1: Identity Check
        reg_result = registration_agent(reg_path, cached_df, cached_ids)
        if not reg_result["access_granted"]:
            return {"status": "Rejected", "reason": f"Insurance ID {reg_result['Insurance']['ID']} not found."}

        # Step 2: Clinical Coding
        code_result = coding_agent(notes_path)

        # Step 3: 🔒 CROSS-DOCUMENT VALIDATION
        # Ensure the Doctor's note belongs to the person on the Insurance card
        reg_id = reg_result["Insurance"]["ID"]
        notes_id_raw = str(code_result.get("insurance_id") or "").strip().upper()
        
        if notes_id_raw and notes_id_raw != reg_id:
            # Check for OCR typo in the note's ID using fuzzy check
            match = difflib.get_close_matches(notes_id_raw, [reg_id], n=1, cutoff=0.8)
            if not match:
                return {
                    "status": "Rejected", 
                    "reason": f"Document Mismatch! Card ID: {reg_id} vs Notes ID: {notes_id_raw}"
                }

        # Step 4: Adjudication
        final_report = adjudication_agent(reg_result, code_result, cached_df)

        # Step 5: Save Claim to Cloud History
        claims_col.insert_one({
            "patient_id": reg_id,
            # "patient_name": reg_result["Patient"]["Name"],
            "patient_info": reg_result.get("Patient"),
            "coding": code_result,
            # "diagnosis": code_result["medical_condition"],
            "report": final_report,
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            # "status": final_report["status"]
        })
        
        print("Full Claim Record saved to MongoDB Atlas.")
        return final_report

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if os.path.exists(reg_path): os.remove(reg_path)
        if os.path.exists(notes_path): os.remove(notes_path)
