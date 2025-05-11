from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr
from datetime import datetime
import uuid
import sqlite3

app = FastAPI()

# Database setup
conn = sqlite3.connect("complaints.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS complaints (
    complaint_id TEXT PRIMARY KEY,
    name TEXT,
    phone_number TEXT,
    email TEXT,
    complaint_details TEXT,
    created_at TEXT
)
""")
conn.commit()

# Pydantic models
class ComplaintInput(BaseModel):
    name: str
    phone_number: str
    email: EmailStr
    complaint_details: str

class ComplaintOutput(BaseModel):
    complaint_id: str
    name: str
    phone_number: str
    email: str
    complaint_details: str
    created_at: str
    
@app.post("/complaints")
def create_complaint(data: ComplaintInput):
    complaint_id = str(uuid.uuid4())[:8].upper()
    created_at = datetime.utcnow().isoformat()
    cursor.execute("""
        INSERT INTO complaints VALUES (?, ?, ?, ?, ?, ?)
    """, (complaint_id, data.name, data.phone_number, data.email, data.complaint_details, created_at))
    conn.commit()
    return {"complaint_id": complaint_id, "message": "Complaint created successfully"}


@app.get("/complaints/{complaint_id}", response_model=ComplaintOutput)
def get_complaint(complaint_id: str):
    cursor.execute("SELECT * FROM complaints WHERE complaint_id = ?", (complaint_id,))
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Complaint not found")
    return ComplaintOutput(
        complaint_id=row[0],
        name=row[1],
        phone_number=row[2],
        email=row[3],
        complaint_details=row[4],
        created_at=row[5]
    )
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("complaint_api:app", host="127.0.0.1", port=8000, reload=True)