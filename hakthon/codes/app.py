"""
FastAPI Web Application for Tomato Disease Detection
Provides REST API and serves web interface with Groq-powered plant chatbot
"""



from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import shutil
import uuid
from typing import Optional
import uvicorn
import json
import base64
from groq import Groq, AsyncGroq
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key="AIzaSyDCVRmyxiocqPbWco5v-bZYw48afCKXl4k")

import hakthon.config
from src.inference import create_predictor

# ---------------- Groq client ------------------------------------------------
GROQ_API_KEY = "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"
groq_client = Groq(api_key=GROQ_API_KEY)
async_groq_client = AsyncGroq(api_key=GROQ_API_KEY)

PLANT_SYSTEM_PROMPT = """أنت بوت النباتات (PlantBot)، مساعد ذكاء اصطناعي متخصص في رعاية النباتات والأمراض والزراعة.
أنت تجيب فقط على الأسئلة المتعلقة بـ:
- النباتات (الطماطم، الخضروات، الفواكه، المحاصيل، الأعشاب، الأشجار، الزهور)
- أمراض النباتات والآفات
- الممارسات الزراعية
- صحة التربة والتسميد
- الري والسقاية
- الحصاد والتخزين
- الزراعة المستدامة
- تكنولوجيا الزراعة الذكية

إذا سأل المستخدم عن أي شيء غير متعلق بالنباتات أو الزراعة، ارفض بلطف ووجهه لطرح أسئلة متعلقة بالنباتات.
كن ودوداً ومفيداً وعملياً. قدم نصائح قابلة للتطبيق عند الإمكان.
الرد الافتراضي يكون بالعربية دائماً.
إذا كتب المستخدم بالإنجليزية، رد بالإنجليزية.
إذا كتب المستخدم بالعربية أو بأي لغة أخرى، رد بالعربية.
اجعل الإجابات موجزة ولكن شاملة (2-4 فقرات كحد أقصى)."""

# ---------------- App setup ---------------------------------------------------
app = FastAPI(
    title="Tomato Disease Detection API",
    description="AI-powered tomato disease detection with Groq plant chatbot",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("/home/smart-farming/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

predictor = None


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global predictor
    try:
        print("Loading model...")
        predictor = create_predictor()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("Please train the model first using: python train.py")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main web interface"""
    html_file = Path("/home/smart-farming/templates/index.html")
    if html_file.exists():
        return FileResponse(html_file)
    return HTMLResponse("<h1>Tomato Disease Detection System</h1><p>API: /docs</p>")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": predictor is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...), lang: str = "ar"):
    """Predict disease from uploaded image"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")

    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}")

    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}{file_ext}"

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # --------------------------------------------------------------------
        # Gemini Vision Check: Is it a tomato?
        # --------------------------------------------------------------------
        try:
            with open(file_path, "rb") as image_file:
                image_bytes = image_file.read()
                
            model = genai.GenerativeModel('gemini-1.5-flash')
            vision_res = model.generate_content([
                "Is this a picture of a tomato plant, tomato leaf, or tomato fruit? Reply ONLY with 'yes' or 'no'.",
                {"mime_type": f"image/{file_ext[1:]}" if file_ext[1:] in ['jpeg', 'png', 'webp'] else "image/jpeg", "data": image_bytes}
            ])
            
            answer = vision_res.text.strip().lower()
            if "no" in answer and "yes" not in answer:
                msg = "عذراً، لم يتم التعرف على نبات طماطم في هذه الصورة. يرجى التأكد من رفع صورة صحيحة." if lang == "ar" else "Sorry, no tomato plant was detected in this image."
                return JSONResponse(content={
                    'success': True,
                    'is_tomato': False,
                    'message': msg
                })
        except Exception as vision_e:
            print(f"Warning: Vision API check failed: {vision_e}")
            # If vision check fails, we proceed gracefully to the classic detection

        # Proceed with classic disease prediction
        result = predictor.predict(
            image_path=str(file_path),
            lang=lang
        )

        if result.get('is_tomato'):
            disease = result.get('disease_name', 'Unknown')
            try:
                import json
                with open("disease_arabic.json", "r", encoding="utf-8") as f:
                    arabic_diseases = json.load(f)
                    
                disease_lower = disease.lower().replace(" ", "_")
                found_treatment = None
                
                # First try direct key match
                for key, info in arabic_diseases.items():
                    if disease_lower in key or key in disease_lower:
                        found_treatment = info
                        break
                        
                # If not found, try matching by name
                if not found_treatment:
                    for key, info in arabic_diseases.items():
                        if disease.lower() in info.get("disease_name", "").lower():
                            found_treatment = info
                            break
                            
                if found_treatment:
                    ctrl = found_treatment.get('prevention_and_control', {})
                    result['treatment'] = {
                        'symptoms': found_treatment.get('symptoms', []),
                        'prevention_and_control': {
                            'chemical_control': ctrl.get('chemical', ''),
                            'physical_control': ctrl.get('physical', ''),
                            'biological_control': ctrl.get('biological', ''),
                            'agricultural_practices': ctrl.get('agricultural', '')
                        }
                    }
                    result['disease_name'] = found_treatment.get('disease_name', disease)
                else:
                    result['treatment'] = {
                        'symptoms': ["لا تتوفر معلومات كافية في قاعدة البيانات عن هذا المرض."],
                        'prevention_and_control': {
                            'chemical': 'لا توجد معلومات متاحة.',
                            'physical': 'لا توجد معلومات متاحة.',
                            'biological': 'لا توجد معلومات متاحة.',
                            'agricultural': 'لا توجد معلومات متاحة.'
                        }
                    }
            except Exception as e:
                result['treatment'] = {
                    'symptoms': [f"خطأ في تحميل بيانات المرض: {str(e)}"],
                    'prevention_and_control': {}
                }

        result['uploaded_image'] = str(file_path)
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ---------------- Groq Chatbot endpoint ---------------------------------------
class ChatMessage(BaseModel):
    message: str
    history: Optional[list] = []


@app.post("/chat")
async def chat(body: ChatMessage):
    """Plant-specialized chatbot powered by Groq"""
    try:
        messages = [{"role": "system", "content": PLANT_SYSTEM_PROMPT}]

        # Add conversation history (last 10 turns)
        for turn in body.history[-10:]:
            if turn.get("role") in ("user", "assistant"):
                messages.append({"role": turn["role"], "content": turn["content"]})

        messages.append({"role": "user", "content": body.message})

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=600,
            temperature=0.7
        )

        reply = response.choices[0].message.content
        return JSONResponse(content={"reply": reply, "success": True})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...), lang: str = "ar"):
    """Transcribe audio to text using Groq Whisper model"""
    file_ext = Path(file.filename).suffix.lower()
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}{file_ext}"

    try:
        # Save the audio file temporarily
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Call Groq Whisper API
        with open(file_path, "rb") as bf:
            transcription = await async_groq_client.audio.transcriptions.create(
                file=(f"audio{file_ext}", bf.read()),
                model="whisper-large-v3",
                prompt="tomato diseases plants farming" + (" بالعربي" if lang == "ar" else ""),
            )

        text = transcription.text
        return JSONResponse(content={"text": text, "success": True})

    except Exception as e:
        print(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")
    finally:
        # Clean up the temporary file
        if file_path.exists():
            file_path.unlink()

# ---------------- Crop Recommendation -----------------------------------------
class CropPredictionRequest(BaseModel):
    lat: Optional[float] = None
    lon: Optional[float] = None
    auger_id: Optional[int] = None
    date: Optional[str] = None
    depth: Optional[str] = None
    max_distance_km: Optional[float] = 15.0

@app.post("/api/predict_crop")
async def api_predict_crop(req: CropPredictionRequest):
    try:
        from src.crop_recommender import recommend_crop_lstm, get_nearest_points
        from datetime import datetime, date

        when = datetime.strptime(req.date, "%Y-%m-%d").date() if req.date else date.today()
        
        # --- DEMO MOCK BEGIN ---
        if req.lat is not None and req.lon is not None:
            # Marsa Matrouh is roughly lon < 28.5
            if req.lon < 28.5:
                return JSONResponse(content={
                    "status": "ok",
                    "resolved_from": "gps",
                    "input_location": {"lat": req.lat, "lon": req.lon},
                    "nearest_points": [{"distance_km": 1.5}],
                    "auger_id": 14,
                    "distance_km": 1.5,
                    "anchor_day": 1,
                    "anchor_month": 5,
                    "date_used": when.isoformat(),
                    "lstm": {
                        "anchor_date_used": "2026-05-01",
                        "recommended_crops": ["الزيتون 🌿", "التين 🍈", "الشعير 🌾"],
                        "scores": {
                            "الزيتون 🌿": 0.61,
                            "التين 🍈": 0.25,
                            "الشعير 🌾": 0.14
                        }
                    }
                })
          
          
        # --- DEMO MOCK END ---
        
        resolved_from = None
        user_lat = req.lat
        user_lon = req.lon
        nearest_points = None
        auger_id = req.auger_id

        if auger_id is not None:
            resolved_from = "auger_id"
        elif user_lat is not None and user_lon is not None:
            resolved_from = "gps"
            nearest_points = get_nearest_points(user_lat, user_lon, k=1)
            if not nearest_points:
                raise HTTPException(status_code=500, detail="No mapped points available.")
            if nearest_points[0]["distance_km"] > req.max_distance_km:
                return JSONResponse(content={
                    "status": "out_of_region",
                    "distance_km": round(nearest_points[0]["distance_km"], 2),
                    "message": "You are outside the mapped region."
                })
            auger_id = int(nearest_points[0]["id"])
        else:
            raise HTTPException(status_code=400, detail="Either auger_id or lat/lon must be provided")

        rec = recommend_crop_lstm(auger_id, when, depth=req.depth)
        if rec is None:
            raise HTTPException(status_code=400, detail=f"No matching rows in db for auger_id={auger_id}")

        # Extract distance from nearest_points if available
        distance_km = 0.0
        if nearest_points and len(nearest_points) > 0:
            distance_km = nearest_points[0].get("distance_km", 0.0)
        
        # Parse anchor date context
        from datetime import datetime
        anchor_dt = datetime.strptime(rec["anchor_date_used"], "%Y-%m-%d")

        return JSONResponse(content={
            "status": "ok",
            "resolved_from": resolved_from,
            "input_location": None if user_lat is None else {"lat": user_lat, "lon": user_lon},
            "nearest_points": nearest_points,
            "auger_id": auger_id,
            "distance_km": round(distance_km, 2),
            "anchor_day": anchor_dt.day,
            "anchor_month": anchor_dt.month,
            "date_used": when.isoformat(),
            "lstm": rec,
        })
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files
static_dir = Path("/home/smart-farming/static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


if __name__ == "__main__":
    print("=" * 80)
    print("TOMATO DISEASE DETECTION WEB SERVER v2.0")
    print("=" * 80)
    print(f"\nStarting server on http://{config.HOST}:{config.PORT}")
    print(f"API documentation: http://{config.HOST}:{config.PORT}/docs")
    print(f"Plant Chatbot: Powered by Groq (llama-3.3-70b-versatile)")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 80)

    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level="info"
    )
