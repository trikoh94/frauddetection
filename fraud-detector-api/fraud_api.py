from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI(title="LinkedIn Fraud Detector API")

# CORS 설정 (Chrome Extension 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로드
print("🔄 모델 로딩 중...")
try:
    with open('fraud_detection_hybrid_v8_fixed.pkl', 'rb') as f:
        model_dict = pickle.load(f)
    
    extractor = model_dict['domain_extractor']
    bert_embedder = model_dict['bert_embedder']
    selector = model_dict['selector']
    models_balanced = model_dict['models_balanced']
    weights = models_balanced['weights']
    
    print("✅ 모델 로드 완료!")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    extractor = None

class JobPosting(BaseModel):
    title: str = ""
    description: str = ""
    company_profile: str = ""
    salary_range: str = ""
    requirements: str = ""
    benefits: str = ""
    has_company_logo: int = 0
    telecommuting: int = 0
    industry: str = ""
    function: str = ""

@app.get("/")
async def root():
    return {
        "message": "🔍 LinkedIn Fraud Detector API",
        "version": "1.0.0",
        "status": "online" if extractor else "model not loaded"
    }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": extractor is not None
    }

@app.post("/analyze")
async def analyze_job(job: JobPosting):
    if extractor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # 입력 검증
        if not job.title and not job.description:
            raise HTTPException(status_code=400, detail="Title or description required")
        
        # DataFrame 생성
        df = pd.DataFrame([job.dict()])
        
        # 특성 추출
        X_domain = extractor.transform(df)
        X_bert = bert_embedder.transform(df)
        X_hybrid = pd.concat([X_domain, X_bert], axis=1)
        X_selected = selector.transform(X_hybrid)
        
        # 예측
        balanced_proba = (
            weights['xgb'] * models_balanced['xgb'].predict_proba(X_selected)[0, 1] +
            weights['lgbm'] * models_balanced['lgbm'].predict_proba(X_selected)[0, 1] +
            weights['cat'] * models_balanced['cat'].predict_proba(X_selected)[0, 1] +
            weights['nn'] * models_balanced['nn'].predict_proba(X_selected)[0, 1]
        )
        
        # 판정
        if balanced_proba > 0.65:
            action = 'BLOCK'
            reason = 'High fraud probability - Immediate block recommended'
        elif balanced_proba > 0.40:
            action = 'REVIEW'
            reason = 'Medium risk - Manual review needed'
        else:
            action = 'PASS'
            reason = 'Appears to be a legitimate job posting'
        
        return {
            'action': action,
            'reason': reason,
            'balanced_prob': float(balanced_proba),
            'features': {
                'keyword_count': int(X_domain['d_keyword'].values[0]),
                'urgency_score': float(X_domain['d_urgency'].values[0]),
                'money_signals': int(X_domain['d_money_raw'].values[0]),
                'completeness': float(X_domain['completeness'].values[0])
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)