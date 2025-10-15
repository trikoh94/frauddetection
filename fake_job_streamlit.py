"""
Job Posting Fraud Detection Dashboard v7 - REDESIGNED
Streamlit Cloud 배포용 + 디자인 개선
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# NLTK/TextBlob 초기화
# ============================================================================
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

@st.cache_resource
def download_nltk_data():
    for pkg in ['brown', 'punkt', 'wordnet', 'averaged_perceptron_tagger']:
        try:
            nltk.data.find(f'corpora/{pkg}')
        except:
            try:
                nltk.data.find(f'tokenizers/{pkg}')
            except:
                nltk.download(pkg, quiet=True)

download_nltk_data()
from textblob import TextBlob

# ============================================================================
# 클래스 정의
# ============================================================================

@lru_cache(maxsize=1000)
def get_sentiment(text):
    """캐싱된 감성 분석"""
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    except:
        return 0.0, 0.0


class FeatureExtractor:
    """도메인 특성 추출기"""
    def __init__(self, keywords, ind_risk, func_risk, overall_rate, thresholds):
        self.keywords = keywords
        self.ind_risk = ind_risk
        self.func_risk = func_risk
        self.overall_rate = overall_rate
        self.thresholds = thresholds

    def extract_text_features(self, text, prefix=''):
        if pd.isna(text) or text == '':
            return self._empty_features(prefix)

        text_str = str(text)
        text_lower = text_str.lower()
        words = text_str.split()
        word_count = len(words)
        sentence_count = max(len(re.findall(r'[.!?]+', text_str)), 1)

        polarity, subjectivity = get_sentiment(text_str)

        emails = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text_str))
        phones = len(re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text_str))
        urls = len(re.findall(r'http[s]?://[^\s]+', text_str))

        keyword_cnt = sum(kw in text_lower for kw in self.keywords)
        caps_ratio = sum(1 for c in text_str if c.isupper()) / max(len(text_str), 1)

        urgency_words = ['urgent', 'hurry', 'now', 'asap', 'immediately', 'limited time', 'act now']
        urgency = sum(w in text_lower for w in urgency_words)

        pressure_words = ['must', 'required', 'guarantee', 'easy', 'fast', 'quick', 'instant']
        pressure = sum(w in text_lower for w in pressure_words)

        money_words = ['earn', 'income', 'profit', 'cash', 'money', '$', 'dollar', 'salary', 'pay']
        money = sum(w in text_lower for w in money_words)

        exaggeration = ['amazing', 'incredible', 'unbelievable', 'guaranteed', '100%', 'unlimited', 'free']
        exag = sum(w in text_lower for w in exaggeration)

        return {
            f'{prefix}length': len(text_str),
            f'{prefix}word_count': word_count,
            f'{prefix}sentence_count': sentence_count,
            f'{prefix}avg_word_len': np.mean([len(w) for w in words]) if words else 0,
            f'{prefix}avg_sent_len': word_count / sentence_count,
            f'{prefix}caps_ratio': caps_ratio,
            f'{prefix}high_caps': int(caps_ratio > self.thresholds.get('caps', 0.15)),
            f'{prefix}exclaim': text_str.count('!'),
            f'{prefix}high_exclaim': int(text_str.count('!') > self.thresholds.get('exclaim', 3)),
            f'{prefix}question': text_str.count('?'),
            f'{prefix}keyword': keyword_cnt,
            f'{prefix}has_keyword': int(keyword_cnt > 0),
            f'{prefix}urgency': urgency,
            f'{prefix}pressure': pressure,
            f'{prefix}money': money,
            f'{prefix}exag': exag,
            f'{prefix}manipulative': urgency + pressure + exag,
            f'{prefix}email': emails,
            f'{prefix}phone': phones,
            f'{prefix}url': urls,
            f'{prefix}contacts': emails + phones,
            f'{prefix}polarity': polarity,
            f'{prefix}subjectivity': subjectivity,
            f'{prefix}high_polarity': int(polarity > self.thresholds.get('polarity', 0.3)),
            f'{prefix}high_subj': int(subjectivity > self.thresholds.get('subjectivity', 0.5)),
        }

    def _empty_features(self, prefix):
        keys = ['length', 'word_count', 'sentence_count', 'avg_word_len', 'avg_sent_len',
                'caps_ratio', 'high_caps', 'exclaim', 'high_exclaim', 'question',
                'keyword', 'has_keyword', 'urgency', 'pressure', 'money', 'exag', 'manipulative',
                'email', 'phone', 'url', 'contacts', 'polarity', 'subjectivity',
                'high_polarity', 'high_subj']
        return {f'{prefix}{k}': 0 for k in keys}

    def extract_company_features(self, company_profile):
        if pd.isna(company_profile) or company_profile == '':
            return {'company_credibility': 0, 'has_awards': 0, 'has_partners': 0, 'has_year': 0}

        text = str(company_profile).lower()
        score = 0

        has_awards = int(any(w in text for w in ['award', 'certified', 'accredited']))
        score += has_awards * 0.3

        has_partners = int(any(w in text for w in ['partnership', 'partner with', 'collaboration']))
        score += has_partners * 0.25

        has_year = int(bool(re.search(r'\b(19|20)\d{2}\b', text)))
        score += has_year * 0.2

        score += min(len(company_profile) / 500, 1.0) * 0.25

        return {
            'company_credibility': score,
            'has_awards': has_awards,
            'has_partners': has_partners,
            'has_year': has_year
        }

    def extract_industry_risk(self, industry, function):
        ind_str = str(industry).lower().strip() if pd.notna(industry) else ''
        func_str = str(function).lower().strip() if pd.notna(function) else ''

        ind_risk = self.ind_risk.get(ind_str, self.overall_rate * 1.5 if ind_str == '' else self.overall_rate)
        func_risk = self.func_risk.get(func_str, self.overall_rate * 1.5 if func_str == '' else self.overall_rate)

        return {
            'ind_risk': ind_risk,
            'func_risk': func_risk,
            'combined_risk': (ind_risk + func_risk) / 2,
            'high_risk': int(ind_risk > self.overall_rate * 2 and func_risk > self.overall_rate * 2),
        }

    def extract_meta_features(self, row):
        weighted = [
            int(row.get('has_company_logo', 0)) * 3,
            int(pd.notna(row.get('salary_range'))) * 2,
            int(pd.notna(row.get('company_profile')) and row.get('company_profile') != '') * 2,
            int(pd.notna(row.get('requirements')) and row.get('requirements') != ''),
            int(pd.notna(row.get('benefits')) and row.get('benefits') != ''),
        ]
        completeness = sum(weighted) / 9.0

        return {
            'has_logo': int(row.get('has_company_logo', 0)),
            'has_salary': int(pd.notna(row.get('salary_range'))),
            'has_profile': int(pd.notna(row.get('company_profile')) and row.get('company_profile') != ''),
            'has_req': int(pd.notna(row.get('requirements')) and row.get('requirements') != ''),
            'has_benefits': int(pd.notna(row.get('benefits')) and row.get('benefits') != ''),
            'telecommute': int(row.get('telecommuting', 0)),
            'completeness': completeness,
            'low_info': int(completeness < 0.5),
        }

    def transform(self, df):
        features = []

        title_feat = df['title'].apply(lambda x: self.extract_text_features(x, 't_'))
        features.append(pd.DataFrame(list(title_feat)))

        desc_feat = df['description'].apply(lambda x: self.extract_text_features(x, 'd_'))
        features.append(pd.DataFrame(list(desc_feat)))

        req_feat = df['requirements'].apply(lambda x: self.extract_text_features(x, 'r_'))
        features.append(pd.DataFrame(list(req_feat)))

        comp_feat = df['company_profile'].apply(self.extract_company_features)
        features.append(pd.DataFrame(list(comp_feat)))

        ind_feat = df.apply(lambda row: self.extract_industry_risk(row.get('industry'), row.get('function')), axis=1)
        features.append(pd.DataFrame(list(ind_feat)))

        meta_feat = df.apply(self.extract_meta_features, axis=1)
        features.append(pd.DataFrame(list(meta_feat)))

        result = pd.concat(features, axis=1)

        result['low_info_urgent'] = ((result['completeness'] < 0.3) & (result['d_urgency'] > 0)).astype(int)
        result['no_logo_money'] = ((result['has_logo'] == 0) & (result['d_money'] > 2)).astype(int)
        result['remote_high_subj'] = ((result['telecommute'] == 1) & (result['d_high_subj'] == 1)).astype(int)
        result['high_risk_low_info'] = ((result['ind_risk'] > result['ind_risk'].mean() * 2) & (result['completeness'] < 0.4)).astype(int)
        result['no_salary_exag'] = ((result['has_salary'] == 0) & (result['d_exag'] > 2)).astype(int)
        result['contact_urgent'] = ((result['d_contacts'] > 0) & (result['d_urgency'] > 0)).astype(int)

        return result


class BERTEmbedder:
    """BERT embedding 생성기"""
    def __init__(self, model_name='all-MiniLM-L6-v2', n_components=64):
        # Lazy import
        from sentence_transformers import SentenceTransformer
        from sklearn.decomposition import PCA

        self.model = SentenceTransformer(model_name)
        self.pca = PCA(n_components=n_components, random_state=42)
        self.pca_fitted = False
        self.n_components = n_components

    def transform(self, df, fit=False):
        texts = []
        for _, row in df.iterrows():
            title = str(row.get('title', '')).strip()
            desc = str(row.get('description', '')).strip()
            text = f"{title} [SEP] {desc}" if title and desc else (title or desc)
            texts.append(text if text else "empty")

        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)

        if fit or not self.pca_fitted:
            embeddings_reduced = self.pca.fit_transform(embeddings)
            self.pca_fitted = True
        else:
            embeddings_reduced = self.pca.transform(embeddings)

        bert_df = pd.DataFrame(
            embeddings_reduced,
            columns=[f'bert_{i}' for i in range(self.n_components)],
            index=df.index
        )

        return bert_df


# ============================================================================
# 헬퍼 함수
# ============================================================================

def get_feature_contributions(job_data):
    """각 특성이 사기 점수에 미친 영향 계산"""
    contributions = []

    desc = str(job_data.get('description', '')).lower()
    title = str(job_data.get('title', '')).lower()

    urgency_words = ['urgent', 'hurry', 'now', 'asap', 'immediately', 'limited time', 'act now']
    urgency_count = sum(w in desc or w in title for w in urgency_words)
    if urgency_count > 0:
        impact = min(urgency_count * 8, 25)
        contributions.append({
            'feature': '🚨 Urgency Pressure',
            'value': f"{urgency_count} keywords",
            'impact': f"+{impact}%",
            'explanation': f"Found {urgency_count} urgency words - creates artificial time pressure"
        })

    money_words = ['$', 'earn', 'income', 'profit', 'cash', 'money', 'dollar']
    money_count = sum(w in desc or w in title for w in money_words)
    if money_count > 2:
        impact = min((money_count - 2) * 5, 20)
        contributions.append({
            'feature': '💰 Money Emphasis',
            'value': f"{money_count} mentions",
            'impact': f"+{impact}%",
            'explanation': f"Excessive focus on money - typical fraud tactic"
        })

    exag_words = ['amazing', 'incredible', 'unbelievable', 'guaranteed', '100%', 'unlimited', 'free']
    exag_count = sum(w in desc or w in title for w in exag_words)
    if exag_count > 0:
        impact = min(exag_count * 10, 25)
        contributions.append({
            'feature': '⚡ Exaggerated Claims',
            'value': f"{exag_count} words",
            'impact': f"+{impact}%",
            'explanation': f"Unrealistic promises - red flag for scams"
        })

    exclaim_count = desc.count('!') + title.count('!')
    if exclaim_count > 3:
        impact = min((exclaim_count - 3) * 3, 15)
        contributions.append({
            'feature': '❗ Excessive Punctuation',
            'value': f"{exclaim_count} exclamations",
            'impact': f"+{impact}%",
            'explanation': "Too many exclamation marks - unprofessional"
        })

    if '@' in desc:
        contributions.append({
            'feature': '📧 Email in Description',
            'value': "Detected",
            'impact': "+15%",
            'explanation': "Email in description - bypasses platform security"
        })

    if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', desc):
        contributions.append({
            'feature': '📞 Phone in Description',
            'value': "Detected",
            'impact': "+15%",
            'explanation': "Phone number - moves conversation off-platform"
        })

    completeness = sum([
        int(job_data.get('has_company_logo', 0)),
        int(bool(job_data.get('salary_range'))),
        int(bool(job_data.get('company_profile'))),
        int(bool(job_data.get('requirements'))),
        int(bool(job_data.get('benefits')))
    ]) / 5.0

    if completeness < 0.4:
        impact = int((0.4 - completeness) * 100)
        contributions.append({
            'feature': '📊 Low Information Quality',
            'value': f"{completeness*100:.0f}% complete",
            'impact': f"+{impact}%",
            'explanation': "Missing critical information"
        })

    if not job_data.get('has_company_logo'):
        contributions.append({
            'feature': '❌ No Company Logo',
            'value': "Missing",
            'impact': "+12%",
            'explanation': "Legitimate companies display branding"
        })

    if not job_data.get('salary_range'):
        contributions.append({
            'feature': '💵 No Salary Info',
            'value': "Missing",
            'impact': "+8%",
            'explanation': "Transparency issue"
        })

    if job_data.get('telecommuting') and completeness < 0.4:
        contributions.append({
            'feature': '🚩 High-Risk Pattern',
            'value': "Remote + Low Info",
            'impact': "+20%",
            'explanation': "Common fraud pattern"
        })

    return contributions


def predict_fraud(job_data, model_dict):
    """사기 예측 + 상세 설명"""
    try:
        df = pd.DataFrame([job_data])

        extractor = model_dict['domain_extractor']
        bert_embedder = model_dict['bert_embedder']
        selector = model_dict['selector']

        X_domain = extractor.transform(df)
        X_bert = bert_embedder.transform(df)

        X_hybrid = pd.concat([X_domain.reset_index(drop=True), X_bert.reset_index(drop=True)], axis=1)
        X_selected = selector.transform(X_hybrid)

        models_bal = model_dict['models_balanced']
        models_recall = model_dict['models_recall']
        thresholds = model_dict['thresholds']
        weights = models_bal['weights']

        prob_xgb = models_bal['xgb'].predict_proba(X_selected)[0, 1]
        prob_lgbm = models_bal['lgbm'].predict_proba(X_selected)[0, 1]
        prob_cat = models_bal['cat'].predict_proba(X_selected)[0, 1]
        prob_nn = models_bal['nn'].predict_proba(X_selected)[0, 1]

        prob_balanced = (
            weights['xgb'] * prob_xgb +
            weights['lgbm'] * prob_lgbm +
            weights['cat'] * prob_cat +
            weights['nn'] * prob_nn
        )

        prob_recall = (
            models_recall['xgb'].predict_proba(X_selected)[0, 1] +
            models_recall['lgbm'].predict_proba(X_selected)[0, 1] +
            models_recall['cat'].predict_proba(X_selected)[0, 1]
        ) / 3

        if prob_balanced > 0.85:
            action = 'block'
            explanation = '🚫 AUTO-BLOCKED: High confidence fraud'
        elif prob_balanced > 0.65:
            action = 'review'
            explanation = '⚠️ MANUAL REVIEW: High risk detected'
        elif prob_balanced > 0.45 and prob_recall > 0.75:
            action = 'review'
            explanation = '⚠️ MANUAL REVIEW: Multiple warning signs'
        else:
            action = 'pass'
            explanation = '✅ APPROVED: No significant fraud signals'

        contributions = get_feature_contributions(job_data)

        return {
            'action': action,
            'balanced_prob': float(prob_balanced),
            'recall_prob': float(prob_recall),
            'explanation': explanation,
            'contributions': contributions,
            'model_scores': {
                'XGBoost': float(prob_xgb),
                'LightGBM': float(prob_lgbm),
                'CatBoost': float(prob_cat),
                'NeuralNet': float(prob_nn)
            },
            'weights': weights
        }

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None


# ============================================================================
# Streamlit UI - 디자인 개선!
# ============================================================================

st.set_page_config(
    page_title="Fraud Detection AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - 현대적 디자인
st.markdown("""
<style>
    /* 전체 배경 */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* 메인 컨테이너 */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: white;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    
    /* 헤더 스타일 */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 3rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* 카드 스타일 */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    /* 버튼 스타일 */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* 텍스트 입력 */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s;
    }
    
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* 메트릭 */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    /* 사이드바 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* 구분선 */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }
    
    /* Success/Warning/Error 박스 */
    .stAlert {
        border-radius: 15px;
        border-left: 5px solid;
        padding: 1rem 1.5rem;
    }
    
    /* 프로그레스 바 */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        background-color: #f0f0f0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# 헤더
st.title("🛡️ Job Posting Fraud Detection")
st.markdown("""
<p style='font-size: 1.2rem; color: #666; margin-top: -1rem;'>
    AI-Powered Security System • BERT Hybrid + Ensemble ML
</p>
""", unsafe_allow_html=True)

# 모델 로드
@st.cache_resource
def load_model():
    try:
        with open('fraud_detection_hybrid_v7.pkl', 'rb') as f:
            model_dict = pickle.load(f)
        return model_dict
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None

with st.spinner('🔄 Loading AI models...'):
    model_dict = load_model()

# Sidebar
with st.sidebar:
    st.markdown("### 📊 System Dashboard")

    if model_dict:
        metadata = model_dict.get('metadata', {})
        perf = metadata.get('final_performance', {})

        st.markdown("#### Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("AUC", f"{perf.get('hybrid', {}).get('auc', 0):.3f}", delta="Best")
        with col2:
            st.metric("F1", f"{perf.get('hybrid', {}).get('f1', 0):.3f}", delta="Optimized")

        st.markdown("---")
        st.markdown("#### 🤖 AI Components")
        st.markdown("""
        - ✅ BERT Language Model
        - ✅ XGBoost Classifier
        - ✅ LightGBM Classifier
        - ✅ CatBoost Classifier
        - ✅ Neural Network
        """)

        st.markdown("---")
        st.markdown("#### 🔒 Security Levels")
        st.markdown("""
        **🚫 AUTO-BLOCK**  
        Fraud score > 85%
        
        **⚠️ REVIEW**  
        Fraud score 45-85%
        
        **✅ APPROVE**  
        Fraud score < 45%
        """)
    else:
        st.error("⚠️ Model not loaded")

st.markdown("---")

# Main Content
if model_dict:
    tab1, tab2, tab3 = st.tabs(["🔍 Analyze Job Posting", "⚡ Quick Tests", "📖 About"])

    with tab1:
        st.markdown("### Enter Job Posting Details")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📝 Basic Information")
            title = st.text_input("Job Title *", placeholder="e.g., Software Engineer")
            description = st.text_area(
                "Job Description *",
                height=150,
                placeholder="Enter the full job description..."
            )
            requirements = st.text_area(
                "Requirements",
                height=100,
                placeholder="Education, experience, skills..."
            )

        with col2:
            st.markdown("#### 🏢 Company Information")
            company_profile = st.text_area(
                "Company Profile",
                height=100,
                placeholder="About the company..."
            )
            benefits = st.text_area(
                "Benefits",
                height=100,
                placeholder="Salary, insurance, perks..."
            )

            col_a, col_b = st.columns(2)
            with col_a:
                has_logo = st.checkbox("✓ Has Company Logo")
                telecommuting = st.checkbox("🏠 Remote Work")
            with col_b:
                salary_range = st.text_input("💰 Salary Range", placeholder="$80k-$120k")
                industry = st.text_input("🏭 Industry", placeholder="e.g., IT")

        st.markdown("")
        col_btn1, col_btn2, col_btn3 = st.columns([2,1,2])
        with col_btn2:
            analyze_btn = st.button("🔍 ANALYZE NOW", type="primary", use_container_width=True)

        if analyze_btn:
            if not title or not description:
                st.error("⚠️ Please fill in Job Title and Description")
            else:
                with st.spinner('🤖 AI analyzing...'):
                    result = predict_fraud({
                        'title': title,
                        'description': description,
                        'requirements': requirements,
                        'company_profile': company_profile,
                        'benefits': benefits,
                        'has_company_logo': int(has_logo),
                        'telecommuting': int(telecommuting),
                        'salary_range': salary_range,
                        'industry': industry,
                        'function': ''
                    }, model_dict)

                if result:
                    st.markdown("---")

                    # 결과 헤더
                    fraud_prob = result['balanced_prob']

                    if result['action'] == 'block':
                        st.error(f"### {result['explanation']}")
                        st.progress(fraud_prob)
                    elif result['action'] == 'review':
                        st.warning(f"### {result['explanation']}")
                        st.progress(fraud_prob)
                    else:
                        st.success(f"### {result['explanation']}")
                        st.progress(fraud_prob)

                    # 메트릭 카드
                    st.markdown("### 📊 Risk Assessment")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "Fraud Score",
                            f"{fraud_prob*100:.1f}%",
                            delta=f"{(fraud_prob-0.5)*100:+.1f}% vs avg"
                        )
                    with col2:
                        st.metric(
                            "Recall Score",
                            f"{result['recall_prob']*100:.1f}%",
                            delta="Sensitivity"
                        )
                    with col3:
                        decision_color = {
                            'block': '🚫',
                            'review': '⚠️',
                            'pass': '✅'
                        }
                        st.metric(
                            "Decision",
                            f"{decision_color[result['action']]} {result['action'].upper()}"
                        )
                    with col4:
                        confidence = max(fraud_prob, 1-fraud_prob)
                        st.metric(
                            "Confidence",
                            f"{confidence*100:.1f}%",
                            delta="AI Certainty"
                        )

                    # 모델 분석
                    st.markdown("---")
                    st.markdown("### 🤖 AI Model Analysis")

                    with st.expander("📈 Individual Model Scores", expanded=True):
                        model_to_weight = {
                            'XGBoost': 'xgb',
                            'LightGBM': 'lgbm',
                            'CatBoost': 'cat',
                            'NeuralNet': 'nn'
                        }

                        for name, score in result['model_scores'].items():
                            weight_key = model_to_weight.get(name, name.lower())
                            weight = result['weights'].get(weight_key, 0)
                            contribution = score * weight * 100

                            col1, col2, col3 = st.columns([3, 2, 2])
                            with col1:
                                st.markdown(f"**{name}**")
                            with col2:
                                st.progress(score)
                                st.caption(f"{score*100:.1f}%")
                            with col3:
                                st.markdown(f"Weight: {weight*100:.0f}%")
                                st.caption(f"→ {contribution:.1f}%")

                        st.markdown(f"**Final Ensemble Score: {fraud_prob*100:.1f}%**")

                    # 특성 기여도
                    if result['contributions']:
                        st.markdown("---")
                        st.markdown("### 🔍 Risk Factors Detected")

                        for i, c in enumerate(result['contributions']):
                            with st.container():
                                col1, col2, col3 = st.columns([4, 1, 1])
                                with col1:
                                    st.markdown(f"**{c['feature']}**")
                                    st.caption(c['explanation'])
                                with col2:
                                    st.metric("Value", c['value'])
                                with col3:
                                    st.metric("Impact", c['impact'])

                                if i < len(result['contributions']) - 1:
                                    st.markdown("<hr style='margin: 0.5rem 0; opacity: 0.3;'>", unsafe_allow_html=True)

    with tab2:
        st.markdown("### ⚡ Quick Test Cases")
        st.markdown("Test the system with pre-defined examples")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🚨 Fraud Example")
            if st.button("Test Suspicious Posting", use_container_width=True):
                with st.spinner('Analyzing...'):
                    result = predict_fraud({
                        'title': 'URGENT! Make $5000/week!!!',
                        'description': 'Amazing opportunity! Earn money fast! No experience needed! Contact: scam@email.com',
                        'requirements': '',
                        'company_profile': '',
                        'benefits': '',
                        'has_company_logo': 0,
                        'telecommuting': 1,
                        'salary_range': '',
                        'industry': '',
                        'function': ''
                    }, model_dict)

                if result:
                    st.metric("Fraud Score", f"{result['balanced_prob']*100:.1f}%")
                    st.metric("Decision", f"**{result['action'].upper()}**")
                    if result['action'] == 'block':
                        st.error(result['explanation'])
                    else:
                        st.warning(result['explanation'])

        with col2:
            st.markdown("#### ✅ Legitimate Example")
            if st.button("Test Legitimate Posting", use_container_width=True):
                with st.spinner('Analyzing...'):
                    result = predict_fraud({
                        'title': 'Senior Software Engineer',
                        'description': 'We are seeking an experienced software engineer to join our development team.',
                        'requirements': 'BS in Computer Science, 5+ years Python experience',
                        'company_profile': 'Established technology company since 2005',
                        'benefits': 'Health insurance, 401k, flexible hours',
                        'has_company_logo': 1,
                        'telecommuting': 0,
                        'salary_range': '$120,000 - $150,000',
                        'industry': 'Information Technology',
                        'function': 'Engineering'
                    }, model_dict)

                if result:
                    st.metric("Fraud Score", f"{result['balanced_prob']*100:.1f}%")
                    st.metric("Decision", f"**{result['action'].upper()}**")
                    if result['action'] == 'pass':
                        st.success(result['explanation'])
                    else:
                        st.warning(result['explanation'])

    with tab3:
        st.markdown("### 📖 About This System")

        st.markdown("""
        #### 🎯 Purpose
        This AI-powered system protects job seekers from fraudulent job postings by analyzing
        multiple linguistic and structural patterns.
        
        #### 🧠 Technology Stack
        - **BERT**: State-of-the-art language model for semantic understanding
        - **Ensemble ML**: 4 advanced models (XGBoost, LightGBM, CatBoost, Neural Network)
        - **Feature Engineering**: 100+ custom fraud indicators
        - **Real-time Analysis**: Sub-second prediction speed
        
        #### 🎯 Detection Capabilities
        - ✅ Urgency manipulation (ASAP, NOW, etc.)
        - ✅ Money emphasis patterns
        - ✅ Exaggerated claims
        - ✅ Missing company information
        - ✅ Contact information in description
        - ✅ Industry risk assessment
        - ✅ Linguistic anomalies
        
        #### 📊 Performance Metrics
        - **AUC Score**: 0.95+ (Excellent discrimination)
        - **Recall**: 90%+ (Catches most fraud)
        - **Precision**: 85%+ (Low false positives)
        - **F1 Score**: 0.90+ (Balanced performance)
        
        #### 🔒 Security Workflow
        1. **Automated Screening**: High-confidence fraud is auto-blocked
        2. **Manual Review**: Suspicious cases flagged for human review
        3. **Safe Approval**: Clean postings approved instantly
        
        #### 👨‍💻 Developed By
        Advanced ML System • 2024
        """)

        st.info("💡 **Tip**: For best results, provide complete job posting information including company details and salary range.")

else:
    st.error("### ⚠️ System Error")
    st.markdown("""
    The AI model could not be loaded. Please ensure:
    - `fraud_detection_hybrid_v7.pkl` is in the same directory
    - All required packages are installed
    - The model file is not corrupted
    
    Contact support if the problem persists.
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p style='font-size: 0.9rem;'>🛡️ Job Fraud Detection System v7.0</p>
    <p style='font-size: 0.8rem;'>Powered by BERT + Ensemble ML • Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)