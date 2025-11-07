"""
Fraud Job Posting Detector - Dashboard
Real-time fraud detection for job postings
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import re
from textblob import TextBlob
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# Page config
st.set_page_config(
    page_title="Fraud Job Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .block-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .review-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 20px;
        border-radius: 10px;
        color: #333;
    }
    .pass-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 20px;
        border-radius: 10px;
        color: #333;
    }
    .stTextArea textarea {
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)


# ========== 🔥 필수 클래스 정의 (모델 로드용) ==========

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
        """텍스트 특성 추출"""
        if pd.isna(text) or text == '':
            return self._empty_features(prefix)

        text_str = str(text)
        text_lower = text_str.lower()
        words = text_str.split()
        word_count = len(words)
        sentence_count = max(len(re.findall(r'[.!?]+', text_str)), 1)
        text_length = len(text_str)

        polarity, subjectivity = get_sentiment(text_str)

        emails = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text_str))
        phones = len(re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text_str))
        urls = len(re.findall(r'http[s]?://[^\s]+', text_str))

        keyword_cnt = sum(kw in text_lower for kw in self.keywords)
        caps_ratio = sum(1 for c in text_str if c.isupper()) / max(len(text_str), 1)

        urgency_words = ['urgent', 'hurry', 'now', 'asap', 'immediately', 'limited time', 'act now', 'quick']
        urgency_raw = sum(w in text_lower for w in urgency_words)

        pressure_words = ['must', 'required', 'guarantee', 'easy', 'fast', 'quick', 'instant']
        pressure_raw = sum(w in text_lower for w in pressure_words)

        money_words = ['earn', 'income', 'profit', 'cash', 'money', '$', 'dollar', 'paid', 'pay']
        money_raw = sum(w in text_lower for w in money_words)

        exaggeration = ['amazing', 'incredible', 'unbelievable', 'guaranteed', '100%', 'unlimited', 'free',
                        'higher than']
        exag_raw = sum(w in text_lower for w in exaggeration)

        length_penalty = max(1.0, 200.0 / max(text_length, 50))

        urgency_weighted = urgency_raw * length_penalty * 3.0
        pressure_weighted = pressure_raw * length_penalty * 2.0
        money_weighted = money_raw * length_penalty * 1.5
        exag_weighted = exag_raw * length_penalty * 2.5

        combo_score = 0
        if urgency_raw > 0 and money_raw > 0:
            combo_score += 5
        if urgency_raw > 0 and exag_raw > 0:
            combo_score += 4
        if money_raw > 1 and exag_raw > 0:
            combo_score += 3
        if urgency_raw > 1 and money_raw > 1:
            combo_score += 6

        return {
            f'{prefix}length': text_length,
            f'{prefix}word_count': word_count,
            f'{prefix}sentence_count': sentence_count,
            f'{prefix}avg_word_len': np.mean([len(w) for w in words]) if words else 0,
            f'{prefix}avg_sent_len': word_count / sentence_count,
            f'{prefix}caps_ratio': caps_ratio,
            f'{prefix}high_caps': int(caps_ratio > self.thresholds['caps']),
            f'{prefix}exclaim': text_str.count('!'),
            f'{prefix}high_exclaim': int(text_str.count('!') > self.thresholds['exclaim']),
            f'{prefix}question': text_str.count('?'),
            f'{prefix}keyword': keyword_cnt,
            f'{prefix}has_keyword': int(keyword_cnt > 0),
            f'{prefix}urgency_raw': urgency_raw,
            f'{prefix}urgency': urgency_weighted,
            f'{prefix}pressure_raw': pressure_raw,
            f'{prefix}pressure': pressure_weighted,
            f'{prefix}money_raw': money_raw,
            f'{prefix}money': money_weighted,
            f'{prefix}exag_raw': exag_raw,
            f'{prefix}exag': exag_weighted,
            f'{prefix}manipulative': urgency_weighted + pressure_weighted + exag_weighted,
            f'{prefix}combo_score': combo_score,
            f'{prefix}is_short': int(text_length < 100),
            f'{prefix}is_very_short': int(text_length < 50),
            f'{prefix}length_penalty': length_penalty,
            f'{prefix}short_risk': urgency_raw + money_raw + exag_raw if text_length < 100 else 0,
            f'{prefix}email': emails,
            f'{prefix}phone': phones,
            f'{prefix}url': urls,
            f'{prefix}contacts': emails + phones,
            f'{prefix}polarity': polarity,
            f'{prefix}subjectivity': subjectivity,
            f'{prefix}high_polarity': int(polarity > self.thresholds['polarity']),
            f'{prefix}high_subj': int(subjectivity > self.thresholds['subjectivity']),
        }

    def _empty_features(self, prefix):
        keys = ['length', 'word_count', 'sentence_count', 'avg_word_len', 'avg_sent_len',
                'caps_ratio', 'high_caps', 'exclaim', 'high_exclaim', 'question',
                'keyword', 'has_keyword',
                'urgency_raw', 'urgency', 'pressure_raw', 'pressure',
                'money_raw', 'money', 'exag_raw', 'exag',
                'manipulative', 'combo_score',
                'is_short', 'is_very_short', 'length_penalty', 'short_risk',
                'email', 'phone', 'url', 'contacts', 'polarity', 'subjectivity',
                'high_polarity', 'high_subj']
        return {f'{prefix}{k}': 0 for k in keys}

    def extract_company_features(self, company_profile):
        """회사 신뢰도"""
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
        """산업/직무 위험도"""
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
        """메타데이터"""
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
            'low_info': int(completeness < 0.3),
        }

    def transform(self, df):
        """전체 변환"""
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

        # 상호작용 특성
        result['low_info_urgent'] = ((result['completeness'] < 0.2) & (result['d_urgency'] > 2)).astype(int)
        result['no_logo_money'] = ((result['has_logo'] == 0) & (result['d_money'] > 5)).astype(int)
        result['remote_high_subj'] = ((result['telecommute'] == 1) & (result['d_high_subj'] == 1)).astype(int)
        result['high_risk_low_info'] = (
                    (result['ind_risk'] > result['ind_risk'].mean() * 2) & (result['completeness'] < 0.4)).astype(int)
        result['no_salary_exag'] = ((result['has_salary'] == 0) & (result['d_exag'] > 2)).astype(int)
        result['contact_urgent'] = ((result['d_contacts'] > 0) & (result['d_urgency'] > 0)).astype(int)

        result['short_urgent'] = ((result['d_is_short'] == 1) & (result['d_urgency_raw'] > 0)).astype(int)
        result['short_money'] = ((result['d_is_short'] == 1) & (result['d_money_raw'] > 1)).astype(int)
        result['short_exag'] = ((result['d_is_short'] == 1) & (result['d_exag_raw'] > 0)).astype(int)
        result['very_short_urgent'] = ((result['d_is_very_short'] == 1) & (result['d_urgency_raw'] > 0)).astype(int)

        result['title_urgent_money'] = ((result['t_urgency_raw'] > 0) & (result['t_money_raw'] > 0)).astype(int)
        result['title_short_urgent'] = ((result['t_is_short'] == 1) & (result['t_urgency_raw'] > 0)).astype(int)

        result['exclaim_low_info'] = ((result['d_exclaim'] > 3) & (result['completeness'] < 0.3)).astype(int)
        result['contacts_low_info'] = ((result['d_contacts'] > 0) & (result['completeness'] < 0.4)).astype(int)
        result['short_contacts'] = ((result['d_is_short'] == 1) & (result['d_contacts'] > 0)).astype(int)
        result['money_exag_combo'] = ((result['d_money_raw'] > 1) & (result['d_exag_raw'] > 1)).astype(int)
        result['triple_threat'] = (
                    (result['d_urgency_raw'] > 0) & (result['d_money_raw'] > 0) & (result['d_exag_raw'] > 0)).astype(
            int)
        result['short_triple'] = ((result['d_is_short'] == 1) & (result['triple_threat'] == 1)).astype(int)

        return result


class TfidfExtractor:
    """TF-IDF 특성 추출기"""

    def __init__(self, max_features=300):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.8,
            strip_accents='unicode',
            lowercase=True,
            stop_words='english'
        )
        self.fitted = False

    def _combine_text(self, df):
        """title + description + requirements 결합"""
        combined = []
        for _, row in df.iterrows():
            parts = []
            if pd.notna(row.get('title')):
                parts.append(str(row['title']))
            if pd.notna(row.get('description')):
                parts.append(str(row['description']))
            if pd.notna(row.get('requirements')):
                parts.append(str(row['requirements']))

            text = ' '.join(parts) if parts else 'empty'
            combined.append(text)
        return combined

    def fit_transform(self, df):
        """학습 + 변환"""
        texts = self._combine_text(df)
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.fitted = True

        feature_names = self.vectorizer.get_feature_names_out()

        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{name}' for name in feature_names],
            index=df.index
        )

        return tfidf_df

    def transform(self, df):
        """변환만"""
        if not self.fitted:
            raise ValueError("TfidfExtractor가 아직 fit되지 않았습니다!")

        texts = self._combine_text(df)
        tfidf_matrix = self.vectorizer.transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()

        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{name}' for name in feature_names],
            index=df.index
        )

        return tfidf_df


class BERTEmbedder:
    """BERT embedding 생성기"""

    def __init__(self, model_name='all-MiniLM-L6-v2', n_components=64):
        self.model = SentenceTransformer(model_name)
        self.pca = PCA(n_components=n_components, random_state=42)
        self.pca_fitted = False
        self.n_components = n_components

    def transform(self, df, fit=False):
        """BERT embeddings 생성"""
        texts = []
        for _, row in df.iterrows():
            title = str(row.get('title', '')).strip()
            desc = str(row.get('description', '')).strip()
            text = f"{title} [SEP] {desc}" if title and desc else (title or desc)
            texts.append(text if text else "empty")

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True
        )

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


# ========== Dashboard Functions ==========

@st.cache_resource
def load_model():
    """Load trained model"""
    try:
        with open('fraud-detector-api/fraud_detection_hybrid_v8_tfidf.pkl', 'rb') as f:
            model_dict = pickle.load(f)
        return model_dict
    except FileNotFoundError:
        st.error("❌ Model file not found! Please train the model first.")
        st.stop()


def parse_linkedin_post(text):
    """Parse LinkedIn job posting text"""
    lines = text.strip().split('\n')

    # Initialize
    data = {
        'title': '',
        'description': '',
        'requirements': '',
        'benefits': '',
        'company_profile': '',
        'salary_range': '',
        'industry': '',
        'function': '',
        'has_company_logo': 0,
        'telecommuting': 0
    }

    # Try to extract title (usually first non-empty line)
    for line in lines:
        if line.strip():
            data['title'] = line.strip()
            break

    # Combine rest as description
    data['description'] = '\n'.join(lines[1:]).strip()

    # Try to detect sections
    full_text = text.lower()

    # Requirements section
    req_keywords = ['requirements', 'qualifications', 'required', 'must have', 'you should', 'ideal candidate']
    for keyword in req_keywords:
        if keyword in full_text:
            idx = full_text.index(keyword)
            data['requirements'] = text[idx:idx+500]
            break

    # Benefits section
    ben_keywords = ['benefits', 'we offer', 'perks', 'compensation', 'what we offer']
    for keyword in ben_keywords:
        if keyword in full_text:
            idx = full_text.index(keyword)
            data['benefits'] = text[idx:idx+300]
            break

    # Detect remote/telecommuting
    remote_keywords = ['remote', 'work from home', 'wfh', 'telecommute', 'virtual']
    if any(kw in full_text for kw in remote_keywords):
        data['telecommuting'] = 1

    # Try to extract salary
    salary_patterns = [
        r'\$[\d,]+\s*-\s*\$[\d,]+',
        r'\$[\d,]+k\s*-\s*\$[\d,]+k',
        r'[\d,]+\s*-\s*[\d,]+\s*per\s*(?:year|annum|hour)'
    ]
    for pattern in salary_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data['salary_range'] = match.group()
            break

    return data


def create_gauge_chart(probability, title):
    """Create gauge chart for probability"""

    # Determine color
    if probability >= 0.65:
        color = "red"
    elif probability >= 0.40:
        color = "orange"
    else:
        color = "green"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#d4edda'},
                {'range': [40, 65], 'color': '#fff3cd'},
                {'range': [65, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 65
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "darkblue", 'family': "Arial"}
    )

    return fig


def predict_fraud(model_dict, job_data):
    """Predict fraud probability"""

    # Extract components
    extractor = model_dict['domain_extractor']
    tfidf_extractor = model_dict['tfidf_extractor']
    bert_embedder = model_dict['bert_embedder']
    selector = model_dict['selector']
    models = model_dict['models_balanced']
    weights = models['weights']
    optimal_threshold_bal = model_dict['thresholds']['balanced']
    optimal_threshold_recall = model_dict['thresholds']['high_recall']

    # Create DataFrame
    df = pd.DataFrame([job_data])

    # Extract features
    X_domain = extractor.transform(df)
    X_tfidf = tfidf_extractor.transform(df)
    X_bert = bert_embedder.transform(df)
    X_hybrid = pd.concat([X_domain, X_tfidf, X_bert], axis=1)
    X_selected = selector.transform(X_hybrid)

    # Predict
    balanced_proba = (
        weights['xgb'] * models['xgb'].predict_proba(X_selected)[0, 1] +
        weights['lgbm'] * models['lgbm'].predict_proba(X_selected)[0, 1] +
        weights['cat'] * models['cat'].predict_proba(X_selected)[0, 1] +
        weights['nn'] * models['nn'].predict_proba(X_selected)[0, 1]
    )

    recall_models = model_dict['models_recall']
    recall_proba = (
        recall_models['xgb'].predict_proba(X_selected)[0, 1] +
        recall_models['lgbm'].predict_proba(X_selected)[0, 1] +
        recall_models['cat'].predict_proba(X_selected)[0, 1]
    ) / 3

    # Normal signals
    desc = str(job_data.get('description', '')).lower()
    professional_signals = [
        'responsibilities', 'qualifications', 'requirements',
        'benefits', 'insurance', 'experience', 'skills',
        'team', 'professional', 'career', 'growth'
    ]
    professional_count = sum(sig in desc for sig in professional_signals)

    has_logo = int(job_data.get('has_company_logo', 0))
    has_salary = int(pd.notna(job_data.get('salary_range')) and job_data.get('salary_range') != '')
    has_benefits = len(str(job_data.get('benefits', ''))) > 20
    is_professional = professional_count >= 4

    normal_score = has_logo + has_salary + int(has_benefits) + int(is_professional)

    # Decision
    if balanced_proba > 0.65:
        action = 'BLOCK'
        reason = 'High fraud confidence (65%+) - Immediate block recommended'
        color = 'red'
    elif balanced_proba > 0.40:
        action = 'REVIEW'
        reason = 'Medium risk - Manual review required'
        color = 'orange'
    elif recall_proba > optimal_threshold_recall:
        if normal_score >= 3:
            action = 'PASS'
            reason = f'Strong normal signals (score: {normal_score}/4)'
            color = 'green'
        else:
            action = 'REVIEW'
            reason = 'Caught by high-recall safety net'
            color = 'orange'
    else:
        action = 'PASS'
        reason = 'Normal job posting'
        color = 'green'

    return {
        'action': action,
        'reason': reason,
        'color': color,
        'balanced_prob': float(balanced_proba),
        'recall_prob': float(recall_proba),
        'normal_score': normal_score,
        'professional_count': professional_count,
        'features': {
            'has_logo': has_logo,
            'has_salary': has_salary,
            'has_benefits': has_benefits,
            'is_professional': is_professional
        }
    }


def main():
    # Header
    st.markdown('<div class="main-header">🔍 Fraud Job Posting Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered real-time fraud detection for job postings</div>', unsafe_allow_html=True)

    # Load model
    with st.spinner('Loading AI model...'):
        model_dict = load_model()

    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/6213/6213690.png", width=100)
        st.title("🎯 Analysis Settings")

        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.info(f"""
        **Version:** {model_dict['metadata']['version']}
        
        **Features:**
        - Rule-based: 102
        - TF-IDF: 300
        - BERT: 64
        
        **Performance:**
        - Precision: {model_dict['metadata']['final_performance']['hybrid']['precision']*100:.1f}%
        - Recall: {model_dict['metadata']['final_performance']['hybrid']['recall']*100:.1f}%
        - F1 Score: {model_dict['metadata']['final_performance']['hybrid']['f1']*100:.1f}%
        """)

        st.markdown("---")
        st.markdown("### 🎨 Decision Thresholds")
        st.markdown(f"""
        - 🔴 **BLOCK**: ≥65% fraud probability
        - 🟡 **REVIEW**: 40-65% fraud probability  
        - 🟢 **PASS**: <40% fraud probability
        """)

        st.markdown("---")
        st.markdown("### ℹ️ How to Use")
        st.markdown("""
        1. **Copy** job posting from LinkedIn
        2. **Paste** into input box below
        3. **Click** 'Analyze Job Posting'
        4. **Review** fraud analysis results
        """)

    # Main content
    tab1, tab2 = st.tabs(["🔍 Quick Analysis", "📝 Detailed Input"])

    with tab1:
        st.markdown("### 📋 Paste LinkedIn Job Posting")
        st.markdown("*Copy the entire job posting from LinkedIn and paste below*")

        job_text = st.text_area(
            "Job Posting Text",
            height=300,
            placeholder="""Paste job posting here, for example:

Exciting Career Opportunity - Join Our Team!

We are looking for dedicated individuals to join our growing company...

Requirements:
- Bachelor's degree or equivalent
- 2+ years of experience
- Strong communication skills

Benefits:
- Competitive salary
- Health insurance
- Career growth opportunities
""",
            label_visibility="collapsed"
        )

        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            analyze_btn = st.button("🚀 Analyze Job Posting", use_container_width=True, type="primary")

        if analyze_btn and job_text.strip():
            with st.spinner('Analyzing job posting...'):
                # Parse text
                job_data = parse_linkedin_post(job_text)

                # Predict
                result = predict_fraud(model_dict, job_data)

                # Display results
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")

                # Result card
                if result['action'] == 'BLOCK':
                    st.markdown(f"""
                    <div class="block-card">
                        <h2>🚫 BLOCK - Fraud Detected</h2>
                        <p style="font-size: 1.2rem;">{result['reason']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif result['action'] == 'REVIEW':
                    st.markdown(f"""
                    <div class="review-card">
                        <h2>⚠️ REVIEW - Manual Check Required</h2>
                        <p style="font-size: 1.2rem;">{result['reason']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="pass-card">
                        <h2>✅ PASS - Legitimate Posting</h2>
                        <p style="font-size: 1.2rem;">{result['reason']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("")

                # Metrics
                col1, col2, col3 = st.columns(3)

                with col1:
                    fig_balanced = create_gauge_chart(result['balanced_prob'], "Fraud Probability")
                    st.plotly_chart(fig_balanced, use_container_width=True)

                with col2:
                    fig_recall = create_gauge_chart(result['recall_prob'], "Safety Net Score")
                    st.plotly_chart(fig_recall, use_container_width=True)

                with col3:
                    st.markdown("### 🎯 Legitimacy Signals")
                    st.markdown(f"""
                    **Overall Score: {result['normal_score']}/4**
                    
                    - {'✅' if result['features']['has_logo'] else '❌'} Company Logo
                    - {'✅' if result['features']['has_salary'] else '❌'} Salary Range
                    - {'✅' if result['features']['has_benefits'] else '❌'} Benefits Listed
                    - {'✅' if result['features']['is_professional'] else '❌'} Professional Tone
                    
                    *Professional keywords: {result['professional_count']}*
                    """)

                # Detailed breakdown
                with st.expander("📈 Detailed Analysis"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### 📝 Extracted Information")
                        st.markdown(f"""
                        **Title:** {job_data['title'][:100]}...
                        
                        **Salary:** {job_data['salary_range'] if job_data['salary_range'] else 'Not specified'}
                        
                        **Remote:** {'Yes' if job_data['telecommuting'] else 'No'}
                        
                        **Description Length:** {len(job_data['description'])} chars
                        
                        **Requirements Found:** {'Yes' if job_data['requirements'] else 'No'}
                        
                        **Benefits Found:** {'Yes' if job_data['benefits'] else 'No'}
                        """)

                    with col2:
                        st.markdown("### 🎲 Confidence Metrics")

                        # Create confidence bar chart
                        confidence_data = pd.DataFrame({
                            'Metric': ['Balanced Model', 'Safety Net', 'Normal Signals'],
                            'Score': [
                                result['balanced_prob'] * 100,
                                result['recall_prob'] * 100,
                                (result['normal_score'] / 4) * 100
                            ]
                        })

                        fig_conf = px.bar(
                            confidence_data,
                            x='Score',
                            y='Metric',
                            orientation='h',
                            color='Score',
                            color_continuous_scale=['green', 'yellow', 'red'],
                            range_color=[0, 100]
                        )
                        fig_conf.update_layout(
                            showlegend=False,
                            height=250,
                            xaxis_title="Score (%)",
                            yaxis_title="",
                            margin=dict(l=0, r=0, t=20, b=0)
                        )
                        st.plotly_chart(fig_conf, use_container_width=True)

        elif analyze_btn:
            st.warning("⚠️ Please paste a job posting to analyze!")

    with tab2:
        st.markdown("### 📝 Manual Input (Advanced)")
        st.markdown("*Fill in each field separately for more accurate analysis*")

        with st.form("detailed_form"):
            col1, col2 = st.columns(2)

            with col1:
                title = st.text_input("Job Title", placeholder="e.g., Senior Software Engineer")
                description = st.text_area("Job Description", height=150, placeholder="Full job description...")
                requirements = st.text_area("Requirements", height=100, placeholder="Required qualifications...")
                company_profile = st.text_area("Company Profile", height=100, placeholder="About the company...")

            with col2:
                benefits = st.text_area("Benefits", height=100, placeholder="Offered benefits...")
                salary_range = st.text_input("Salary Range", placeholder="e.g., $80,000-$100,000")
                industry = st.text_input("Industry", placeholder="e.g., Technology")
                function = st.text_input("Function", placeholder="e.g., Engineering")

                has_logo = st.checkbox("Company Logo Present")
                telecommute = st.checkbox("Remote/Telecommute")

            submit_btn = st.form_submit_button("🚀 Analyze", use_container_width=True, type="primary")

            if submit_btn:
                job_data = {
                    'title': title,
                    'description': description,
                    'requirements': requirements,
                    'company_profile': company_profile,
                    'benefits': benefits,
                    'salary_range': salary_range,
                    'industry': industry,
                    'function': function,
                    'has_company_logo': int(has_logo),
                    'telecommuting': int(telecommute)
                }

                with st.spinner('Analyzing...'):
                    result = predict_fraud(model_dict, job_data)

                # Show same results as tab1
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")

                if result['action'] == 'BLOCK':
                    st.error(f"🚫 **BLOCK** - {result['reason']}")
                elif result['action'] == 'REVIEW':
                    st.warning(f"⚠️ **REVIEW** - {result['reason']}")
                else:
                    st.success(f"✅ **PASS** - {result['reason']}")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Fraud Probability", f"{result['balanced_prob']*100:.1f}%")

                with col2:
                    st.metric("Safety Net Score", f"{result['recall_prob']*100:.1f}%")

                with col3:
                    st.metric("Normal Score", f"{result['normal_score']}/4")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🤖 Powered by AI | v8 + TF-IDF Model | Built with Streamlit</p>
        <p style="font-size: 0.9rem;">⚠️ This tool provides AI-based predictions. Always verify suspicious postings manually.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

