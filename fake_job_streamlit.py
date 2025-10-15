"""
Job Posting Fraud Detection Dashboard v7 - FIXED
모델 로딩 호환성 문제 해결
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re

from functools import lru_cache
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
# NLTK 초기화 (Streamlit Cloud용)
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

# 이제 TextBlob import
from textblob import TextBlob
# ============================================================================
# 필수: 학습 시 사용한 클래스들 (pickle 로드 전에 정의 필요!)
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
# 대시보드용 헬퍼 함수들
# ============================================================================

def get_feature_contributions(job_data):
    """각 특성이 사기 점수에 미친 영향 계산"""
    contributions = []

    desc = str(job_data.get('description', '')).lower()
    title = str(job_data.get('title', '')).lower()

    # 긴급성
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

    # 금전 강조
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

    # 과장
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

    # 느낌표
    exclaim_count = desc.count('!') + title.count('!')
    if exclaim_count > 3:
        impact = min((exclaim_count - 3) * 3, 15)
        contributions.append({
            'feature': '❗ Excessive Punctuation',
            'value': f"{exclaim_count} exclamations",
            'impact': f"+{impact}%",
            'explanation': "Too many exclamation marks - unprofessional"
        })

    # 연락처
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

    # 정보 완성도
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

        # 개별 모델 예측
        prob_xgb = models_bal['xgb'].predict_proba(X_selected)[0, 1]
        prob_lgbm = models_bal['lgbm'].predict_proba(X_selected)[0, 1]
        prob_cat = models_bal['cat'].predict_proba(X_selected)[0, 1]
        prob_nn = models_bal['nn'].predict_proba(X_selected)[0, 1]

        # 가중 평균
        prob_balanced = (
            weights['xgb'] * prob_xgb +
            weights['lgbm'] * prob_lgbm +
            weights['cat'] * prob_cat +
            weights['nn'] * prob_nn
        )

        # High recall
        prob_recall = (
            models_recall['xgb'].predict_proba(X_selected)[0, 1] +
            models_recall['lgbm'].predict_proba(X_selected)[0, 1] +
            models_recall['cat'].predict_proba(X_selected)[0, 1]
        ) / 3

        # 결정
        if prob_balanced > 0.85:
            action = 'block'
            explanation = '🚫 AUTO-BLOCKED: High confidence fraud'
        elif prob_balanced > 0.65:  # Balanced 모델만 보기
            action = 'review'
            explanation = '⚠️ MANUAL REVIEW: High risk detected'
        elif prob_balanced > 0.45 and prob_recall > 0.75:  # 두 모델 다 의심
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
# Streamlit UI
# ============================================================================

st.set_page_config(page_title="Fraud Detection v7", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 1.5rem;}
    h1 {color: #1f77b4;}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Job Posting Fraud Detection v7")
st.caption("BERT Hybrid + 4-Model Ensemble + Detailed Explanations")
st.divider()

# 모델 로드
@st.cache_resource
def load_model():
    try:
        with open('fraud_detection_hybrid_v7.pkl', 'rb') as f:
            model_dict = pickle.load(f)
        st.sidebar.success("✅ Model loaded!")
        return model_dict
    except Exception as e:
        st.sidebar.error(f"❌ Load failed: {str(e)}")
        return None

model_dict = load_model()

# Sidebar
with st.sidebar:
    st.header("📊 System Status")
    if model_dict:
        metadata = model_dict.get('metadata', {})
        perf = metadata.get('final_performance', {})
        st.metric("Test AUC", f"{perf.get('hybrid', {}).get('auc', 0):.4f}")
        st.metric("Test F1", f"{perf.get('hybrid', {}).get('f1', 0):.4f}")

# Main
if model_dict:
    tab1, tab2 = st.tabs(["🔍 Analyze", "⚡ Quick Test"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            title = st.text_input("Job Title *")
            description = st.text_area("Description *", height=150)
            requirements = st.text_area("Requirements", height=80)

        with col2:
            company_profile = st.text_area("Company Profile", height=100)
            benefits = st.text_area("Benefits", height=80)

            col_a, col_b = st.columns(2)
            with col_a:
                has_logo = st.checkbox("Has Logo")
                telecommuting = st.checkbox("Remote")
            with col_b:
                salary_range = st.text_input("Salary")
                industry = st.text_input("Industry")

        if st.button("🔍 Analyze", type="primary"):
            if not title or not description:
                st.error("Title and Description required")
            else:
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
                    st.divider()

                    # 결과 표시
                    fraud_prob = result['balanced_prob']

                    if result['action'] == 'block':
                        st.error(f"### {result['explanation']}")
                    elif result['action'] == 'review':
                        st.warning(f"### {result['explanation']}")
                    else:
                        st.success(f"### {result['explanation']}")

                    # 메트릭
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Fraud Score", f"{fraud_prob*100:.1f}%")
                    col2.metric("Recall Score", f"{result['recall_prob']*100:.1f}%")
                    col3.metric("Decision", result['action'].upper())

                    # 모델 분석
                    st.divider()
                    st.markdown("### 🎯 Why This Score?")

                    # Replace the "Model Breakdown" section (around line 530-540) with this:

                    st.markdown("#### 🤖 Model Breakdown")

                    # Proper mapping from model names to weight keys
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
                        st.write(f"**{name}:** {score * 100:.1f}% × {weight * 100:.1f}% = {contribution:.1f}%")

                    st.write(f"**→ Final: {fraud_prob * 100:.1f}%**")
                    # 특성 기여도
                    if result['contributions']:
                        st.divider()
                        st.markdown("#### 🔍 Feature Contributions")

                        for c in result['contributions']:
                            col1, col2, col3 = st.columns([3, 1, 1])
                            col1.write(f"**{c['feature']}**")
                            col1.caption(c['explanation'])
                            col2.metric("Value", c['value'])
                            col3.metric("Impact", c['impact'])
                            st.markdown("---")

    with tab2:
        st.subheader("Quick Tests")

        if st.button("🚨 Test Fraud Case"):
            result = predict_fraud({
                'title': 'URGENT! Make $5000/week!!!',
                'description': 'Amazing! Earn money fast! Email: scam@email.com',
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
                st.metric("Score", f"{result['balanced_prob']*100:.1f}%")
                st.write(f"Decision: **{result['action'].upper()}**")

        if st.button("✅ Test Legitimate Case"):
            result = predict_fraud({
                'title': 'Senior Software Engineer',
                'description': 'Seeking experienced engineer for our team.',
                'requirements': 'BS CS, 5+ years Python',
                'company_profile': 'Tech company since 2005',
                'benefits': 'Health, 401k',
                'has_company_logo': 1,
                'telecommuting': 0,
                'salary_range': '$120k-$150k',
                'industry': 'IT',
                'function': 'Engineering'
            }, model_dict)

            if result:
                st.metric("Score", f"{result['balanced_prob']*100:.1f}%")
                st.write(f"Decision: **{result['action'].upper()}**")

else:
    st.error("### ⚠️ Model Not Loaded")
    st.info("Please run: `python fraud_detection_complete_v7.py`")

st.divider()
st.caption("🛡️ Fraud Detection v7")

