"""
Job Posting Fraud Detection Dashboard
Streamlit App with 2-Stage Defense System
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from functools import lru_cache
from datetime import datetime

# ============================================================================
# Core Functions & Classes
# ============================================================================

@lru_cache(maxsize=1000)
def get_sentiment(text):
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    except:
        return 0.0, 0.0


class FeatureExtractor:
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
            f'{prefix}high_caps': int(caps_ratio > self.thresholds['caps']),
            f'{prefix}exclaim': text_str.count('!'),
            f'{prefix}high_exclaim': int(text_str.count('!') > self.thresholds['exclaim']),
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
            f'{prefix}high_polarity': int(polarity > self.thresholds['polarity']),
            f'{prefix}high_subj': int(subjectivity > self.thresholds['subjectivity']),
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
        return result


def analyze_fraud_signals(job_data, extractor):
    signals = []
    desc = str(job_data.get('description', '')).lower()
    title = str(job_data.get('title', '')).lower()

    if not job_data.get('has_company_logo'):
        signals.append("Missing company logo")
    if not job_data.get('company_profile'):
        signals.append("No company profile")

    urgency = ['urgent', 'hurry', 'now', 'asap', 'immediately']
    if any(w in desc or w in title for w in urgency):
        signals.append("Urgency pressure")

    money = ['$', 'earn', 'income', 'profit']
    if sum(w in desc or w in title for w in money) > 2:
        signals.append("Excessive money emphasis")

    if desc.count('!') + title.count('!') > 3:
        signals.append(f"Too many exclamations ({desc.count('!')+ title.count('!')})")

    if '@' in desc:
        signals.append("Email in description")

    completeness = sum([
        job_data.get('has_company_logo', 0),
        bool(job_data.get('salary_range')),
        bool(job_data.get('requirements'))
    ]) / 3

    if completeness < 0.3:
        signals.append(f"Low information quality ({completeness*100:.0f}%)")

    return signals


def two_stage_defense(job_data, model_dict):
    df = pd.DataFrame([job_data])
    extractor = model_dict['extractor']
    selector = model_dict['selector']

    X = extractor.transform(df)
    X_selected = selector.transform(X)

    balanced_model = model_dict['models']['balanced']
    if balanced_model.get('is_ensemble'):
        w = balanced_model['weights']
        prob_balanced = (
            w['xgb'] * balanced_model['xgb'].predict_proba(X_selected)[0,1] +
            w['lgbm'] * balanced_model['lgbm'].predict_proba(X_selected)[0,1] +
            w['cat'] * balanced_model['cat'].predict_proba(X_selected)[0,1]
        )
    else:
        prob_balanced = balanced_model['model'].predict_proba(X_selected)[0,1]

    recall_model = model_dict['models']['high_recall']
    if recall_model.get('is_ensemble'):
        prob_recall = (
            recall_model['xgb'].predict_proba(X_selected)[0,1] +
            recall_model['lgbm'].predict_proba(X_selected)[0,1] +
            recall_model['cat'].predict_proba(X_selected)[0,1]
        ) / 3
    else:
        prob_recall = recall_model['model'].predict_proba(X_selected)[0,1]

    if prob_balanced > 0.85:
        return {
            'action': 'block',
            'stage': 1,
            'probability': float(prob_balanced),
            'confidence': float(prob_balanced * 100),
            'explanation': 'AUTO-BLOCKED: High confidence fraud detection',
            'details': analyze_fraud_signals(job_data, extractor)
        }
    elif prob_recall > 0.50:
        return {
            'action': 'review',
            'stage': 2,
            'probability': float(prob_recall),
            'confidence': float(prob_recall * 100),
            'explanation': 'MANUAL REVIEW: Suspicious elements detected',
            'details': analyze_fraud_signals(job_data, extractor)
        }
    else:
        return {
            'action': 'pass',
            'stage': None,
            'probability': float(prob_recall),
            'confidence': float((1-prob_recall) * 100),
            'explanation': 'APPROVED: No fraud signals detected',
            'details': []
        }


# ============================================================================
# Streamlit UI
# ============================================================================

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 1rem;}
    .stAlert {margin-top: 0.5rem; margin-bottom: 0.5rem;}
    h1 {color: #1f77b4; font-size: 2.2rem;}
    h2 {font-size: 1.5rem; margin-top: 1rem;}
    h3 {font-size: 1.2rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 2rem;}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Job Posting Fraud Detection")
st.caption("2-Stage Defense System powered by ML Ensemble")
st.divider()

@st.cache_resource
def load_model():
    try:
        with open('fraud_detection_complete.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

def analyze_job(job_data, model_dict):
    defaults = {
        'title': '', 'description': '', 'requirements': '',
        'company_profile': '', 'benefits': '',
        'has_company_logo': 0, 'telecommuting': 0,
        'salary_range': '', 'industry': '', 'function': ''
    }
    job_data = {**defaults, **job_data}
    return two_stage_defense(job_data, model_dict)

# Sidebar
with st.sidebar:
    st.header("System Status")
    model_dict = load_model()

    if model_dict:
        st.success("✅ Model Active")
        perf = model_dict.get('performance', {})

        st.metric("Detection Rate", f"{perf.get('total_recall', 0)*100:.1f}%")
        st.metric("Auto-Blocked", f"{perf.get('stage1_detected', 0)}")
        st.metric("Manual Review", f"{perf.get('stage2_detected', 0)}")

        st.divider()

        st.subheader("📊 Understanding Risk Levels")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### Fraud Score Ranges
            
            | Score | Risk | Action |
            |-------|------|--------|
            | 85%+ | 🔴 Critical | Auto-block |
            | 50-85% | 🟠 High | Review |
            | 30-50% | 🟡 Medium | Approve* |
            | <30% | 🟢 Low | Approve |
            
            *Medium risk posts are approved but should be monitored
            """)

        with col2:
            st.markdown("##### 📈 Test Results")
            st.code(f"""
Total Fraud Cases:  130
✅ Auto-blocked:    {perf.get('stage1_detected', 0)} (Stage 1)
✅ In Review:       {perf.get('stage2_detected', 0)} (Stage 2)
❌ Missed:          {130 - (perf.get('stage1_detected', 0) + perf.get('stage2_detected', 0))}

Detection Rate:    {perf.get('total_recall', 0)*100:.1f}%
Review Workload:   {perf.get('review_workload', 0)} posts
            """, language="text")
            st.markdown("""
            #### What the scores mean
            
            **46% Fraud Score Example:**
            - 46% chance it's fraud
            - 54% chance it's legitimate
            - Below 50% threshold → Approved
            - But flagged as Medium Risk
            - Recommended: Monitor this posting
            
            **Why we approve 30-50%:**
            - To avoid false positives
            - Manual review burden
            - Business flexibility
            """)

        st.divider()

        with st.expander("ℹ️ How it works"):
            st.markdown("""
            **Risk Levels:**
            
            🔴 **CRITICAL** (85%+)  
            → Auto-blocked immediately
            
            🟠 **HIGH** (50-85%)  
            → Manual review required
            
            🟡 **MEDIUM** (30-50%)  
            → Approved, but monitor
            
            🟢 **LOW** (<30%)  
            → Safe to approve
            
            ---
            
            **2-Stage System:**
            - Stage 1: Auto-block (>85%)
            - Stage 2: Review (50-85%)
            - Pass: Approve (<50%)
            """)
    else:
        st.error("⚠️ Model Not Found")

# Main Content
if model_dict:
    tab1, tab2, tab3 = st.tabs(["🔍 Analyze", "⚡ Quick Test", "📊 Performance"])

    # ========================================================================
    # TAB 1: Analyze
    # ========================================================================
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Job Details")
            title = st.text_input("Job Title *", placeholder="e.g., Senior Software Engineer")
            company_profile = st.text_area("Company Info", height=80, placeholder="Founded year, certifications...")

            col_a, col_b = st.columns(2)
            with col_a:
                industry = st.text_input("Industry")
            with col_b:
                function = st.text_input("Function")

            col_c, col_d, col_e = st.columns(3)
            with col_c:
                has_logo = st.checkbox("Has Logo")
            with col_d:
                telecommuting = st.checkbox("Remote")
            with col_e:
                salary_range = st.text_input("Salary")

        with col2:
            st.subheader("Description")
            description = st.text_area("Job Description *", height=150, placeholder="Main responsibilities...")
            requirements = st.text_area("Requirements", height=80, placeholder="Skills, experience...")
            benefits = st.text_area("Benefits", height=80, placeholder="Health insurance, PTO...")

        st.divider()

        col_btn1, col_btn2 = st.columns([1, 5])
        with col_btn1:
            analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

        if analyze_btn:
            if not title or not description:
                st.error("⚠️ Title and Description are required")
            else:
                with st.spinner("Analyzing..."):
                    job_data = {
                        'title': title, 'description': description,
                        'requirements': requirements, 'company_profile': company_profile,
                        'benefits': benefits, 'has_company_logo': int(has_logo),
                        'telecommuting': int(telecommuting), 'salary_range': salary_range,
                        'industry': industry, 'function': function
                    }
                    result = analyze_job(job_data, model_dict)

                st.divider()

                # 위험도 계산
                fraud_prob = result['probability']
                if fraud_prob >= 0.85:
                    risk_level = "CRITICAL"
                    risk_color = "🔴"
                    risk_emoji = "🚨"
                elif fraud_prob >= 0.50:
                    risk_level = "HIGH"
                    risk_color = "🟠"
                    risk_emoji = "⚠️"
                elif fraud_prob >= 0.30:
                    risk_level = "MEDIUM"
                    risk_color = "🟡"
                    risk_emoji = "⚡"
                else:
                    risk_level = "LOW"
                    risk_color = "🟢"
                    risk_emoji = "✅"

                if result['action'] == 'block':
                    st.error(f"### 🚫 {result['explanation']}")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Fraud Score", f"{result['probability']*100:.1f}%",
                               delta=f"Critical", delta_color="inverse")
                    col2.metric("Risk Level", f"{risk_emoji} {risk_level}")
                    col3.metric("Decision", "🚫 BLOCKED")

                    st.error(f"**Action Required:** This posting is automatically blocked due to high fraud probability.")

                elif result['action'] == 'review':
                    st.warning(f"### ⚠️ {result['explanation']}")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Fraud Score", f"{result['probability']*100:.1f}%",
                               delta=f"High Risk", delta_color="inverse")
                    col2.metric("Risk Level", f"{risk_emoji} {risk_level}")
                    col3.metric("Decision", "👁️ REVIEW")

                    st.warning(f"**Action Required:** Manual review recommended. Check fraud signals below.")

                else:
                    # APPROVED 케이스
                    if fraud_prob >= 0.30:
                        # Medium Risk (30-50%)
                        st.warning(f"### ⚡ APPROVED (Medium Risk)")
                        st.caption("✅ Approved, but monitor closely - fraud score is moderately elevated")

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Fraud Score", f"{result['probability']*100:.1f}%",
                                   delta=f"Monitor", delta_color="normal")
                        col2.metric("Risk Level", f"{risk_emoji} {risk_level}")
                        col3.metric("Decision", "✅ APPROVED*")

                        st.info(f"**Note:** Fraud probability is {result['probability']*100:.0f}% (below 50% threshold). "
                               f"This is approved but recommended for monitoring.")
                    else:
                        # Low Risk (<30%)
                        st.success(f"### ✅ {result['explanation']}")
                        st.caption("Safe to proceed - very low fraud indicators")

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Fraud Score", f"{result['probability']*100:.1f}%",
                                   delta=f"Safe", delta_color="normal")
                        col2.metric("Risk Level", f"{risk_emoji} {risk_level}")
                        col3.metric("Decision", "✅ APPROVED")

                if result.get('details'):
                    with st.expander("🔍 Fraud Signals Detected"):
                        for signal in result['details']:
                            st.markdown(f"• {signal}")

    # ========================================================================
    # TAB 2: Quick Test
    # ========================================================================
    with tab2:
        st.subheader("Quick Test Cases")
        st.caption("Test the system with predefined examples")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 🚨 Obvious Fraud")
            st.caption("Expected: AUTO-BLOCKED")
            if st.button("Run Test 1", use_container_width=True):
                with st.spinner("Analyzing..."):
                    result = analyze_job({
                        'title': 'Work From Home - Earn $5000/week!',
                        'description': 'URGENT! Make EASY money NOW! Email: scam@fake.com. No experience needed!',
                        'has_company_logo': 0, 'telecommuting': 1
                    }, model_dict)

                    fraud_score = result['probability']*100
                    if result['action'] == 'block':
                        st.error(f"🚫 **BLOCKED** | Fraud Score: {fraud_score:.0f}%")
                    elif result['action'] == 'review':
                        st.warning(f"⚠️ **REVIEW NEEDED** | Fraud Score: {fraud_score:.0f}%")
                    else:
                        if fraud_score >= 30:
                            st.warning(f"⚡ **APPROVED** (Medium Risk: {fraud_score:.0f}%)")
                        else:
                            st.success(f"✅ **APPROVED** (Low Risk: {fraud_score:.0f}%)")

        with col2:
            st.markdown("##### ⚠️ Suspicious")
            st.caption("Expected: MANUAL REVIEW")
            if st.button("Run Test 2", use_container_width=True):
                with st.spinner("Analyzing..."):
                    result = analyze_job({
                        'title': 'Sales Rep - High Earning',
                        'description': 'Great income opportunity! Flexible hours.',
                        'has_company_logo': 0, 'telecommuting': 1
                    }, model_dict)

                    if result['action'] == 'block':
                        st.error(f"**BLOCKED** ({result['probability']*100:.0f}%)")
                    elif result['action'] == 'review':
                        st.warning(f"**REVIEW** ({result['probability']*100:.0f}%)")
                    else:
                        st.success(f"**PASSED** ({result['probability']*100:.0f}%)")

        with col3:
            st.markdown("##### ✅ Legitimate")
            st.caption("Expected: APPROVED")
            if st.button("Run Test 3", use_container_width=True):
                with st.spinner("Analyzing..."):
                    result = analyze_job({
                        'title': 'Senior Software Engineer',
                        'description': 'Seeking experienced engineer. Competitive salary and benefits.',
                        'company_profile': 'Tech company founded 2010, ISO certified',
                        'requirements': 'BS in CS, 5+ years experience',
                        'has_company_logo': 1, 'salary_range': '$120k-150k'
                    }, model_dict)

                    if result['action'] == 'block':
                        st.error(f"**BLOCKED** ({result['probability']*100:.0f}%)")
                    elif result['action'] == 'review':
                        st.warning(f"**REVIEW** ({result['probability']*100:.0f}%)")
                    else:
                        st.success(f"**PASSED** ({result['probability']*100:.0f}%)")

    # ========================================================================
    # TAB 3: Performance
    # ========================================================================
    with tab3:
        st.subheader("System Performance Metrics")

        perf = model_dict.get('performance', {})
        config = model_dict.get('two_stage_config', {})

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Detection Rate", f"{perf.get('total_recall', 0)*100:.1f}%",
                   help="% of fraud detected")
        col2.metric("Auto-Blocked", f"{perf.get('stage1_detected', 0)}",
                   help="Automatically blocked frauds")
        col3.metric("Review Queue", f"{perf.get('stage2_detected', 0)}",
                   help="Frauds caught in manual review")
        col4.metric("Workload", f"{perf.get('review_workload', 0)}",
                   help="Total posts requiring review")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### ⚙️ Thresholds")
            st.code(f"""
Stage 1 (Auto-block):  {config.get('stage1_threshold', 0.85)} (85%)
Stage 2 (Review):      {config.get('stage2_threshold', 0.50)} (50%)

Risk Levels:
- Critical: 85%+  (Auto-block)
- High:     50-85% (Review)
- Medium:   30-50% (Approve with caution)
- Low:      <30%   (Safe)
            """, language="text")

        with col2:
            st.markdown("##### 📈 Model Stack")
            st.info("""
            **Ensemble of 3 Models:**
            - XGBoost
            - LightGBM  
            - CatBoost
            
            **Features:** 46 selected from 91
            """)

else:
    st.error("### ⚠️ Model Not Loaded")
    st.info("""
    **Steps to fix:**
    1. Run `fraud_detection_complete.py` to train model
    2. Ensure `fraud_detection_complete.pkl` exists
    3. Restart this dashboard
    """)

st.divider()
st.caption("🛡️ Fraud Detection System v3.0 | XGBoost + LightGBM + CatBoost")