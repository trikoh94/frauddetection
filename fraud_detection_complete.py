"""
사기 탐지 모델 - 완전판 v3 (오류 수정)
- 3가지 모드: balanced / high_recall / high_precision
- 2단계 방어 시스템
- 실시간 분석 함수
- 설명 가능한 예측
"""

import pandas as pd
import numpy as np
import pickle
import re
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from collections import Counter
from functools import lru_cache
import warnings
import logging
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# 캐싱된 감성 분석
# ============================================================================
@lru_cache(maxsize=1000)
def get_sentiment(text):
    """캐싱된 감성 분석"""
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    except Exception as e:
        return 0.0, 0.0


# ============================================================================
# 데이터 분석 함수들
# ============================================================================
def extract_keywords(df, top_n=30):
    """사기 키워드 추출"""
    logger.info("\n🔍 사기 키워드 추출 중...")
    fraud_texts = ' '.join(df[df['fraudulent']==1]['description'].fillna(''))
    normal_texts = ' '.join(df[df['fraudulent']==0]['description'].fillna(''))

    fraud_words = [w.lower() for w in re.findall(r'\b[a-z]{3,}\b', fraud_texts)]
    normal_words = [w.lower() for w in re.findall(r'\b[a-z]{3,}\b', normal_texts)]

    fraud_freq = Counter(fraud_words)
    normal_freq = Counter(normal_words)

    fraud_ratio = {}
    for word in fraud_freq:
        if fraud_freq[word] >= 5:
            fraud_rate = fraud_freq[word] / len(fraud_words)
            normal_rate = normal_freq.get(word, 0) / max(len(normal_words), 1)
            if fraud_rate > normal_rate * 2:
                fraud_ratio[word] = fraud_rate / max(normal_rate, 0.0001)

    top = sorted(fraud_ratio.items(), key=lambda x: x[1], reverse=True)[:top_n]
    logger.info(f"   ✓ Top 10: {[w for w, _ in top[:10]]}")
    return [w for w, _ in top]


def calculate_risks(df):
    """산업/직무 위험도 계산"""
    industry_risk = df.groupby('industry')['fraudulent'].agg(['mean', 'count'])
    industry_risk = industry_risk[industry_risk['count'] >= 10]

    function_risk = df.groupby('function')['fraudulent'].agg(['mean', 'count'])
    function_risk = function_risk[function_risk['count'] >= 10]

    ind_dict = {str(k).lower(): v for k, v in industry_risk['mean'].items()}
    func_dict = {str(k).lower(): v for k, v in function_risk['mean'].items()}

    return ind_dict, func_dict, df['fraudulent'].mean()


def calculate_thresholds(df):
    """임계값 계산"""
    logger.info("\n📊 임계값 계산 중...")
    thresholds = {}
    fraud_df = df[df['fraudulent']==1]

    polarities = []
    for desc in fraud_df['description'].fillna(''):
        pol, _ = get_sentiment(str(desc))
        polarities.append(pol)
    thresholds['polarity'] = np.percentile(polarities, 75) if polarities else 0.3

    subjs = []
    for desc in fraud_df['description'].fillna(''):
        _, subj = get_sentiment(str(desc))
        subjs.append(subj)
    thresholds['subjectivity'] = np.percentile(subjs, 75) if subjs else 0.5

    caps = [sum(1 for c in str(d) if c.isupper())/max(len(str(d)),1)
            for d in fraud_df['description'].fillna('') if len(str(d))>0]
    thresholds['caps'] = np.percentile(caps, 75) if caps else 0.15

    thresholds['exclaim'] = int(fraud_df['description'].fillna('').apply(lambda x: str(x).count('!')).quantile(0.75))

    logger.info(f"   ✓ 완료")
    return thresholds


# ============================================================================
# 특성 추출기
# ============================================================================
class FeatureExtractor:
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
            'low_info': int(completeness < 0.5),
        }

    def transform(self, df):
        """전체 변환"""
        logger.info(f"🔬 특성 추출 중... ({len(df)}개)")

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
        logger.info(f"   ✓ {len(result.columns)}개 특성 생성")
        return result


# ============================================================================
# 🆕 2단계 방어 시스템 (수정)
# ============================================================================
def two_stage_defense(job_data, model_dict):
    """
    2단계 방어 시스템

    Stage 1: High Precision 모델 (자동 차단)
    Stage 2: High Recall 모델 (관리자 검토)
    """
    df = pd.DataFrame([job_data])
    extractor = model_dict['extractor']
    selector = model_dict['selector']  # 🔧 추가

    # 특성 추출 + 선택
    X = extractor.transform(df)
    X_selected = selector.transform(X)  # 🔧 추가: 91개 → 46개

    # Stage 1: Balanced 모델 (높은 임계값)
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

    # Stage 2: High Recall 모델 (낮은 임계값)
    recall_model = model_dict['models']['high_recall']
    if recall_model.get('is_ensemble'):
        prob_recall = (
            recall_model['xgb'].predict_proba(X_selected)[0,1] +
            recall_model['lgbm'].predict_proba(X_selected)[0,1] +
            recall_model['cat'].predict_proba(X_selected)[0,1]
        ) / 3
    else:
        prob_recall = recall_model['model'].predict_proba(X_selected)[0,1]

    # 판단
    if prob_balanced > 0.85:
        return {
            'action': 'block',
            'stage': 1,
            'probability': float(prob_balanced),
            'confidence': float(prob_balanced * 100),
            'explanation': '🚫 자동 차단: 높은 확신도로 사기 판정 (Precision 모델)',
            'details': analyze_fraud_signals(job_data, extractor)
        }

    elif prob_recall > 0.50:
        return {
            'action': 'review',
            'stage': 2,
            'probability': float(prob_recall),
            'confidence': float(prob_recall * 100),
            'explanation': '⚠️ 관리자 검토 필요: 의심스러운 요소 발견 (Recall 모델)',
            'details': analyze_fraud_signals(job_data, extractor)
        }

    else:
        return {
            'action': 'pass',
            'stage': None,
            'probability': float(prob_recall),
            'confidence': float((1-prob_recall) * 100),
            'explanation': '✅ 정상: 사기 신호 없음',
            'details': []
        }


def analyze_fraud_signals(job_data, extractor):
    """사기 신호 분석"""
    signals = []

    desc = str(job_data.get('description', '')).lower()
    title = str(job_data.get('title', '')).lower()

    if not job_data.get('has_company_logo'):
        signals.append("회사 로고 없음")

    if not job_data.get('company_profile'):
        signals.append("회사 소개 없음")

    urgency = ['urgent', 'hurry', 'now', 'asap', 'immediately']
    if any(w in desc or w in title for w in urgency):
        signals.append("긴급성 강조")

    money = ['$', 'earn', 'income', 'profit']
    if sum(w in desc or w in title for w in money) > 2:
        signals.append("금전 과다 강조")

    if desc.count('!') + title.count('!') > 3:
        signals.append(f"느낌표 과다 ({desc.count('!')+ title.count('!')}개)")

    if '@' in desc:
        signals.append("설명에 이메일 포함")

    completeness = sum([
        job_data.get('has_company_logo', 0),
        bool(job_data.get('salary_range')),
        bool(job_data.get('requirements'))
    ]) / 3

    if completeness < 0.3:
        signals.append(f"정보 부족 ({completeness*100:.0f}%)")

    return signals


# ============================================================================
# 실시간 분석 함수 (수정)
# ============================================================================
def analyze_job_posting(job_data, model_path='fraud_detection_complete.pkl', use_two_stage=True):
    """
    실시간 채용 공고 분석
    """
    try:
        with open(model_path, 'rb') as f:
            model_dict = pickle.load(f)

        defaults = {
            'title': '', 'description': '', 'requirements': '',
            'company_profile': '', 'benefits': '',
            'has_company_logo': 0, 'telecommuting': 0,
            'salary_range': '', 'industry': '', 'function': ''
        }
        job_data = {**defaults, **job_data}

        if use_two_stage:
            return two_stage_defense(job_data, model_dict)
        else:
            # 기존 단일 모델 방식
            df = pd.DataFrame([job_data])
            extractor = model_dict['extractor']
            selector = model_dict['selector']  # 🔧 추가

            X = extractor.transform(df)
            X_selected = selector.transform(X)  # 🔧 추가: 91개 → 46개

            model = model_dict['models']['balanced']
            if model.get('is_ensemble'):
                w = model['weights']
                prob = (w['xgb'] * model['xgb'].predict_proba(X_selected)[0,1] +
                        w['lgbm'] * model['lgbm'].predict_proba(X_selected)[0,1] +
                        w['cat'] * model['cat'].predict_proba(X_selected)[0,1])
            else:
                prob = model['model'].predict_proba(X_selected)[0,1]

            threshold = model.get('threshold', 0.65)

            return {
                'fraud_probability': float(prob),
                'is_fraud': bool(prob > threshold),
                'confidence': float(max(prob, 1-prob) * 100),
                'risk_level': 'high' if prob > 0.7 else ('medium' if prob > 0.3 else 'low'),
                'signals': analyze_fraud_signals(job_data, model_dict['extractor'])
            }

    except Exception as e:
        return {'error': str(e)}


def print_analysis_report(result):
    """분석 결과 출력"""
    print("\n" + "="*70)
    print("📊 채용 공고 사기 분석 결과 (2단계 방어 시스템)")
    print("="*70)

    if 'error' in result:
        print(f"\n❌ 오류: {result['error']}")
        return

    if result['action'] == 'block':
        print(f"\n🚫 [1단계] 자동 차단")
        print(f"사기 확률: {result['probability']*100:.1f}%")
        print(f"신뢰도: {result['confidence']:.1f}%")
        print(f"\n{result['explanation']}")

    elif result['action'] == 'review':
        print(f"\n⚠️ [2단계] 관리자 검토 필요")
        print(f"사기 확률: {result['probability']*100:.1f}%")
        print(f"\n{result['explanation']}")

    else:
        print(f"\n✅ 정상 공고")
        print(f"사기 확률: {result['probability']*100:.1f}%")
        print(f"신뢰도: {result['confidence']:.1f}%")

    if result.get('details'):
        print(f"\n🔍 발견된 의심 신호:")
        for signal in result['details']:
            print(f"   • {signal}")

    print("="*70)


# ============================================================================
# 메인 (3가지 모드 학습)
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 사기 탐지 모델 학습 - 완전판 v3")
    print("="*70)

    # 데이터 로드
    df = pd.read_csv('fake_job_postings.csv')
    fraud_cnt = df['fraudulent'].sum()
    normal_cnt = len(df) - fraud_cnt

    logger.info(f"\n📊 데이터")
    logger.info(f"   전체: {len(df):,}개")
    logger.info(f"   정상: {normal_cnt:,}개 ({normal_cnt/len(df)*100:.1f}%)")
    logger.info(f"   사기: {fraud_cnt:,}개 ({fraud_cnt/len(df)*100:.1f}%)")

    # 분할
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['fraudulent'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['fraudulent'], random_state=42)

    logger.info(f"\n   Train: {len(train_df):,}개")
    logger.info(f"   Val: {len(val_df):,}개")
    logger.info(f"   Test: {len(test_df):,}개")

    # 특성 추출
    keywords = extract_keywords(train_df, 30)
    ind_risk, func_risk, overall_rate = calculate_risks(train_df)
    thresholds = calculate_thresholds(val_df)

    extractor = FeatureExtractor(keywords, ind_risk, func_risk, overall_rate, thresholds)
    X_train = extractor.transform(train_df)
    y_train = train_df['fraudulent'].values
    X_val = extractor.transform(val_df)
    y_val = val_df['fraudulent'].values
    X_test = extractor.transform(test_df)
    y_test = test_df['fraudulent'].values

    # 특성 선택
    logger.info("\n🎯 특성 선택 중...")
    selector = SelectFromModel(XGBClassifier(n_estimators=50, random_state=42), threshold='median')
    selector.fit(X_train, y_train)
    X_train_sel = selector.transform(X_train)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)
    logger.info(f"   ✓ {X_train_sel.shape[1]}개 특성 선택")

    models = {}

    # ========================================================================
    # 모드 1: Balanced (기본)
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info(f"📊 모드 1: BALANCED (F1 최적화)")
    logger.info(f"{'='*70}")

    pos_weight = (normal_cnt / fraud_cnt) * 1.2

    xgb_bal = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                            scale_pos_weight=pos_weight, subsample=0.8,
                            random_state=42, eval_metric='logloss')
    xgb_bal.fit(X_train_sel, y_train)

    lgbm_bal = LGBMClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                              scale_pos_weight=pos_weight, subsample=0.8,
                              random_state=42, verbose=-1)
    lgbm_bal.fit(X_train_sel, y_train)

    cat_bal = CatBoostClassifier(iterations=150, depth=5, learning_rate=0.05,
                                 scale_pos_weight=pos_weight, random_state=42, verbose=0)
    cat_bal.fit(X_train_sel, y_train)

    xgb_val = xgb_bal.predict_proba(X_val_sel)[:,1]
    lgbm_val = lgbm_bal.predict_proba(X_val_sel)[:,1]
    cat_val = cat_bal.predict_proba(X_val_sel)[:,1]

    xgb_score = roc_auc_score(y_val, xgb_val)
    lgbm_score = roc_auc_score(y_val, lgbm_val)
    cat_score = roc_auc_score(y_val, cat_val)

    total = xgb_score + lgbm_score + cat_score
    w_bal = {'xgb': xgb_score/total, 'lgbm': lgbm_score/total, 'cat': cat_score/total}

    ensemble_val = w_bal['xgb']*xgb_val + w_bal['lgbm']*lgbm_val + w_bal['cat']*cat_val

    best_f1 = 0
    best_thresh_bal = 0.5
    for th in np.arange(0.5, 0.8, 0.05):
        pred = (ensemble_val > th).astype(int)
        f1 = f1_score(y_val, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh_bal = th

    models['balanced'] = {
        'xgb': xgb_bal,
        'lgbm': lgbm_bal,
        'cat': cat_bal,
        'weights': w_bal,
        'threshold': best_thresh_bal,
        'is_ensemble': True
    }

    logger.info(f"   임계값: {best_thresh_bal:.2f}")
    logger.info(f"   F1: {best_f1:.3f}")

    # ========================================================================
    # 모드 2: High Recall
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info(f"🎯 모드 2: HIGH RECALL (사기 탐지 극대화)")
    logger.info(f"{'='*70}")

    logger.info(f"   SMOTE 적용...")
    smote = BorderlineSMOTE(sampling_strategy=0.35, random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_sel, y_train)
    logger.info(f"   사기: {sum(y_train==1)} → {sum(y_train_smote==1)}")

    pos_weight_high = (sum(y_train_smote==0) / sum(y_train_smote==1)) * 2.5

    xgb_rec = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.03,
                            scale_pos_weight=pos_weight_high, subsample=0.8,
                            random_state=42, eval_metric='logloss')
    xgb_rec.fit(X_train_smote, y_train_smote)

    lgbm_rec = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.03,
                              scale_pos_weight=pos_weight_high, subsample=0.8,
                              random_state=42, verbose=-1)
    lgbm_rec.fit(X_train_smote, y_train_smote)

    cat_rec = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.03,
                                 scale_pos_weight=pos_weight_high, random_state=42, verbose=0)
    cat_rec.fit(X_train_smote, y_train_smote)

    xgb_rec_val = xgb_rec.predict_proba(X_val_sel)[:,1]
    lgbm_rec_val = lgbm_rec.predict_proba(X_val_sel)[:,1]
    cat_rec_val = cat_rec.predict_proba(X_val_sel)[:,1]

    ensemble_rec_val = (xgb_rec_val + lgbm_rec_val + cat_rec_val) / 3

    best_recall = 0
    best_thresh_rec = 0.5
    for th in np.arange(0.3, 0.6, 0.05):
        pred = (ensemble_rec_val > th).astype(int)
        rec = recall_score(y_val, pred)
        if rec > best_recall:
            best_recall = rec
            best_thresh_rec = th

    models['high_recall'] = {
        'xgb': xgb_rec,
        'lgbm': lgbm_rec,
        'cat': cat_rec,
        'threshold': best_thresh_rec,
        'is_ensemble': True
    }

    logger.info(f"   임계값: {best_thresh_rec:.2f}")
    logger.info(f"   Recall: {best_recall:.3f}")

    # ========================================================================
    # 모드 3: High Precision
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info(f"🎯 모드 3: HIGH PRECISION (오탐 최소화)")
    logger.info(f"{'='*70}")

    # Balanced 모델 재사용, 높은 임계값만 적용
    best_prec = 0
    best_thresh_prec = 0.8
    for th in np.arange(0.75, 0.95, 0.05):
        pred = (ensemble_val > th).astype(int)
        if sum(pred) > 0:  # 예측이 있을 때만
            prec = precision_score(y_val, pred, zero_division=0)
            if prec > best_prec:
                best_prec = prec
                best_thresh_prec = th

    models['high_precision'] = {
        'xgb': xgb_bal,
        'lgbm': lgbm_bal,
        'cat': cat_bal,
        'weights': w_bal,
        'threshold': best_thresh_prec,
        'is_ensemble': True
    }

    logger.info(f"   임계값: {best_thresh_prec:.2f}")
    logger.info(f"   Precision: {best_prec:.3f}")

    # ========================================================================
    # Test 평가
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info(f"🎯 Test 최종 평가")
    logger.info(f"{'='*70}")

    for mode_name, model in models.items():
        logger.info(f"\n[{mode_name.upper()}]")

        if mode_name == 'high_recall':
            xgb_test = model['xgb'].predict_proba(X_test_sel)[:,1]
            lgbm_test = model['lgbm'].predict_proba(X_test_sel)[:,1]
            cat_test = model['cat'].predict_proba(X_test_sel)[:,1]
            test_proba = (xgb_test + lgbm_test + cat_test) / 3
        else:
            xgb_test = model['xgb'].predict_proba(X_test_sel)[:,1]
            lgbm_test = model['lgbm'].predict_proba(X_test_sel)[:,1]
            cat_test = model['cat'].predict_proba(X_test_sel)[:,1]
            w = model['weights']
            test_proba = w['xgb']*xgb_test + w['lgbm']*lgbm_test + w['cat']*cat_test

        test_pred = (test_proba > model['threshold']).astype(int)

        test_prec = precision_score(y_test, test_pred, zero_division=0)
        test_rec = recall_score(y_test, test_pred, zero_division=0)
        test_f1 = f1_score(y_test, test_pred, zero_division=0)
        test_auc = roc_auc_score(y_test, test_proba)

        cm = confusion_matrix(y_test, test_pred)
        tn, fp, fn, tp = cm.ravel()

        logger.info(f"   ROC-AUC:   {test_auc:.3f}")
        logger.info(f"   Precision: {test_prec:.3f}")
        logger.info(f"   Recall:    {test_rec:.3f}")
        logger.info(f"   F1-Score:  {test_f1:.3f}")
        logger.info(f"   미탐(FN):  {fn}개 / {fn+tp}개")
        logger.info(f"   오탐(FP):  {fp}개 / {fp+tn}개")

    # ========================================================================
    # 2단계 시스템 시뮬레이션
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info(f"🛡️ 2단계 방어 시스템 시뮬레이션")
    logger.info(f"{'='*70}")

    # Stage 1: High Precision (0.85 이상)
    bal_proba = (w_bal['xgb']*xgb_test + w_bal['lgbm']*lgbm_test + w_bal['cat']*cat_test)
    stage1_block = (bal_proba > 0.85).astype(int)

    # Stage 2: High Recall (0.50 이상, Stage 1에서 통과한 것만)
    rec_proba = (xgb_rec.predict_proba(X_test_sel)[:,1] +
                 lgbm_rec.predict_proba(X_test_sel)[:,1] +
                 cat_rec.predict_proba(X_test_sel)[:,1]) / 3
    stage2_review = ((rec_proba > 0.50) & (bal_proba <= 0.85)).astype(int)

    # 통과
    stage_pass = ((bal_proba <= 0.85) & (rec_proba <= 0.50)).astype(int)

    # 결과 집계
    stage1_fraud = sum((stage1_block == 1) & (y_test == 1))
    stage1_normal = sum((stage1_block == 1) & (y_test == 0))

    stage2_fraud = sum((stage2_review == 1) & (y_test == 1))
    stage2_normal = sum((stage2_review == 1) & (y_test == 0))

    pass_fraud = sum((stage_pass == 1) & (y_test == 1))
    pass_normal = sum((stage_pass == 1) & (y_test == 0))

    total_fraud = sum(y_test == 1)
    total_normal = sum(y_test == 0)

    logger.info(f"\n[Stage 1: 자동 차단] 임계값 > 0.85")
    logger.info(f"   차단: {sum(stage1_block)}개")
    logger.info(f"   - 진짜 사기: {stage1_fraud}개")
    logger.info(f"   - 정상(오탐): {stage1_normal}개")
    logger.info(f"   Precision: {stage1_fraud/(stage1_fraud+stage1_normal):.3f}" if (stage1_fraud+stage1_normal) > 0 else "   Precision: N/A")

    logger.info(f"\n[Stage 2: 관리자 검토] 0.50 < 임계값 <= 0.85")
    logger.info(f"   검토 필요: {sum(stage2_review)}개")
    logger.info(f"   - 진짜 사기: {stage2_fraud}개")
    logger.info(f"   - 정상(오탐): {stage2_normal}개")

    logger.info(f"\n[통과] 임계값 <= 0.50")
    logger.info(f"   통과: {sum(stage_pass)}개")
    logger.info(f"   - 놓친 사기: {pass_fraud}개")
    logger.info(f"   - 정상: {pass_normal}개")

    total_detected = stage1_fraud + stage2_fraud
    total_recall = total_detected / total_fraud
    total_review_workload = sum(stage2_review)

    logger.info(f"\n[종합]")
    logger.info(f"   총 사기 탐지: {total_detected}개 / {total_fraud}개 ({total_recall*100:.1f}%)")
    logger.info(f"   자동 차단: {stage1_fraud}개")
    logger.info(f"   관리자 검토로 발견: {stage2_fraud}개")
    logger.info(f"   놓친 사기: {pass_fraud}개")
    logger.info(f"   관리자 검토 부담: {total_review_workload}개")

    # ========================================================================
    # 모델 저장
    # ========================================================================
    save_dict = {
        'models': models,
        'extractor': extractor,
        'selector': selector,
        'two_stage_config': {
            'stage1_threshold': 0.85,
            'stage2_threshold': 0.50
        },
        'performance': {
            'total_recall': total_recall,
            'stage1_detected': stage1_fraud,
            'stage2_detected': stage2_fraud,
            'review_workload': total_review_workload
        }
    }

    with open('fraud_detection_complete.pkl', 'wb') as f:
        pickle.dump(save_dict, f)

    logger.info(f"\n💾 모델 저장: fraud_detection_complete.pkl")

    # ========================================================================
    # 실전 테스트
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info(f"🧪 실전 테스트")
    logger.info(f"{'='*70}")

    # 테스트 케이스 1: 명백한 사기
    test_job_1 = {
        'title': 'Work From Home - Earn $5000/week!',
        'description': 'URGENT! Make EASY money NOW! Email: scam@fake.com. Guaranteed income!',
        'requirements': '',
        'company_profile': '',
        'has_company_logo': 0,
        'telecommuting': 1,
    }

    logger.info("\n[테스트 1] 명백한 사기")
    result1 = analyze_job_posting(test_job_1, 'fraud_detection_complete.pkl', use_two_stage=True)
    print_analysis_report(result1)

    # 테스트 케이스 2: 정상 공고
    test_job_2 = {
        'title': 'Senior Software Engineer',
        'description': 'Looking for experienced engineer. Competitive salary and benefits.',
        'requirements': 'BS in CS, 5+ years Python',
        'company_profile': 'Tech company founded in 2010, ISO certified',
        'has_company_logo': 1,
        'salary_range': '$120k-150k',
    }

    logger.info("\n[테스트 2] 정상 공고")
    result2 = analyze_job_posting(test_job_2, 'fraud_detection_complete.pkl', use_two_stage=True)
    print_analysis_report(result2)

    # 테스트 케이스 3: 애매한 공고 (관리자 검토 대상)
    test_job_3 = {
        'title': 'Sales Rep - High Earning Potential',
        'description': 'Join our team! Great income opportunity. Flexible hours.',
        'requirements': 'Good communication',
        'company_profile': '',
        'has_company_logo': 0,
        'telecommuting': 1,
    }

    logger.info("\n[테스트 3] 애매한 공고")
    result3 = analyze_job_posting(test_job_3, 'fraud_detection_complete.pkl', use_two_stage=True)
    print_analysis_report(result3)

    logger.info(f"\n{'='*70}")
    logger.info(f"✅ 완료!")
    logger.info(f"{'='*70}")

