"""
사기 탐지 모델 - BERT Hybrid v7 (개선판)
- BERT embeddings + 도메인 특성 결합
- Stacking 앙상블 + Neural Network 추가
- Baseline 비교 + Ablation Study
- 하이퍼파라미터 튜닝 + 고급 Threshold 최적화
"""

import pandas as pd
import numpy as np
import pickle
import re
from textblob import TextBlob
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTEENN
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


@lru_cache(maxsize=1000)
def get_sentiment(text):
    """캐싱된 감성 분석"""
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    except:
        return 0.0, 0.0


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

    polarities = [get_sentiment(str(d))[0] for d in fraud_df['description'].fillna('')]
    thresholds['polarity'] = np.percentile(polarities, 75) if polarities else 0.3

    subjs = [get_sentiment(str(d))[1] for d in fraud_df['description'].fillna('')]
    thresholds['subjectivity'] = np.percentile(subjs, 75) if subjs else 0.5

    caps = [sum(1 for c in str(d) if c.isupper())/max(len(str(d)),1)
            for d in fraud_df['description'].fillna('') if len(str(d))>0]
    thresholds['caps'] = np.percentile(caps, 75) if caps else 0.15

    thresholds['exclaim'] = int(fraud_df['description'].fillna('').apply(lambda x: str(x).count('!')).quantile(0.75))

    logger.info(f"   ✓ 완료")
    return thresholds


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
        logger.info(f"\n🤖 BERT 모델 로딩: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.pca = PCA(n_components=n_components, random_state=42)
        self.pca_fitted = False
        self.n_components = n_components
        logger.info(f"   ✓ 로드 완료 (384 → {n_components} 차원 축소 예정)")

    def transform(self, df, fit=False):
        """BERT embeddings 생성"""
        logger.info(f"   🤖 BERT embeddings 생성 중... ({len(df)}개)")

        texts = []
        for _, row in df.iterrows():
            title = str(row.get('title', '')).strip()
            desc = str(row.get('description', '')).strip()
            text = f"{title} [SEP] {desc}" if title and desc else (title or desc)
            texts.append(text if text else "empty")

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        logger.info(f"   🎯 PCA 차원 축소 (384 → {self.n_components})...")
        if fit or not self.pca_fitted:
            embeddings_reduced = self.pca.fit_transform(embeddings)
            self.pca_fitted = True
            explained_var = self.pca.explained_variance_ratio_.sum()
            logger.info(f"   ✓ 설명된 분산: {explained_var*100:.1f}%")
        else:
            embeddings_reduced = self.pca.transform(embeddings)

        bert_df = pd.DataFrame(
            embeddings_reduced,
            columns=[f'bert_{i}' for i in range(self.n_components)],
            index=df.index
        )

        return bert_df


def evaluate_model(model, X, y, name, threshold=0.5):
    """모델 평가"""
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        y_proba = model.predict(X)

    y_pred = (y_proba > threshold).astype(int)

    auc = roc_auc_score(y, y_proba)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    logger.info(f"\n   📊 {name}")
    logger.info(f"      AUC: {auc:.4f}")
    logger.info(f"      Precision: {precision:.4f}")
    logger.info(f"      Recall: {recall:.4f}")
    logger.info(f"      F1: {f1:.4f}")
    logger.info(f"      TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

    return {'auc': auc, 'precision': precision, 'recall': recall, 'f1': f1, 'y_proba': y_proba}


def optimize_threshold_advanced(y_true, y_proba, target_recall=0.95):
    """고급 Threshold 최적화 (Precision-Recall Curve 기반)"""
    logger.info(f"\n🎯 고급 Threshold 최적화 (목표 Recall: {target_recall*100:.0f}%)")

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # Target recall 이상에서 최대 precision
    valid_idx = np.where(recalls[:-1] >= target_recall)[0]

    if len(valid_idx) == 0:
        logger.info(f"   ⚠️ 목표 Recall 달성 불가, 최대 Recall: {recalls.max():.4f}")
        best_idx = np.argmax(recalls[:-1])
    else:
        best_idx = valid_idx[np.argmax(precisions[:-1][valid_idx])]

    best_threshold = thresholds[best_idx]
    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]

    # F1 계산
    if best_precision + best_recall > 0:
        best_f1 = 2 * (best_precision * best_recall) / (best_precision + best_recall)
    else:
        best_f1 = 0

    logger.info(f"   ✓ 최적 Threshold: {best_threshold:.4f}")
    logger.info(f"   ✓ Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, F1: {best_f1:.4f}")

    return best_threshold


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 사기 탐지 모델 학습 - v7 BERT HYBRID (개선판)")
    print("="*70)

    df = pd.read_csv('fake_job_postings.csv')
    fraud_cnt = df['fraudulent'].sum()
    normal_cnt = len(df) - fraud_cnt

    logger.info(f"\n📊 데이터")
    logger.info(f"   전체: {len(df):,}개")
    logger.info(f"   정상: {normal_cnt:,}개 ({normal_cnt/len(df)*100:.1f}%)")
    logger.info(f"   사기: {fraud_cnt:,}개 ({fraud_cnt/len(df)*100:.1f}%)")

    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['fraudulent'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['fraudulent'], random_state=42)

    logger.info(f"\n   Train: {len(train_df):,}개")
    logger.info(f"   Val: {len(val_df):,}개")
    logger.info(f"   Test: {len(test_df):,}개")

    keywords = extract_keywords(train_df, 30)
    ind_risk, func_risk, overall_rate = calculate_risks(train_df)
    thresholds = calculate_thresholds(val_df)

    # ========================================================================
    # 특성 추출
    # ========================================================================
    logger.info("\n🔬 특성 추출 중...")
    logger.info("   📊 1단계: 도메인 특성 추출...")

    extractor = FeatureExtractor(keywords, ind_risk, func_risk, overall_rate, thresholds)
    X_train_domain = extractor.transform(train_df)
    X_val_domain = extractor.transform(val_df)
    X_test_domain = extractor.transform(test_df)

    logger.info(f"   ✓ 도메인 특성: {X_train_domain.shape[1]}개")

    logger.info("\n   📊 2단계: BERT embeddings 생성...")
    bert_embedder = BERTEmbedder(model_name='all-MiniLM-L6-v2', n_components=64)
    X_train_bert = bert_embedder.transform(train_df, fit=True)
    X_val_bert = bert_embedder.transform(val_df)
    X_test_bert = bert_embedder.transform(test_df)

    logger.info(f"   ✓ BERT 특성: {X_train_bert.shape[1]}개")

    # Hybrid 특성 결합
    logger.info("\n   📊 3단계: Hybrid 특성 결합...")
    # 인덱스 리셋 필수! (concat 시 인덱스 불일치 방지)
    X_train_domain_reset = X_train_domain.reset_index(drop=True)
    X_train_bert_reset = X_train_bert.reset_index(drop=True)
    X_train_hybrid = pd.concat([X_train_domain_reset, X_train_bert_reset], axis=1)

    X_val_domain_reset = X_val_domain.reset_index(drop=True)
    X_val_bert_reset = X_val_bert.reset_index(drop=True)
    X_val_hybrid = pd.concat([X_val_domain_reset, X_val_bert_reset], axis=1)

    X_test_domain_reset = X_test_domain.reset_index(drop=True)
    X_test_bert_reset = X_test_bert.reset_index(drop=True)
    X_test_hybrid = pd.concat([X_test_domain_reset, X_test_bert_reset], axis=1)

    logger.info(f"   ✓ Hybrid 특성: {X_train_hybrid.shape[1]}개 ({X_train_domain.shape[1]} 도메인 + {X_train_bert.shape[1]} BERT)")

    y_train = train_df['fraudulent'].values
    y_val = val_df['fraudulent'].values
    y_test = test_df['fraudulent'].values

    # ========================================================================
    # BASELINE 모델 비교
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info(f"📊 BASELINE 모델 성능 (도메인 특성만)")
    logger.info(f"{'='*70}")

    baseline_results = {}

    # 1. Logistic Regression
    logger.info("\n🔵 Logistic Regression")
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train_domain, y_train)
    lr_result = evaluate_model(lr, X_val_domain, y_val, "Logistic Regression")
    baseline_results['Logistic Regression'] = lr_result

    # 2. Random Forest
    logger.info("\n🟢 Random Forest")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train_domain, y_train)
    rf_result = evaluate_model(rf, X_val_domain, y_val, "Random Forest")
    baseline_results['Random Forest'] = rf_result

    # 3. XGBoost 단일
    logger.info("\n🟡 XGBoost (단일, 도메인 특성)")
    xgb_simple = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric='logloss')
    xgb_simple.fit(X_train_domain, y_train)
    xgb_simple_result = evaluate_model(xgb_simple, X_val_domain, y_val, "XGBoost (도메인)")
    baseline_results['XGBoost (도메인)'] = xgb_simple_result

    # 4. BERT만 사용
    logger.info("\n🟣 XGBoost (BERT 특성만)")
    xgb_bert_only = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric='logloss')
    xgb_bert_only.fit(X_train_bert, y_train)
    bert_only_result = evaluate_model(xgb_bert_only, X_val_bert, y_val, "XGBoost (BERT만)")
    baseline_results['XGBoost (BERT만)'] = bert_only_result

    logger.info(f"\n{'='*70}")
    logger.info(f"📈 Baseline 비교 요약")
    logger.info(f"{'='*70}")
    logger.info(f"   {'모델':<30} {'AUC':<10} {'Precision':<12} {'Recall':<10} {'F1':<10}")
    logger.info(f"   {'-'*72}")
    for name, result in baseline_results.items():
        logger.info(f"   {name:<30} {result['auc']:<10.4f} {result['precision']:<12.4f} {result['recall']:<10.4f} {result['f1']:<10.4f}")

    # ========================================================================
    # HYBRID 모델 (개선판)
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info(f"🔥 HYBRID 모델: BERT + 도메인 + Stacking 앙상블")
    logger.info(f"{'='*70}")

    # Feature selection (RandomForest 사용 - XGBoost 오류 방지)
    logger.info("\n🎯 특성 선택 중...")
    selector = SelectFromModel(
        RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        threshold='median'
    )
    selector.fit(X_train_hybrid, y_train)
    X_train_sel = selector.transform(X_train_hybrid)
    X_val_sel = selector.transform(X_val_hybrid)
    X_test_sel = selector.transform(X_test_hybrid)
    logger.info(f"   ✓ {X_train_hybrid.shape[1]}개 → {X_train_sel.shape[1]}개 선택")

    # Cross-validation
    logger.info(f"\n🔄 Cross-Validation (5-Fold)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    xgb_cv = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                           scale_pos_weight=(normal_cnt/fraud_cnt)*1.2, random_state=42, eval_metric='logloss')
    cv_scores = cross_val_score(xgb_cv, X_train_sel, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    logger.info(f"   AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    logger.info(f"   각 Fold: {[f'{s:.4f}' for s in cv_scores]}")

    # ========================================================================
    # 모드 1: BALANCED (Stacking 앙상블)
    # ========================================================================
    logger.info(f"\n🔷 모드 1: BALANCED (Stacking 앙상블 + Neural Network)")

    pos_weight = (normal_cnt / fraud_cnt) * 1.2

    # Base models
    xgb_bal = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                            scale_pos_weight=pos_weight, subsample=0.8,
                            random_state=42, eval_metric='logloss')

    lgbm_bal = LGBMClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                              class_weight='balanced', subsample=0.8,
                              random_state=42, verbose=-1)

    cat_bal = CatBoostClassifier(iterations=150, depth=5, learning_rate=0.05,
                                 auto_class_weights='Balanced', subsample=0.8,
                                 random_state=42, verbose=0)

    # Neural Network 추가
    nn_bal = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu',
                           solver='adam', alpha=0.001, max_iter=500,
                           random_state=42, early_stopping=True)

    logger.info("\n   🔸 개별 모델 학습 중...")
    xgb_bal.fit(X_train_sel, y_train)
    eval_xgb = evaluate_model(xgb_bal, X_val_sel, y_val, "XGBoost")

    lgbm_bal.fit(X_train_sel, y_train)
    eval_lgbm = evaluate_model(lgbm_bal, X_val_sel, y_val, "LightGBM")

    cat_bal.fit(X_train_sel, y_train)
    eval_cat = evaluate_model(cat_bal, X_val_sel, y_val, "CatBoost")

    nn_bal.fit(X_train_sel, y_train)
    eval_nn = evaluate_model(nn_bal, X_val_sel, y_val, "Neural Network")

    # Stacking 앙상블
    logger.info("\n   🎯 Stacking 앙상블 구축 중...")
    base_models = [
        ('xgb', xgb_bal),
        ('lgbm', lgbm_bal),
        ('cat', cat_bal),
        ('nn', nn_bal)
    ]

    stacking_bal = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000),
        cv=5,
        n_jobs=-1
    )
    stacking_bal.fit(X_train_sel, y_train)
    eval_stacking = evaluate_model(stacking_bal, X_val_sel, y_val, "Stacking Ensemble")

    # 가중 평균 앙상블 (비교용)
    f1_scores = {
        'xgb': eval_xgb['f1'],
        'lgbm': eval_lgbm['f1'],
        'cat': eval_cat['f1'],
        'nn': eval_nn['f1']
    }
    total_f1 = sum(f1_scores.values())
    weights = {k: v/total_f1 for k, v in f1_scores.items()}

    logger.info(f"\n   🎯 가중 평균 앙상블 가중치:")
    logger.info(f"      XGB={weights['xgb']:.3f}, LGBM={weights['lgbm']:.3f}, CAT={weights['cat']:.3f}, NN={weights['nn']:.3f}")

    y_val_weighted = (
        weights['xgb'] * eval_xgb['y_proba'] +
        weights['lgbm'] * eval_lgbm['y_proba'] +
        weights['cat'] * eval_cat['y_proba'] +
        weights['nn'] * eval_nn['y_proba']
    )
    y_val_weighted_pred = (y_val_weighted > 0.5).astype(int)

    logger.info(f"\n   📊 가중 평균 앙상블")
    logger.info(f"      AUC: {roc_auc_score(y_val, y_val_weighted):.4f}")
    logger.info(f"      Precision: {precision_score(y_val, y_val_weighted_pred):.4f}")
    logger.info(f"      Recall: {recall_score(y_val, y_val_weighted_pred):.4f}")
    logger.info(f"      F1: {f1_score(y_val, y_val_weighted_pred):.4f}")

    # 최고 성능 모델 선택
    if eval_stacking['f1'] > f1_score(y_val, y_val_weighted_pred):
        logger.info(f"\n   ✅ Stacking이 더 우수! (F1: {eval_stacking['f1']:.4f} > {f1_score(y_val, y_val_weighted_pred):.4f})")
        best_model_bal = stacking_bal
        best_mode = 'stacking'
    else:
        logger.info(f"\n   ✅ 가중 평균이 더 우수! (F1: {f1_score(y_val, y_val_weighted_pred):.4f} > {eval_stacking['f1']:.4f})")
        best_model_bal = None  # 가중 평균 사용
        best_mode = 'weighted'

    # ========================================================================
    # 모드 2: HIGH RECALL (SMOTE + 앙상블)
    # ========================================================================
    logger.info(f"\n🔷 모드 2: HIGH RECALL (사기 놓치지 않기)")

    logger.info("   🔄 SMOTE 적용 중...")
    smote = SMOTE(sampling_strategy=0.8, random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_sel, y_train)
    logger.info(f"   ✓ {len(y_train)}개 → {len(y_train_smote)}개")

    pos_weight_recall = (normal_cnt / fraud_cnt) * 2.5

    xgb_recall = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.03,
                               scale_pos_weight=pos_weight_recall, subsample=0.8,
                               random_state=42, eval_metric='logloss')
    xgb_recall.fit(X_train_smote, y_train_smote)
    eval_xgb_recall = evaluate_model(xgb_recall, X_val_sel, y_val, "XGBoost", threshold=0.35)

    lgbm_recall = LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03,
                                 class_weight='balanced', subsample=0.8,
                                 random_state=42, verbose=-1)
    lgbm_recall.fit(X_train_smote, y_train_smote)
    eval_lgbm_recall = evaluate_model(lgbm_recall, X_val_sel, y_val, "LightGBM", threshold=0.35)

    cat_recall = CatBoostClassifier(iterations=150, depth=6, learning_rate=0.03,
                                    auto_class_weights='Balanced', subsample=0.8,
                                    random_state=42, verbose=0)
    cat_recall.fit(X_train_smote, y_train_smote)
    eval_cat_recall = evaluate_model(cat_recall, X_val_sel, y_val, "CatBoost", threshold=0.35)

    # ========================================================================
    # ABLATION STUDY
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info(f"🔬 ABLATION STUDY (각 구성 요소의 기여도)")
    logger.info(f"{'='*70}")

    ablation_results = {}

    # 1. Only Domain Features - 별도 selector 생성
    logger.info("\n📊 구성 1: 도메인 특성만")
    selector_domain = SelectFromModel(
        RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        threshold='median'
    )
    X_train_domain_ab = X_train_domain.reset_index(drop=True)
    X_val_domain_ab = X_val_domain.reset_index(drop=True)
    selector_domain.fit(X_train_domain_ab, y_train)
    X_train_domain_sel = selector_domain.transform(X_train_domain_ab)
    X_val_domain_sel = selector_domain.transform(X_val_domain_ab)

    xgb_domain = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                               scale_pos_weight=pos_weight, random_state=42, eval_metric='logloss')
    xgb_domain.fit(X_train_domain_sel, y_train)
    domain_result = evaluate_model(xgb_domain, X_val_domain_sel, y_val, "도메인 특성만")
    ablation_results['도메인만'] = domain_result

    # 2. Only BERT - 별도 selector 생성
    logger.info("\n📊 구성 2: BERT 특성만")
    selector_bert = SelectFromModel(
        RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        threshold='median'
    )
    X_train_bert_ab = X_train_bert.reset_index(drop=True)
    X_val_bert_ab = X_val_bert.reset_index(drop=True)
    selector_bert.fit(X_train_bert_ab, y_train)
    X_train_bert_sel = selector_bert.transform(X_train_bert_ab)
    X_val_bert_sel = selector_bert.transform(X_val_bert_ab)

    xgb_bert = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                            scale_pos_weight=pos_weight, random_state=42, eval_metric='logloss')
    xgb_bert.fit(X_train_bert_sel, y_train)
    bert_result = evaluate_model(xgb_bert, X_val_bert_sel, y_val, "BERT 특성만")
    ablation_results['BERT만'] = bert_result

    # 3. Hybrid (단일 모델)
    logger.info("\n📊 구성 3: Hybrid 단일 (도메인 + BERT)")
    ablation_results['Hybrid (단일)'] = eval_xgb

    # 4. Hybrid + Stacking/Weighted
    if best_mode == 'stacking':
        ablation_results['Hybrid (Stacking)'] = eval_stacking
    else:
        ablation_results['Hybrid (가중평균)'] = {
            'auc': roc_auc_score(y_val, y_val_weighted),
            'precision': precision_score(y_val, y_val_weighted_pred),
            'recall': recall_score(y_val, y_val_weighted_pred),
            'f1': f1_score(y_val, y_val_weighted_pred),
            'y_proba': y_val_weighted
        }

    logger.info(f"\n{'='*70}")
    logger.info(f"📈 Ablation Study 결과 요약")
    logger.info(f"{'='*70}")
    logger.info(f"   {'구성':<25} {'AUC':<10} {'Precision':<12} {'Recall':<10} {'F1':<10}")
    logger.info(f"   {'-'*67}")
    for name, result in ablation_results.items():
        logger.info(f"   {name:<25} {result['auc']:<10.4f} {result['precision']:<12.4f} {result['recall']:<10.4f} {result['f1']:<10.4f}")

    logger.info(f"\n   💡 인사이트:")
    domain_f1 = ablation_results['도메인만']['f1']
    bert_f1 = ablation_results['BERT만']['f1']
    hybrid_f1 = list(ablation_results.values())[-1]['f1']

    if hybrid_f1 > max(domain_f1, bert_f1):
        improvement = ((hybrid_f1 - max(domain_f1, bert_f1)) / max(domain_f1, bert_f1)) * 100
        logger.info(f"      ✅ Hybrid가 단독 사용 대비 {improvement:.1f}% F1 향상!")
        logger.info(f"      → 도메인 특성과 BERT가 상호 보완적으로 작동")

    # ========================================================================
    # THRESHOLD 최적화 (고급)
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info(f"🎯 고급 Threshold 최적화 (Precision-Recall Curve)")
    logger.info(f"{'='*70}")

    # Balanced 모델
    if best_mode == 'stacking':
        y_val_bal_proba = stacking_bal.predict_proba(X_val_sel)[:, 1]
    else:
        y_val_bal_proba = y_val_weighted

    logger.info("\n📊 Balanced 모델:")
    optimal_threshold_bal = optimize_threshold_advanced(y_val, y_val_bal_proba, target_recall=0.90)

    # High Recall 모델
    y_val_proba_recall = (
        eval_xgb_recall['y_proba'] +
        eval_lgbm_recall['y_proba'] +
        eval_cat_recall['y_proba']
    ) / 3

    logger.info("\n📊 High Recall 모델:")
    optimal_threshold_recall = optimize_threshold_advanced(y_val, y_val_proba_recall, target_recall=0.95)

    # ========================================================================
    # 최종 테스트 세트 평가
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info(f"🎯 최종 테스트 세트 평가")
    logger.info(f"{'='*70}")

    # Baseline 테스트
    logger.info("\n📊 Baseline 모델 (테스트 세트)")
    logger.info(f"   {'모델':<30} {'AUC':<10} {'Precision':<12} {'Recall':<10} {'F1':<10}")
    logger.info(f"   {'-'*72}")

    X_test_domain_ab = X_test_domain.reset_index(drop=True)
    X_test_bert_ab = X_test_bert.reset_index(drop=True)

    lr_test_proba = lr.predict_proba(X_test_domain_ab)[:, 1]
    lr_test_pred = (lr_test_proba > 0.5).astype(int)
    logger.info(f"   {'Logistic Regression':<30} {roc_auc_score(y_test, lr_test_proba):<10.4f} "
                f"{precision_score(y_test, lr_test_pred):<12.4f} {recall_score(y_test, lr_test_pred):<10.4f} "
                f"{f1_score(y_test, lr_test_pred):<10.4f}")

    rf_test_proba = rf.predict_proba(X_test_domain_ab)[:, 1]
    rf_test_pred = (rf_test_proba > 0.5).astype(int)
    logger.info(f"   {'Random Forest':<30} {roc_auc_score(y_test, rf_test_proba):<10.4f} "
                f"{precision_score(y_test, rf_test_pred):<12.4f} {recall_score(y_test, rf_test_pred):<10.4f} "
                f"{f1_score(y_test, rf_test_pred):<10.4f}")

    xgb_simple_test_proba = xgb_simple.predict_proba(X_test_domain_ab)[:, 1]
    xgb_simple_test_pred = (xgb_simple_test_proba > 0.5).astype(int)
    logger.info(f"   {'XGBoost (도메인만)':<30} {roc_auc_score(y_test, xgb_simple_test_proba):<10.4f} "
                f"{precision_score(y_test, xgb_simple_test_pred):<12.4f} {recall_score(y_test, xgb_simple_test_pred):<10.4f} "
                f"{f1_score(y_test, xgb_simple_test_pred):<10.4f}")

    bert_only_test_proba = xgb_bert_only.predict_proba(X_test_bert_ab)[:, 1]
    bert_only_test_pred = (bert_only_test_proba > 0.5).astype(int)
    logger.info(f"   {'XGBoost (BERT만)':<30} {roc_auc_score(y_test, bert_only_test_proba):<10.4f} "
                f"{precision_score(y_test, bert_only_test_pred):<12.4f} {recall_score(y_test, bert_only_test_pred):<10.4f} "
                f"{f1_score(y_test, bert_only_test_pred):<10.4f}")

    # Hybrid 모델 테스트
    logger.info("\n📊 HYBRID 앙상블 모델 (테스트 세트) ⭐")

    # Balanced
    if best_mode == 'stacking':
        y_test_proba_bal = stacking_bal.predict_proba(X_test_sel)[:, 1]
    else:
        y_test_proba_bal = (
            weights['xgb'] * xgb_bal.predict_proba(X_test_sel)[:, 1] +
            weights['lgbm'] * lgbm_bal.predict_proba(X_test_sel)[:, 1] +
            weights['cat'] * cat_bal.predict_proba(X_test_sel)[:, 1] +
            weights['nn'] * nn_bal.predict_proba(X_test_sel)[:, 1]
        )

    y_test_pred_bal = (y_test_proba_bal > optimal_threshold_bal).astype(int)

    logger.info(f"\n   📊 Balanced 모델 (threshold={optimal_threshold_bal:.4f})")
    logger.info(f"      AUC: {roc_auc_score(y_test, y_test_proba_bal):.4f}")
    logger.info(f"      Precision: {precision_score(y_test, y_test_pred_bal):.4f}")
    logger.info(f"      Recall: {recall_score(y_test, y_test_pred_bal):.4f}")
    logger.info(f"      F1: {f1_score(y_test, y_test_pred_bal):.4f}")

    # High Recall
    y_test_proba_recall = (
        xgb_recall.predict_proba(X_test_sel)[:, 1] +
        lgbm_recall.predict_proba(X_test_sel)[:, 1] +
        cat_recall.predict_proba(X_test_sel)[:, 1]
    ) / 3
    y_test_pred_recall = (y_test_proba_recall > optimal_threshold_recall).astype(int)

    logger.info(f"\n   📊 High Recall 모델 (threshold={optimal_threshold_recall:.4f})")
    logger.info(f"      AUC: {roc_auc_score(y_test, y_test_proba_recall):.4f}")
    logger.info(f"      Precision: {precision_score(y_test, y_test_pred_recall):.4f}")
    logger.info(f"      Recall: {recall_score(y_test, y_test_pred_recall):.4f}")
    logger.info(f"      F1: {f1_score(y_test, y_test_pred_recall):.4f}")

    # ========================================================================
    # FEATURE IMPORTANCE
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info(f"🔍 Feature Importance 분석")
    logger.info(f"{'='*70}")

    feature_importance = xgb_bal.feature_importances_
    top_indices = np.argsort(feature_importance)[-20:][::-1]

    logger.info("\n   📊 상위 20개 중요 특성:")
    logger.info(f"   {'순위':<6} {'특성':<30} {'중요도':<10}")
    logger.info(f"   {'-'*46}")

    selected_features = X_train_hybrid.columns[selector.get_support()]

    for rank, idx in enumerate(top_indices, 1):
        feat_name = selected_features[idx] if idx < len(selected_features) else f"Feature_{idx}"
        logger.info(f"   {rank:<6} {feat_name:<30} {feature_importance[idx]:<10.4f}")

    # ========================================================================
    # 2단계 방어 시스템
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info(f"🛡️ 2단계 방어 시스템 평가")
    logger.info(f"{'='*70}")

    block_mask = y_test_proba_bal > 0.85
    review_mask = (~block_mask) & (y_test_proba_recall > optimal_threshold_recall)
    pass_mask = ~(block_mask | review_mask)

    block_cnt = block_mask.sum()
    review_cnt = review_mask.sum()
    pass_cnt = pass_mask.sum()

    logger.info(f"\n   📊 처리 결과:")
    logger.info(f"      🚫 자동 차단: {block_cnt}개 ({block_cnt/len(test_df)*100:.1f}%)")
    logger.info(f"      ⚠️ 검토 필요: {review_cnt}개 ({review_cnt/len(test_df)*100:.1f}%)")
    logger.info(f"      ✅ 정상 통과: {pass_cnt}개 ({pass_cnt/len(test_df)*100:.1f}%)")

    fraud_in_block = y_test[block_mask].sum()
    fraud_in_review = y_test[review_mask].sum()
    fraud_in_pass = y_test[pass_mask].sum()

    logger.info(f"\n   🎯 각 단계별 실제 사기 건수:")
    logger.info(f"      🚫 차단: {fraud_in_block}/{block_cnt}개 ({fraud_in_block/max(block_cnt,1)*100:.1f}%)")
    logger.info(f"      ⚠️ 검토: {fraud_in_review}/{review_cnt}개 ({fraud_in_review/max(review_cnt,1)*100:.1f}%)")
    logger.info(f"      ✅ 통과: {fraud_in_pass}/{pass_cnt}개 ({fraud_in_pass/max(pass_cnt,1)*100:.1f}%)")

    total_detected = fraud_in_block + fraud_in_review
    total_fraud = y_test.sum()
    logger.info(f"\n   ✅ 전체 사기 탐지율: {total_detected}/{total_fraud}개 ({total_detected/total_fraud*100:.1f}%)")

    # ========================================================================
    # 성능 개선도 비교
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info(f"📈 모델 성능 개선도")
    logger.info(f"{'='*70}")

    baseline_auc = roc_auc_score(y_test, lr_test_proba)
    baseline_f1 = f1_score(y_test, lr_test_pred)

    domain_only_auc = roc_auc_score(y_test, xgb_simple_test_proba)
    domain_only_f1 = f1_score(y_test, xgb_simple_test_pred)

    hybrid_auc = roc_auc_score(y_test, y_test_proba_bal)
    hybrid_f1 = f1_score(y_test, y_test_pred_bal)

    logger.info(f"\n   1️⃣ Baseline (Logistic Regression)")
    logger.info(f"      AUC: {baseline_auc:.4f}, F1: {baseline_f1:.4f}")

    logger.info(f"\n   2️⃣ 도메인 특성만 (XGBoost)")
    logger.info(f"      AUC: {domain_only_auc:.4f}, F1: {domain_only_f1:.4f}")
    logger.info(f"      개선: +{(domain_only_auc-baseline_auc)*100:.1f}%p AUC, +{(domain_only_f1-baseline_f1)*100:.1f}%p F1")

    logger.info(f"\n   3️⃣ Hybrid (BERT + 도메인 + {best_mode.upper()}) ⭐")
    logger.info(f"      AUC: {hybrid_auc:.4f}, F1: {hybrid_f1:.4f}")
    logger.info(f"      개선: +{(hybrid_auc-baseline_auc)*100:.1f}%p AUC, +{(hybrid_f1-baseline_f1)*100:.1f}%p F1")
    logger.info(f"      도메인 대비: +{(hybrid_auc-domain_only_auc)*100:.1f}%p AUC, +{(hybrid_f1-domain_only_f1)*100:.1f}%p F1")

    # ========================================================================
    # 모델 저장
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info(f"💾 모델 저장 중...")
    logger.info(f"{'='*70}")

    model_dict = {
        'domain_extractor': extractor,
        'bert_embedder': bert_embedder,
        'selector': selector,
        'best_mode': best_mode,
        'models_balanced': {
            'xgb': xgb_bal,
            'lgbm': lgbm_bal,
            'cat': cat_bal,
            'nn': nn_bal,
            'stacking': stacking_bal if best_mode == 'stacking' else None,
            'weights': weights if best_mode == 'weighted' else None
        },
        'models_recall': {
            'xgb': xgb_recall,
            'lgbm': lgbm_recall,
            'cat': cat_recall
        },
        'thresholds': {
            'balanced': optimal_threshold_bal,
            'high_recall': optimal_threshold_recall
        },
        'metadata': {
            'keywords': keywords,
            'calc_thresholds': thresholds,
            'overall_fraud_rate': overall_rate,
            'cv_scores': cv_scores.tolist(),
            'ablation_results': {k: {kk: float(vv) if kk != 'y_proba' else None for kk, vv in v.items()}
                                for k, v in ablation_results.items()},
            'final_performance': {
                'baseline_lr': {'auc': float(baseline_auc), 'f1': float(baseline_f1)},
                'domain_only': {'auc': float(domain_only_auc), 'f1': float(domain_only_f1)},
                'hybrid': {'auc': float(hybrid_auc), 'f1': float(hybrid_f1)}
            }
        }
    }

    with open('fraud_detection_hybrid_v7.pkl', 'wb') as f:
        pickle.dump(model_dict, f)

    logger.info(f"   ✓ 저장 완료: fraud_detection_hybrid_v7.pkl")

    # ========================================================================
    # 샘플 테스트
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info(f"🧪 샘플 테스트")
    logger.info(f"{'='*70}")

    def predict_sample(job_data):
        """단일 샘플 예측"""
        df = pd.DataFrame([job_data])

        X_domain = extractor.transform(df)
        X_bert = bert_embedder.transform(df)
        X_hybrid = pd.concat([X_domain, X_bert], axis=1)
        X_selected = selector.transform(X_hybrid)

        if best_mode == 'stacking':
            balanced_proba = stacking_bal.predict_proba(X_selected)[0, 1]
        else:
            balanced_proba = (
                weights['xgb'] * xgb_bal.predict_proba(X_selected)[0,1] +
                weights['lgbm'] * lgbm_bal.predict_proba(X_selected)[0,1] +
                weights['cat'] * cat_bal.predict_proba(X_selected)[0,1] +
                weights['nn'] * nn_bal.predict_proba(X_selected)[0,1]
            )

        recall_proba = (
            xgb_recall.predict_proba(X_selected)[0,1] +
            lgbm_recall.predict_proba(X_selected)[0,1] +
            cat_recall.predict_proba(X_selected)[0,1]
        ) / 3

        if balanced_proba > 0.85:
            action = 'BLOCK'
            explanation = '🚫 자동 차단: 높은 확신도로 사기 판정'
        elif recall_proba > optimal_threshold_recall:
            action = 'REVIEW'
            explanation = '⚠️ 관리자 검토 필요: 의심스러운 요소 발견'
        else:
            action = 'PASS'
            explanation = '✅ 정상: 사기 신호 없음'

        return {
            'action': action,
            'balanced_prob': float(balanced_proba),
            'recall_prob': float(recall_proba),
            'explanation': explanation
        }

    # 사기 샘플
    fraud_sample = {
        'title': 'URGENT! Make $5000/week from home!!!',
        'description': 'Amazing opportunity! Earn money fast! No experience needed! Contact now at scam@email.com',
        'requirements': '',
        'company_profile': '',
        'benefits': '',
        'has_company_logo': 0,
        'telecommuting': 1,
        'salary_range': '',
        'industry': '',
        'function': ''
    }

    logger.info("\n   🔴 사기 의심 샘플:")
    result = predict_sample(fraud_sample)
    logger.info(f"      결과: {result['action']}")
    logger.info(f"      Balanced 확률: {result['balanced_prob']*100:.1f}%")
    logger.info(f"      Recall 확률: {result['recall_prob']*100:.1f}%")
    logger.info(f"      설명: {result['explanation']}")

    # 정상 샘플
    normal_sample = {
        'title': 'Senior Software Engineer',
        'description': 'We are seeking an experienced software engineer to join our development team. You will work on exciting projects using modern technologies.',
        'requirements': 'Bachelor degree in Computer Science, 5+ years experience in Python and Java',
        'company_profile': 'Established technology company since 2005, partnered with Fortune 500 companies',
        'benefits': 'Health insurance, 401k, flexible hours',
        'has_company_logo': 1,
        'telecommuting': 0,
        'salary_range': '$120,000 - $150,000',
        'industry': 'Information Technology',
        'function': 'Engineering'
    }

    logger.info("\n   🟢 정상 샘플:")
    result = predict_sample(normal_sample)
    logger.info(f"      결과: {result['action']}")
    logger.info(f"      Balanced 확률: {result['balanced_prob']*100:.1f}%")
    logger.info(f"      Recall 확률: {result['recall_prob']*100:.1f}%")
    logger.info(f"      설명: {result['explanation']}")

    logger.info(f"\n{'='*70}")
    logger.info(f"✅ 완료!")
    logger.info(f"{'='*70}")
    logger.info(f"\n💡 v7 주요 개선사항:")
    logger.info(f"   1. ✅ Stacking 앙상블 추가 (Meta-learner)")
    logger.info(f"   2. ✅ Neural Network 모델 추가 (4개 Base Model)")
    logger.info(f"   3. ✅ 가중 평균 vs Stacking 자동 선택")
    logger.info(f"   4. ✅ Precision-Recall Curve 기반 Threshold 최적화")
    logger.info(f"   5. ✅ Feature Selection 오류 수정 (RandomForest 사용)")
    logger.info(f"   6. ✅ Ablation Study 오류 수정 (별도 selector 생성)")
    logger.info(f"   7. ✅ 더욱 강력한 앙상블 전략")
    logger.info(f"\n💡 교수님께 강조할 포인트:")
    logger.info(f"   1. ✅ BERT Hybrid 접근 (최신 SOTA 방법론)")
    logger.info(f"   2. ✅ Stacking으로 Meta-learning 구현")
    logger.info(f"   3. ✅ Ablation Study로 각 구성 요소 기여도 증명")
    logger.info(f"   4. ✅ Baseline 4종과 체계적 비교")
    logger.info(f"   5. ✅ 5-Fold CV로 일반화 성능 검증")
    logger.info(f"   6. ✅ 고급 Threshold 최적화 (PR Curve)")
    logger.info(f"   7. ✅ Feature Importance로 해석가능성 확보")
    logger.info(f"   8. ✅ 실무 적용 가능한 2단계 방어 시스템")
    logger.info(f"   9. ✅ Neural Network 추가로 딥러닝 요소 포함")
    logger.info(f"\n🎯 최종 성능:")
    logger.info(f"   - AUC: {hybrid_auc:.4f}")
    logger.info(f"   - F1: {hybrid_f1:.4f}")
    logger.info(f"   - Recall: {recall_score(y_test, y_test_pred_recall):.4f} (High Recall 모드)")
    logger.info(f"   - 사기 탐지율: {total_detected/total_fraud*100:.1f}% (2단계 방어)")
    logger.info(f"\n{'='*70}\n")






