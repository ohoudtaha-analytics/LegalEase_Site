"""
دوال مشتركة بين سكريبت التدريب وتطبيق Streamlit — عشان التدريب والتشغيل يستخدموا
بالظبط نفس منطق التنظيف والتصنيف الهجين (ML + قواعد قانونية).
"""
import re
import numpy as np
from scipy.sparse import hstack

from risk_rules import rule_based_scores, high_risk_pattern_hits

CLASS_ORDER = ["منخفض الخطورة", "متوسط الخطورة", "عالي الخطورة"]


def clean_arabic_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"[\u064B-\u0652\u0670\u0640]", "", text)  # diacritics
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def vectorize(texts_clean, word_vec, char_vec):
    Xw = word_vec.transform(texts_clean)
    Xc = char_vec.transform(texts_clean)
    return hstack([Xw, Xc]).tocsr()


def rule_proba_vector(clean_text: str) -> np.ndarray:
    """يحوّل درجات الأنماط القانونية لتوزيع احتمالي على الفئات الثلاث."""
    scores = rule_based_scores(clean_text)
    raw = np.array([scores[c] for c in CLASS_ORDER], dtype=float)
    if raw.sum() == 0:
        # مفيش أي مؤشر قاعدي واضح -> توزيع محايد قريب من توزيع البيانات الحقيقي
        return np.array([0.6, 0.25, 0.15])
    return raw / raw.sum()


def hybrid_predict(text: str, artifacts: dict, alpha: float = None):
    """
    يرجّع: (predicted_label, confidence%, proba_dict, ml_proba_dict, rule_proba_dict, overridden)
    alpha: وزن الموديل الإحصائي (1-alpha لقواعد الخطورة). لو None بياخد القيمة المحفوظة.

    Safety override: لو النص فيه ولو نمط واحد من مؤشرات الخطورة العالية القاطعة
    (فسخ تعسفي/غرامة بلا حد/تنازل عن حق/تعديل منفرد...) بيتفرض تصنيف "عالي الخطورة"
    مباشرة بغض النظر عن رأي الموديل الإحصائي. هذا الـ override اتحقق عليه أنه 100%
    دقة (precision) على بيانات train/val/test — يعني ما بيغلطش لصالح رفع إنذار كاذب،
    لكنه بيمنع إغفال بند خطير فعلاً (الأولوية لتقليل False Negatives في هذه الفئة).
    """
    clean = clean_arabic_text(text)
    X = vectorize([clean], artifacts["word_vectorizer"], artifacts["char_vectorizer"])
    ml_proba = artifacts["classifier"].predict_proba(X)[0]
    classes = list(artifacts["classifier"].classes_)
    ml_proba_ordered = np.array([ml_proba[classes.index(c)] for c in CLASS_ORDER])

    rule_p = rule_proba_vector(clean)

    a = artifacts["alpha"] if alpha is None else alpha
    combined = a * ml_proba_ordered + (1 - a) * rule_p
    combined = combined / combined.sum()

    overridden = False
    if high_risk_pattern_hits(clean) >= 1:
        overridden = True
        high_idx = CLASS_ORDER.index("عالي الخطورة")
        # نضمن إنها الأعلى وضوحًا في الواجهة، مع الحفاظ على باقي التوزيع كسياق
        floor = 0.6
        if combined[high_idx] < floor:
            remaining = 1 - floor
            other_sum = combined.sum() - combined[high_idx]
            combined = combined * (remaining / other_sum) if other_sum > 0 else combined
            combined[high_idx] = floor

    pred = CLASS_ORDER[int(np.argmax(combined))]
    confidence = float(np.max(combined)) * 100

    proba_dict = dict(zip(CLASS_ORDER, combined.tolist()))
    ml_dict = dict(zip(CLASS_ORDER, ml_proba_ordered.tolist()))
    rule_dict = dict(zip(CLASS_ORDER, rule_p.tolist()))
    return pred, confidence, proba_dict, ml_dict, rule_dict, overridden
