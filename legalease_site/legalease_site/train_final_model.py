"""
يبني موديل التصنيف النهائي (هجين: TF-IDF كلمات + حروف + Logistic Regression،
ممزوج بكاشف قواعدي لمؤشرات الخطورة القانونية) لموقع Legal Ease.

ليه مش نموذج BERT/embeddings ضخم في النسخة المنشورة؟
  - صفر تحميل نموذج خارجي وقت التشغيل (بدون اعتماد على Hugging Face Hub)
  - يشتغل فورًا على أي استضافة مجانية (Streamlit Cloud) بدون تعليق أو Rate limit
  - حجم النموذج النهائي أقل من 2 ميجابايت — رفع فوري على GitHub

ليه إضافة كاشف قواعدي فوق TF-IDF؟
  - البيانات صغيرة جدًا (330 بند فقط، وفئتا الخطورة العالية/المتوسطة نادرتان)
  - TF-IDF بمفرده بيميل لحفظ عبارات التدريب حرفيًا بدل تعميم المفهوم القانوني
  - الكاشف القواعدي مبني على نفس الـ rubric اللي استخدمه المشروع في التصنيف اليدوي
    (فسخ تعسفي، غرامات غير محددة، تنازل عن الحقوق، تعديل منفرد، إعفاء من المسؤولية)
  - مزج الاثنين (hybrid) بيحسّن استدعاء (recall) الفئات الخطرة من غير ما يضحي بدقة
    الفئة الغالبة (منخفض الخطورة)
"""
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

from model_utils import clean_arabic_text, vectorize, hybrid_predict, CLASS_ORDER

DATA_DIR = "data"
LABEL_MAP_AR_EN = {
    "منخفض الخطورة": {"en": "Low Risk", "color": "#2e7d32", "icon": "🟢"},
    "متوسط الخطورة": {"en": "Medium Risk", "color": "#e08e00", "icon": "🟡"},
    "عالي الخطورة": {"en": "High Risk", "color": "#c62828", "icon": "🔴"},
}


def load_split(name):
    df = pd.read_csv(f"{DATA_DIR}/{name}.csv", encoding="utf-8-sig")
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["text_clean"] = df["text"].apply(clean_arabic_text)
    return df


def build_vectorizers(texts_clean):
    word_vec = TfidfVectorizer(max_features=6000, ngram_range=(1, 2), min_df=1)
    char_vec = TfidfVectorizer(max_features=6000, ngram_range=(2, 4), analyzer="char_wb", min_df=1)
    word_vec.fit(texts_clean)
    char_vec.fit(texts_clean)
    return word_vec, char_vec


def evaluate_hybrid(df, artifacts, alpha):
    preds = []
    for t in df["text"]:
        pred, conf, proba, ml_p, rule_p, overridden = hybrid_predict(t, artifacts, alpha=alpha)
        preds.append(pred)
    return preds


def main():
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")
    train_df = train_df.drop_duplicates(subset=["text_clean"]).reset_index(drop=True)

    print("Shapes:", train_df.shape, val_df.shape, test_df.shape)

    # ---- honest evaluation: fit everything on TRAIN only ----
    word_vec_eval, char_vec_eval = build_vectorizers(train_df["text_clean"])
    Xtr = vectorize(train_df["text_clean"], word_vec_eval, char_vec_eval)
    clf_eval = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42, C=5.0)
    clf_eval.fit(Xtr, train_df["label"])

    eval_artifacts = {
        "word_vectorizer": word_vec_eval,
        "char_vectorizer": char_vec_eval,
        "classifier": clf_eval,
        "alpha": 1.0,  # placeholder, tuned below
    }

    # ---- tune alpha (ML weight vs rule weight) on validation set for best macro F1 ----
    best_alpha, best_f1 = 1.0, -1
    for alpha in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]:
        val_preds = evaluate_hybrid(val_df, eval_artifacts, alpha)
        f1m = f1_score(val_df["label"], val_preds, average="macro")
        print(f"alpha={alpha:.1f} -> val macro-F1={f1m:.3f}")
        if f1m > best_f1:
            best_f1, best_alpha = f1m, alpha

    print(f"\nBest alpha on validation: {best_alpha} (macro-F1={best_f1:.3f})")
    eval_artifacts["alpha"] = best_alpha

    val_preds = evaluate_hybrid(val_df, eval_artifacts, best_alpha)
    test_preds = evaluate_hybrid(test_df, eval_artifacts, best_alpha)

    print("\n=== Validation (hybrid) ===")
    print("Accuracy:", accuracy_score(val_df["label"], val_preds))
    print("F1-macro:", f1_score(val_df["label"], val_preds, average="macro"))
    print(classification_report(val_df["label"], val_preds))

    print("\n=== Test (hybrid) ===")
    print("Accuracy:", accuracy_score(test_df["label"], test_preds))
    print("F1-macro:", f1_score(test_df["label"], test_preds, average="macro"))
    print(classification_report(test_df["label"], test_preds))

    metrics = {
        "val_accuracy": float(accuracy_score(val_df["label"], val_preds)),
        "val_f1_macro": float(f1_score(val_df["label"], val_preds, average="macro")),
        "test_accuracy": float(accuracy_score(test_df["label"], test_preds)),
        "test_f1_macro": float(f1_score(test_df["label"], test_preds, average="macro")),
        "alpha": best_alpha,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "class_distribution_train": train_df["label"].value_counts().to_dict(),
    }

    # ---- FINAL deployed model: retrain on ALL labeled data (train+val+test) ----
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df = full_df.drop_duplicates(subset=["text_clean"]).reset_index(drop=True)

    word_vec_final, char_vec_final = build_vectorizers(full_df["text_clean"])
    Xfull = vectorize(full_df["text_clean"], word_vec_final, char_vec_final)
    clf_final = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42, C=5.0)
    clf_final.fit(Xfull, full_df["label"])

    final_artifacts = {
        "word_vectorizer": word_vec_final,
        "char_vectorizer": char_vec_final,
        "classifier": clf_final,
        "kb_texts": full_df["text"].tolist(),
        "kb_texts_clean": full_df["text_clean"].tolist(),
        "kb_labels": full_df["label"].tolist(),
        "kb_matrix": Xfull,
        "label_map": LABEL_MAP_AR_EN,
        "alpha": best_alpha,
    }

    with open("model_artifacts.pkl", "wb") as f:
        pickle.dump(final_artifacts, f)

    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\nSaved model_artifacts.pkl and metrics.json")

    # ---- quick sanity check on adversarial examples not from the dataset ----
    print("\n=== Sanity check on new unseen clauses ===")
    sanity = [
        ("يحق للطرف الأول فسخ العقد في أي وقت دون إبداء أسباب ودون تعويض الطرف الثاني.", "عالي الخطورة"),
        ("يلتزم الطرف الثاني بدفع غرامة تأخير يحددها الطرف الأول وفقاً لتقديره الخاص دون حد أقصى.", "عالي الخطورة"),
        ("يقر الطرفان بأنهما اطلعا على جميع بنود هذا العقد ووافقا عليها.", "منخفض الخطورة"),
        ("يحتفظ الطرف الأول بحق تعديل شروط هذا العقد في أي وقت دون الرجوع إلى الطرف الثاني.", "عالي الخطورة"),
        ("كل نزاع ينشأ عن هذا العقد يكون من اختصاص محكمة القاهرة الابتدائية.", "متوسط الخطورة"),
    ]
    for text, expected in sanity:
        pred, conf, proba, ml_p, rule_p, overridden = hybrid_predict(text, final_artifacts)
        mark = "✅" if pred == expected else "❌"
        print(f"{mark} expected={expected} got={pred} ({conf:.0f}%) | {text[:60]}")


if __name__ == "__main__":
    main()
