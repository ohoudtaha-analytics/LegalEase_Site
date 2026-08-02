import base64
import json
import os
import shutil
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from model_utils import hybrid_predict, clean_arabic_text, CLASS_ORDER
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Legal Ease | Digilians",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
TESSDATA_DIR = ROOT / "tessdata"

# ----------------------------------------------------------------------------
# Brand palette
# ----------------------------------------------------------------------------
CREAM = "#F1E6CE"
CREAM_LIGHT = "#F8F1E2"
BROWN = "#3D2617"
BROWN_SOFT = "#6B4A34"
TERRACOTTA = "#C97B4A"
GREEN = "#2e7d32"
AMBER = "#e08e00"
RED = "#c62828"

RISK_STYLE = {
    "منخفض الخطورة": {"en": "Low Risk", "color": GREEN, "icon": "🟢"},
    "متوسط الخطورة": {"en": "Medium Risk", "color": AMBER, "icon": "🟡"},
    "عالي الخطورة": {"en": "High Risk", "color": RED, "icon": "🔴"},
}


def img_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


LOGO_B64 = img_to_b64(ASSETS / "logo_crop.png")
QR_B64 = img_to_b64(ASSETS / "qr_code.png")

# ----------------------------------------------------------------------------
# Global CSS
# ----------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Cairo', sans-serif;
    direction: rtl;
}}
.stApp {{ background: linear-gradient(180deg, {CREAM_LIGHT} 0%, {CREAM} 100%); }}

section[data-testid="stSidebar"] {{ background-color: {BROWN}; }}
section[data-testid="stSidebar"] * {{ color: {CREAM_LIGHT} !important; }}

h1, h2, h3 {{ color: {BROWN} !important; font-weight: 800 !important; }}

.hero {{ display:flex; align-items:center; justify-content:center; gap:24px; padding:18px 0 8px 0; text-align:center; }}
.hero img {{ height:110px; }}
.hero-title {{ font-size:2.4rem; font-weight:800; color:{BROWN}; margin:0; }}
.hero-sub {{ font-size:1.05rem; color:{BROWN_SOFT}; margin-top:-6px; }}

.risk-card {{ border-radius:16px; padding:22px 26px; margin-top:14px; border:2px solid; }}
.override-badge {{
    display:inline-block; margin-top:10px; padding:4px 12px; border-radius:8px;
    background:#fff3cd; color:#7a5b00; font-size:0.85rem; border:1px solid #ffe08a;
}}

.metric-card {{
    background:white; border-radius:14px; padding:16px 18px; text-align:center;
    box-shadow:0 2px 10px rgba(61,38,23,0.08); border:1px solid {TERRACOTTA}33;
}}
.metric-card .num {{ font-size:1.9rem; font-weight:800; color:{BROWN}; }}
.metric-card .lbl {{ font-size:0.85rem; color:{BROWN_SOFT}; }}

.evidence-item {{
    background:white; border-radius:10px; padding:10px 14px; margin-bottom:8px;
    border-right:4px solid {TERRACOTTA}; font-size:0.92rem; color:{BROWN};
}}

.stButton>button {{
    background:{TERRACOTTA}; color:white; font-weight:700; border-radius:10px;
    border:none; padding:10px 26px; font-size:1.05rem;
}}
.stButton>button:hover {{ background:{BROWN}; color:{CREAM}; }}

.timeline-step {{
    background:white; border-radius:12px; padding:14px 18px; margin-bottom:10px;
    border-right:5px solid {TERRACOTTA}; box-shadow:0 1px 6px rgba(61,38,23,0.06);
}}
footer, #MainMenu {{visibility:hidden;}}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Load artifacts
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    with open(ROOT / "model_artifacts.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_metrics():
    with open(ROOT / "metrics.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def setup_ocr():
    """يجهّز Tesseract مع حزمة اللغة العربية (نسخة احتياطية مرفقة مع المشروع
    في حال عدم توفر tesseract-ocr-ara عبر packages.txt)."""
    import pytesseract
    try:
        langs = pytesseract.get_languages(config="")
    except Exception:
        langs = []
    if "ara" not in langs and TESSDATA_DIR.exists():
        os.environ["TESSDATA_PREFIX"] = str(TESSDATA_DIR)
    return pytesseract


artifacts = load_artifacts()
metrics = load_metrics()
kb_matrix = artifacts["kb_matrix"]
kb_texts = artifacts["kb_texts"]
kb_labels = artifacts["kb_labels"]


def get_evidence(clean_text: str, k: int = 5):
    from model_utils import vectorize
    X = vectorize([clean_text], artifacts["word_vectorizer"], artifacts["char_vectorizer"])
    sims = cosine_similarity(X, kb_matrix)[0]
    top_idx = sims.argsort()[-k:][::-1]
    return [{"text": kb_texts[i], "label": kb_labels[i], "sim": float(sims[i])}
            for i in top_idx if sims[i] > 0]


def ocr_extract_text(image: Image.Image) -> str:
    pytesseract = setup_ocr()
    gray = ImageOps.grayscale(image)
    # تكبير الصورة لتحسين دقة التعرف على الحروف الصغيرة
    w, h = gray.size
    if max(w, h) < 1600:
        scale = 1600 / max(w, h)
        gray = gray.resize((int(w * scale), int(h * scale)))
    config = "--psm 6"
    if TESSDATA_DIR.exists():
        config += f' --tessdata-dir "{TESSDATA_DIR}"'
    text = pytesseract.image_to_string(gray, lang="ara", config=config)
    return text.strip()


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f'<img src="data:image/png;base64,{LOGO_B64}" style="width:100%; border-radius:12px; margin-bottom:10px;">', unsafe_allow_html=True)
    st.markdown("### ⚖️ Legal Ease")
    st.caption("مساعد ذكي لتحليل خطورة بنود العقود العربية")
    st.markdown("---")
    page = st.radio(
        "التنقل",
        ["🏠 الرئيسية", "🔍 تحليل بند", "📊 لوحة الأداء", "ℹ️ عن المشروع"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**📱 مسح الكود للمشاركة**")
    st.markdown(f'<img src="data:image/png;base64,{QR_B64}" style="width:100%; border-radius:10px;">', unsafe_allow_html=True)
    st.caption("تجهيز رابط النشر النهائي وتحديث الكود من `assets/qr_code.png`")

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
    <img src="data:image/png;base64,{LOGO_B64}">
    <div>
        <p class="hero-title">Legal Ease</p>
        <p class="hero-sub">تحليل ذكي لخطورة بنود العقود العربية — Digilians</p>
    </div>
</div>
""", unsafe_allow_html=True)


def render_prediction(pred, confidence, proba, overridden, evidence):
    style = RISK_STYLE[pred]
    st.markdown(f"""
    <div class="risk-card" style="background:{style['color']}18; border-color:{style['color']};">
        <span style="font-size:2rem">{style['icon']}</span>
        <span style="font-size:1.5rem; font-weight:800; color:{style['color']}; margin-right:10px;">
            {pred} ({style['en']})
        </span>
        <div style="margin-top:8px; color:{BROWN};">مستوى الثقة: <b>{confidence:.1f}%</b></div>
        {"<div class='override-badge'>🚩 تم رصد مؤشر خطورة قانوني قاطع في النص (فسخ تعسفي / غرامة غير محددة / تنازل عن حق...)، لذلك تم رفع التصنيف تلقائيًا لضمان عدم إغفال بند خطير.</div>" if overridden else ""}
    </div>
    """, unsafe_allow_html=True)

    pc1, pc2, pc3 = st.columns(3)
    for col, label in zip([pc1, pc2, pc3], CLASS_ORDER):
        p = proba.get(label, 0) * 100
        s = RISK_STYLE[label]
        col.markdown(f"""<div class="metric-card">
        <div style="font-size:1.3rem">{s['icon']}</div>
        <div class="num" style="color:{s['color']}">{p:.0f}%</div>
        <div class="lbl">{label}</div></div>""", unsafe_allow_html=True)

    st.markdown("#### 📚 بنود مشابهة من الأرشيف (دليل)")
    if evidence:
        for ev in evidence:
            es = RISK_STYLE[ev["label"]]
            st.markdown(f"""<div class="evidence-item">
            {es['icon']} <b>{ev['label']}</b> · تشابه {ev['sim']*100:.0f}%<br>
            {ev['text'][:220]}{'…' if len(ev['text']) > 220 else ''}
            </div>""", unsafe_allow_html=True)
    else:
        st.caption("لا توجد بنود مشابهة كفاية في الأرشيف الحالي.")

    st.caption("⚠️ هذا تحليل آلي بغرض المساعدة الأولية، ولا يغني عن استشارة قانونية متخصصة.")


# ============================================================================
# PAGE: HOME
# ============================================================================
if page == "🏠 الرئيسية":
    st.markdown("""
    <div style="text-align:center; max-width:750px; margin:0 auto 20px auto;">
    <p style="font-size:1.1rem;">
    منصة ذكاء اصطناعي تقرأ بنود العقود العربية (بيع، إيجار، عمل، شراكة) وتقيّم مستوى الخطورة
    لكل طرف — للمساعدة في اكتشاف البنود التي تحتاج انتباهًا قبل التوقيع.
    </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("📄", "50+", "عقد حقيقي"),
        ("📑", "376", "بند مُستخرج"),
        ("✅", str(metrics["n_train"] + metrics["n_val"] + metrics["n_test"]), "بند مُصنّف"),
        ("🎯", f'{metrics["test_accuracy"]*100:.0f}%', "دقة على بيانات غير مرئية"),
    ]
    for col, (icon, num, lbl) in zip([c1, c2, c3, c4], stats):
        col.markdown(f"""<div class="metric-card"><div style="font-size:1.6rem">{icon}</div>
        <div class="num">{num}</div><div class="lbl">{lbl}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🚀 إمكانيات المنصة")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""<div class="timeline-step">
        <b>🔍 تحليل بند</b><br>كتابة بند، اختياره من أمثلة جاهزة، أو رفع صورة عقد — والنتيجة فورية
        مع نسبة الثقة والبنود المشابهة كدليل.
        </div>""", unsafe_allow_html=True)
    with f2:
        st.markdown("""<div class="timeline-step">
        <b>📊 لوحة الأداء</b><br>عرض دقة الموديل، توزيع الفئات، وتفاصيل التقييم بشكل تفاعلي.
        </div>""", unsafe_allow_html=True)
    with f3:
        st.markdown("""<div class="timeline-step">
        <b>ℹ️ عن المشروع</b><br>منهجية جمع البيانات والتصنيف والتدريب خطوة بخطوة.
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 من القائمة الجانبية، اختيار **🔍 تحليل بند** لتجربة الموديل الآن.")

# ============================================================================
# PAGE: CLASSIFY
# ============================================================================
elif page == "🔍 تحليل بند":
    st.markdown("### 🔍 تحليل خطورة بند العقد")
    st.caption("التحليل يتم فورًا وبالكامل داخل هذا التطبيق، بدون أي اتصال خارجي.")

    input_mode = st.radio(
        "طريقة إدخال البند",
        ["✍️ كتابة نص", "📋 اختيار مثال جاهز", "🖼️ رفع صورة عقد"],
        horizontal=True,
    )

    clause_text = ""

    if input_mode == "✍️ كتابة نص":
        clause_text = st.text_area(
            "نص البند",
            height=140,
            placeholder="مثال: يلتزم الطرف الثاني بسداد كامل المبلغ خلال ٣٠ يومًا من تاريخ التوقيع...",
        )

    elif input_mode == "📋 اختيار مثال جاهز":
        example_clauses = {
            "— اختيار مثال —": "",
            "مثال: بند فسخ تعسفي": "يحق للطرف الأول فسخ العقد في أي وقت ودون إبداء أسباب، مع إخطار الطرف الثاني بمدة يوم واحد فقط ودون أي تعويض.",
            "مثال: بند عام": "يقر الطرفان بأنهما اطلعا على جميع بنود هذا العقد وأنهما موافقان عليها بكامل الإرادة والاختيار.",
            "مثال: بند غرامة غير محددة": "في حالة التأخير في السداد، يلتزم الطرف الثاني بدفع غرامة يحددها الطرف الأول وفقًا لتقديره الخاص.",
            "مثال: بند اختصاص قضائي": "كل نزاع ينشأ عن هذا العقد يكون من اختصاص محكمة القاهرة الابتدائية.",
        }
        choice = st.selectbox("الأمثلة الجاهزة", list(example_clauses.keys()), label_visibility="collapsed")
        clause_text = example_clauses[choice]
        if clause_text:
            st.text_area("نص البند المختار", value=clause_text, height=100, disabled=True)

    elif input_mode == "🖼️ رفع صورة عقد":
        uploaded = st.file_uploader("رفع صورة للبند أو العقد (JPG / PNG)", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")
            col_img, col_txt = st.columns([1, 1.4])
            with col_img:
                st.image(image, caption="الصورة المرفوعة", use_container_width=True)
            with st.spinner("جاري استخراج النص من الصورة (OCR)..."):
                try:
                    extracted = ocr_extract_text(image)
                except Exception as e:
                    extracted = ""
                    st.error(f"تعذّر تشغيل OCR: {e}")
            with col_txt:
                clause_text = st.text_area(
                    "النص المستخرج (قابل للتعديل قبل التحليل)",
                    value=extracted, height=200,
                )
            if extracted:
                st.caption("يُنصح بمراجعة النص المستخرج وتصحيح أي كلمات غير واضحة قبل الضغط على تحليل.")

    k = st.slider("عدد البنود المشابهة المعروضة كدليل", 3, 10, 5)

    if st.button("⚖️ تحليل البند", use_container_width=False):
        if not clause_text or not clause_text.strip():
            st.warning("يرجى إدخال نص البند أولاً (كتابة، اختيار مثال، أو رفع صورة).")
        else:
            with st.spinner("جاري التحليل..."):
                pred, confidence, proba, ml_p, rule_p, overridden = hybrid_predict(clause_text, artifacts)
                clean = clean_arabic_text(clause_text)
                evidence = get_evidence(clean, k=k)
            render_prediction(pred, confidence, proba, overridden, evidence)

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================
elif page == "📊 لوحة الأداء":
    st.markdown("### 📊 أداء الموديل")

    d1, d2, d3, d4 = st.columns(4)
    d1.markdown(f"""<div class="metric-card"><div class="num">{metrics['val_accuracy']*100:.1f}%</div><div class="lbl">دقة (Validation)</div></div>""", unsafe_allow_html=True)
    d2.markdown(f"""<div class="metric-card"><div class="num">{metrics['test_accuracy']*100:.1f}%</div><div class="lbl">دقة (Test)</div></div>""", unsafe_allow_html=True)
    d3.markdown(f"""<div class="metric-card"><div class="num">{metrics['val_f1_macro']:.2f}</div><div class="lbl">F1-macro (Val)</div></div>""", unsafe_allow_html=True)
    d4.markdown(f"""<div class="metric-card"><div class="num">{metrics['test_f1_macro']:.2f}</div><div class="lbl">F1-macro (Test)</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### توزيع فئات الخطورة في بيانات التدريب")
    dist = metrics["class_distribution_train"]
    df_dist = pd.DataFrame({"الفئة": list(dist.keys()), "العدد": list(dist.values())})
    st.bar_chart(df_dist.set_index("الفئة"))

    st.markdown("#### بيانات تجريبية من مجموعة الاختبار")
    test_df = pd.read_csv(ROOT / "data" / "test.csv", encoding="utf-8-sig")
    st.dataframe(test_df[["text", "label"]].sample(min(10, len(test_df)), random_state=1),
                 use_container_width=True, hide_index=True)

# ============================================================================
# PAGE: ABOUT
# ============================================================================
elif page == "ℹ️ عن المشروع":
    st.markdown("### ℹ️ عن مشروع Legal Ease")
    st.markdown("""
    منصة **Legal Ease** تستخدم الذكاء الاصطناعي لتحليل بنود العقود العربية (بيع عقارات، إيجار،
    عمل، شراكة) وتصنيف مستوى الخطورة (منخفض / متوسط / عالي) بناءً على معايير واضحة: عدم توازن
    الالتزامات، الغرامات غير المحددة، الغموض، والفسخ التعسفي.
    """)

    st.markdown("#### 🛠️ رحلة المشروع")
    steps = [
        ("📥", "جمع البيانات", "أكتر من 50 عقد عربي حقيقي (مصري، سعودي، مغربي) بأنواع مختلفة."),
        ("🔤", "الرقمنة", "OCR للمطبوع + نقل يدوي للعقود المكتوبة بخط اليد."),
        ("✂️", "تقسيم البنود", "استخراج 376 بند عبر تعرف تلقائي على عناوين المواد."),
        ("🏷️", "التصنيف", "تصنيف 359 بند بمساعدة AI ومراجعة بشرية، لكل طرف على حدة."),
        ("⚖️", "معالجة عدم التوازن", "بيانات اصطناعية للفئة النادرة + موازين فئوية أثناء التدريب."),
        ("🤖", "التدريب والمقارنة", "TF-IDF+SVM، AraBERT، CAMeLBERT، Sentence-Embeddings — واختيار الأنسب للنشر."),
        ("🩹", "تحسين الموديل المنشور", "دمج TF-IDF (كلمات + حروف) مع كاشف قواعدي لمؤشرات الخطورة القانونية، لتحسين رصد الحالات النادرة."),
        ("🌐", "النشر", "موقع Streamlit دائم بهوية بصرية موحدة، بدون اعتماد على تحميل نماذج ضخمة."),
    ]
    for icon, title, desc in steps:
        st.markdown(f"""<div class="timeline-step">
        <span style="font-size:1.3rem">{icon}</span>
        <b style="margin-right:8px;">{title}</b><br>{desc}</div>""", unsafe_allow_html=True)

    st.markdown("#### 🧠 لماذا موديل هجين (TF-IDF + قواعد قانونية) بدل BERT ضخم؟")
    st.markdown(f"""
    - **الموثوقية أهم من فارق دقة صغير**: هذا الموديل حقق دقة **{metrics['test_accuracy']*100:.0f}%**
      على بيانات اختبار غير مرئية، قريبة من نتائج CAMeLBERT، لكن بدون أي تحميل نموذج خارجي وقت التشغيل.
    - **حجم الموديل النهائي أقل من 2 ميجابايت** بدل مئات الميجابايت — رفع فوري على GitHub.
    - **مشكلة بيانات صغيرة**: 330 بند فقط، وفئتا الخطورة العالية/المتوسطة نادرتان. موديل TF-IDF
      وحده كان أحيانًا يخطئ في بنود واضحة الخطورة بصياغة مختلفة عن أمثلة التدريب.
    - **الحل**: كاشف قواعدي مبني على نفس الـ rubric المستخدم في التصنيف اليدوي (فسخ تعسفي، غرامة
      بلا حد، تنازل عن حق، تعديل منفرد، إعفاء من المسؤولية). أي بند يحتوي مؤشرًا قاطعًا من هذه
      المؤشرات يُصنَّف "عالي الخطورة" تلقائيًا — وهذا التحقق أثبت دقة 100% على كل بيانات
      التدريب/التحقق/الاختبار (لا ينتج إنذارات كاذبة)، وبيمنع إغفال بنود خطيرة فعلاً.
    """)

    st.warning("⚠️ هذا الموقع أداة مساعدة أولية بالذكاء الاصطناعي، ونتائجه لا تغني عن استشارة محامٍ مختص.")
