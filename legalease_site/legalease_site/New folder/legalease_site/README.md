# ⚖️ Legal Ease — Digilians

موقع تفاعلي يحلّل خطورة بنود العقود العربية بالذكاء الاصطناعي — عن طريق كتابة نص، اختيار
مثال جاهز، أو رفع صورة عقد (مع استخراج النص تلقائيًا عبر OCR) — بنفس الهوية البصرية
لملصقات المشروع.

## بنية المشروع
```
legalease_site/
├── app.py                  # تطبيق Streamlit الرئيسي (كل الصفحات)
├── model_utils.py           # منطق التصنيف الهجين (TF-IDF + قواعد قانونية) — مشترك بين التدريب والتطبيق
├── risk_rules.py             # قواعد كشف مؤشرات الخطورة القانونية (rubric المشروع)
├── train_final_model.py     # سكريبت تدريب الموديل النهائي
├── model_artifacts.pkl       # الموديل المدرَّب + قاعدة البنود للمقارنة (~2.6MB)
├── metrics.json              # نتائج التقييم (val/test)
├── requirements.txt          # مكتبات Python
├── packages.txt              # حزم النظام (tesseract-ocr للتعرف الضوئي على الصور)
├── assets/
│   ├── logo.png              # الملصق الأصلي (5 نسخ)
│   ├── logo_crop.png          # الشعار المفرد المستخدم في الموقع
│   └── qr_code.png            # كود QR — استبداله بالرابط الحقيقي بعد النشر
├── tessdata/
│   └── ara.traineddata        # حزمة اللغة العربية لـ Tesseract (نسخة احتياطية)
└── data/                     # train.csv / val.csv / test.csv / clauses_manual_verified.csv
```

## التشغيل محليًا
```bash
pip install -r requirements.txt
# لو Tesseract مش مثبّت على الجهاز (لتجربة رفع الصور محليًا):
#   Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-ara
#   macOS: brew install tesseract tesseract-lang
streamlit run app.py
```

## النشر الدائم (مجاني) على Streamlit Community Cloud
1. رفع هذا الفولدر بالكامل (محتوياته، وليس الفولدر نفسه) إلى **جذر** مستودع GitHub جديد.
   المطلوب أن يظهر `app.py` مباشرة عند فتح المستودع، وليس داخل فولدر فرعي.
2. الدخول على https://share.streamlit.io وتسجيل الدخول بحساب GitHub.
3. اختيار "New app" → اختيار المستودع → الفرع main → الملف الرئيسي `app.py` → Deploy.
4. Streamlit Cloud بيقرأ `packages.txt` تلقائيًا ويثبّت `tesseract-ocr` و`tesseract-ocr-ara`
   على السيرفر (خطوة ضرورية لتشغيل ميزة رفع صور العقود).
5. الحصول على رابط دائم زي: `https://your-app-name.streamlit.app`

### تحديث كود QR بالرابط الحقيقي
بعد ما ياخد الموقع رابطه النهائي:
```bash
python3 -c "
import qrcode
qr = qrcode.QRCode(box_size=10, border=2, error_correction=qrcode.constants.ERROR_CORRECT_H)
qr.add_data('الرابط_الحقيقي_هنا')
qr.make(fit=True)
img = qr.make_image(fill_color='#3D2617', back_color='#F1E6CE').convert('RGB')
img.save('assets/qr_code.png')
"
```
ثم رفع التحديث لنفس المستودع — الموقع هيتحدث تلقائيًا.

> في حال وجود صورة الـ QR الأصلية من واجهة Gradio، يمكن حفظها باسم
> `assets/qr_code.png` بدل الصورة الحالية مباشرة.

## إعادة تدريب الموديل
```bash
python3 train_final_model.py
```
بيعيد بناء `model_artifacts.pkl` و`metrics.json`، وبيطبع تقرير تقييم على val/test بالإضافة
لفحص سريع (sanity check) على أمثلة جديدة غير موجودة في البيانات.

## لماذا موديل هجين (TF-IDF + قواعد قانونية) بدل BERT/Embeddings ضخم؟
- **صفر تحميل خارجي وقت التشغيل** (مفيش اعتماد على Hugging Face Hub، فمفيش خطر تعليق أو Rate limit).
- **حجم الموديل ~2.6MB** بدل مئات الميجابايت — رفع فوري على GitHub بدون Git LFS.
- **مشكلة بيانات صغيرة (330 بند)**: TF-IDF وحده أحيانًا كان يخطئ في بنود واضحة الخطورة
  بصياغة مختلفة عن أمثلة التدريب (مشكلة تعميم كلاسيكية مع البيانات القليلة).
- **الحل**: كاشف قواعدي (`risk_rules.py`) مبني على نفس الـ rubric المستخدم في التصنيف
  اليدوي للمشروع (فسخ تعسفي، غرامة بلا حد، تنازل عن حق، تعديل منفرد، إعفاء من المسؤولية).
  أي بند فيه مؤشر قاطع من دول بيتصنّف "عالي الخطورة" تلقائيًا (safety override) — تم
  التحقق من أن هذا الـ override دقته 100% (precision) على كل بيانات train/val/test،
  يعني بيحسّن رصد الحالات الخطرة من غير ما يرفع إنذارات كاذبة.
