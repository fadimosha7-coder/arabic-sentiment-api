import os
import sys
import time
import re
from pathlib import Path
import pandas as pd
import streamlit as st
from huggingface_hub import snapshot_download
from deep_translator import GoogleTranslator

# إعداد المسار للوصول لمجلد src
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sentiment_api.model import SentimentModel, clean_text # noqa: E402

MAX_CHARS = 1000
LOW_CONF = 0.8

# إعدادات الصفحة
st.set_page_config(page_title="Arabic Sentiment API", page_icon="💬", layout="centered")

# --- إضافة CSS لجعل الواجهة عربية (RTL) ---
st.markdown("""
<style>
    /* جعل الواجهة من اليمين لليسار */
    .stApp { direction: rtl; }
    /* جعل مربع النص من اليمين لليسار */
    textarea { direction: rtl !important; text-align: right !important; }
    /* ضبط محاذاة العناوين والفقرات */
    h1, h2, h3, p, li { text-align: right; }
    /* تعديل شكل شريط التقدم */
    .stProgress > div > div > div > div { background-color: #28a745; } /* لون أخضر للنتيجة الإيجابية */
    .stProgress.negative > div > div > div > div { background-color: #dc3545; } /* لون أحمر للنتيجة السلبية */
</style>
""", unsafe_allow_html=True)

# --- القائمة الجانبية (Sidebar) ---
with st.sidebar:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=50)
    st.title("عن المشروع")
    st.info("مشروع تعليمي لتحليل المشاعر في التغريدات العربية (الشامية) باستخدام MARBERTv2.")
    st.markdown("### ⚠️ حدود الموديل")
    st.markdown("""
    - لا يعرف الفئة **المحايدة**.
    - لا يفهم **السخرية**.
    - **لا تكتب أي معلومات شخصية.**
    """)
    st.markdown("---")
    st.markdown("🔗 **الروابط:** [GitHub](https://github.com/fadimosha7-coder/arabic-sentiment-api)")

# تحميل الموديل
@st.cache_resource(show_spinner="جاري تحميل الموديل... (قد يأخذ دقيقة أول مرة)")
def load_model():
    os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
    repo_id = st.secrets["MODEL_REPO"]
    local_path = snapshot_download(repo_id=repo_id, token=os.environ["HF_TOKEN"])
    return SentimentModel(local_path)

# دالة للتحقق إذا كان النص إنجليزي وترجمته
def translate_if_english(text):
    # إذا كان النص يحتوي على حروف عربية، اتركه كما هو
    if re.search(r'[\u0600-\u06FF]', text):
        return text, False
    try:
        # ترجمة النص الإنجليزي إلى العربية
        translated = GoogleTranslator(source='auto', target='ar').translate(text)
        return translated, True
    except:
        return text, False

# الواجهة الرئيسية
st.markdown("<h1 style='text-align: center;'>💬 تصنيف مشاعر التغريدات العربية</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>MARBERTv2 - مشروع تعليمي غير تجاري</p>", unsafe_allow_html=True)

# التبويبات
tab1, tab2 = st.tabs(["📝 نص واحد (Single Text)", "📊 رفع ملف (Batch CSV)"])

# ==========================================
# التبويب الأول: نص واحد
# ==========================================
with tab1:
    text_input = st.text_area("أدخل التغريدة أو النص هنا (عربي أو إنجليزي):", height=150, max_chars=MAX_CHARS, placeholder="اكتب جملة هنا...")
    
    if st.button("تحليل النص", type="primary"):
        text = text_input.strip()
        if not text:
            st.error("الرجاء إدخال نص أولاً.")
        else:
            # التحقق من الترجمة
            processed_text, is_translated = translate_if_english(text)
            if is_translated:
                st.info(f"تم ترجمة النص الإنجليزي إلى: **{processed_text}**")
            
            with st.spinner("جاري التحليل..."):
                svc = load_model()
                start_time = time.time()
                result = svc.predict([processed_text])[0]
                end_time = time.time()
                
                label = result['label']
                conf = result['confidence']
                latency = (end_time - start_time) * 1000
                
                st.markdown("### النتيجة:")
                
                # شريط تقدم مخصص
                if label == 'pos':
                    st.markdown(f"""
                    <div style="background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; padding: 10px; text-align: center; margin-bottom: 10px;">
                        <h3 style="color: #155724; margin: 0;">إيجابي 🟢 (ثقة: {conf:.1%})</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(conf)
                else:
                    st.markdown(f"""
                    <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; padding: 10px; text-align: center; margin-bottom: 10px;">
                        <h3 style="color: #721c24; margin: 0;">سلبي 🔴 (ثقة: {conf:.1%})</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(conf)
                
                if conf < LOW_CONF:
                    st.warning("⚠️ **ثقة منخفضة:** الموديل غير متأكد من النتيجة. قد تكون الجملة سخرية أو كلمات متضاربة.")
                st.caption(f"⏱️ زمن التنبؤ: {latency:.0f} ms")

# ==========================================
# التبويب الثاني: رفع ملف CSV (نفس الكود السابق مع بعض التعديلات البسيطة)
# ==========================================
with tab2:
    st.markdown("ارفع ملف CSV يحتوي على عمود باسم **`Tweet`** (عربي فقط).")
    uploaded_file = st.file_uploader("اختر ملف CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        if "Tweet" not in df.columns:
            st.error("الملف يجب أن يحتوي على عمود باسم 'Tweet'.")
        else:
            if len(df) > 100:
                st.warning(f"الملف يحتوي على {len(df)} صف. سيتم تحليل أول 100 صف فقط.")
                df = df.head(100)
            
            if st.button("بدء تحليل الملف", type="primary"):
                with st.spinner("جاري تحليل التغريدات..."):
                    svc = load_model()
                    texts = df["Tweet"].astype(str).tolist()
                    results = svc.predict(texts)
                    
                    labels = [r['label'] for r in results]
                    confs = [r['confidence'] for r in results]
                    
                    df['Prediction'] = ['إيجابي' if l == 'pos' else 'سلبي' for l in labels]
                    df['Confidence'] = confs
                    
                    pos_count = labels.count('pos')
                    neg_count = labels.count('neg')
                    
                    st.success("تم التحليل بنجاح!")
                    col1, col2 = st.columns(2)
                    col1.metric("إيجابي 🟢", f"{pos_count} ({pos_count/len(df)*100:.1f}%)")
                    col2.metric("سلبي 🔴", f"{neg_count} ({neg_count/len(df)*100:.1f}%)")
                    
                    st.markdown("### تفاصيل النتائج:")
                    st.dataframe(df[["Tweet", "Prediction", "Confidence"]])
                    
                    csv = df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="تحميل النتائج كملف CSV",
                        data=csv,
                        file_name='sentiment_results.csv',
                        mime='text/csv',
                    )