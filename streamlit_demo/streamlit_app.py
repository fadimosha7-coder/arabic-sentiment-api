import os
import sys
import time
from pathlib import Path
import pandas as pd
import streamlit as st
from huggingface_hub import snapshot_download

# إعداد المسار للوصول لمجلد src
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sentiment_api.model import SentimentModel, clean_text # noqa: E402

MAX_CHARS = 1000
LOW_CONF = 0.8

# إعدادات الصفحة
st.set_page_config(page_title="Arabic Sentiment API", page_icon="💬", layout="centered")

# --- القائمة الجانبية (Sidebar) للمعلومات ---
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

# تحميل الموديل (مخبأ في الذاكرة)
@st.cache_resource(show_spinner="جاري تحميل الموديل... (قد يأخذ دقيقة أول مرة)")
def load_model():
    os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
    repo_id = st.secrets["MODEL_REPO"]
    local_path = snapshot_download(repo_id=repo_id, token=os.environ["HF_TOKEN"])
    return SentimentModel(local_path)

# الواجهة الرئيسية
st.title("💬 تصنيف مشاعر التغريدات العربية")
st.caption("MARBERTv2 - مشروع تعليمي غير تجاري")

# تقسيم الواجهة إلى تبويبات
tab1, tab2 = st.tabs(["📝 نص واحد (Single Text)", "📊 رفع ملف (Batch CSV)"])

# ==========================================
# التبويب الأول: نص واحد
# ==========================================
with tab1:
    text_input = st.text_area("أدخل التغريدة أو النص هنا:", height=150, max_chars=MAX_CHARS, placeholder="اكتب جملة شامية هنا...")
    
    if st.button("تحليل النص", type="primary"):
        text = text_input.strip()
        if not text:
            st.error("الرجاء إدخال نص أولاً.")
        elif not clean_text(text):
            st.warning("النص لا يحتوي على كلمات صالحة.")
        else:
            with st.spinner("جاري التحليل..."):
                svc = load_model()
                start_time = time.time()
                result = svc.predict([text])[0]
                end_time = time.time()
                
                label = result['label']
                conf = result['confidence']
                latency = (end_time - start_time) * 1000
                
                st.markdown("### النتيجة:")
                if label == 'pos':
                    st.success(f"**إيجابي 🟢**")
                    st.metric(label="نسبة الثقة", value=f"{conf:.1%}")
                else:
                    st.error(f"**سلبي 🔴**")
                    st.metric(label="نسبة الثقة", value=f"{conf:.1%}")
                
                if conf < LOW_CONF:
                    st.warning("⚠️ **ثقة منخفضة:** الموديل غير متأكد.")
                st.caption(f"⏱️ زمن التنبؤ: {latency:.0f} ms")

# ==========================================
# التبويب الثاني: رفع ملف CSV
# ==========================================
with tab2:
    st.markdown("ارفع ملف CSV يحتوي على عمود باسم **`Tweet`** (يحتوي على النصوص).")
    uploaded_file = st.file_uploader("اختر ملف CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        if "Tweet" not in df.columns:
            st.error("الملف يجب أن يحتوي على عمود باسم 'Tweet'.")
        else:
            # لأخذ عينة لتجنب انهيار السيرفر المجاني (الحد الأقصى 100 صف)
            if len(df) > 100:
                st.warning(f"الملف يحتوي على {len(df)} صف. سيتم تحليل أول 100 صف فقط لتجنب انهيار السيرفر المجاني.")
                df = df.head(100)
            
            if st.button("بدء تحليل الملف", type="primary"):
                with st.spinner("جاري تحليل التغريدات..."):
                    svc = load_model()
                    texts = df["Tweet"].astype(str).tolist()
                    
                    # إرسال النصوص للموديل
                    results = svc.predict(texts)
                    
                    # استخراج النتائج
                    labels = [r['label'] for r in results]
                    confs = [r['confidence'] for r in results]
                    
                    df['Prediction'] = ['إيجابي' if l == 'pos' else 'سلبي' for l in labels]
                    df['Confidence'] = confs
                    
                    # إحصائيات
                    pos_count = labels.count('pos')
                    neg_count = labels.count('neg')
                    
                    st.success("تم التحليل بنجاح!")
                    
                    # عرض ملخص
                    col1, col2 = st.columns(2)
                    col1.metric("إيجابي 🟢", f"{pos_count} ({pos_count/len(df)*100:.1f}%)")
                    col2.metric("سلبي 🔴", f"{neg_count} ({neg_count/len(df)*100:.1f}%)")
                    
                    st.markdown("### تفاصيل النتائج:")
                    st.dataframe(df[["Tweet", "Prediction", "Confidence"]])
                    
                    # زر لتحميل النتائج
                    csv = df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="تحميل النتائج كملف CSV",
                        data=csv,
                        file_name='sentiment_results.csv',
                        mime='text/csv',
                    )