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
from sentiment_api.model import SentimentModel, clean_text  # noqa: E402,F401

MAX_CHARS = 1000
LOW_CONF = 0.8
GREEN = "#16a34a"
RED = "#dc2626"
ORANGE = "#f97316"

st.set_page_config(page_title="Arabic Sentiment API", page_icon="💬", layout="centered")

# ------------------------------------------------------------------
# النصوص بالعربي والإنجليزي
# ------------------------------------------------------------------
T = {
    "ar": {
        "title": "💬 تصنيف مشاعر التغريدات العربية",
        "subtitle": "MARBERTv2 - مشروع تعليمي غير تجاري",
        "about": "عن المشروع",
        "about_text": "مشروع تعليمي لتحليل المشاعر في التغريدات العربية (الشامية) باستخدام MARBERTv2.",
        "limits": "⚠️ حدود الموديل",
        "limits_list": [
            "يصنّف النص إلى فئتين فقط (إيجابي / سلبي)، ولا يدعم الفئة **المحايدة**.",
            "قد تنخفض دقته مع **السخرية** والعبارات ذات المشاعر المتضاربة.",
            "**خصوصية بياناتك:** لا يقوم التطبيق بتخزين النصوص المُدخلة أو مشاركتها، وتُعالَج لحظياً لإنتاج النتيجة فقط. (النصوص الإنجليزية تُرسَل إلى خدمة ترجمة خارجية قبل التحليل.)",
        ],
        "links": "🔗 **الروابط:**",
        "loading": "جاري تحميل الموديل... (قد يأخذ دقيقة أول مرة)",
        "tab1": "📝 نص واحد (Single Text)",
        "tab2": "📊 رفع ملف (Batch CSV)",
        "input_label": "أدخل التغريدة أو النص هنا (عربي أو إنجليزي):",
        "placeholder": "اكتب جملة هنا...",
        "analyze": "تحليل النص (Analyze)",
        "clear": "مسح (Clear)",
        "empty": "الرجاء إدخال نص أولاً.",
        "translated": "تم ترجمة النص الإنجليزي إلى:",
        "analyzing": "جاري التحليل...",
        "result": "النتيجة:",
        "pos": "إيجابي",
        "neg": "سلبي",
        "conf": "ثقة",
        "low_conf": "⚠️ **ثقة منخفضة:** الموديل غير متأكد من النتيجة. قد تكون الجملة سخرية أو كلمات متضاربة.",
        "latency": "⏱️ زمن التنبؤ",
        "csv_hint": "ارفع ملف CSV يحتوي على عمود باسم **`Tweet`** (عربي فقط).",
        "upload": "اختر ملف CSV",
        "no_col": "الملف يجب أن يحتوي على عمود باسم 'Tweet'.",
        "too_many": "الملف يحتوي على {n} صف. سيتم تحليل أول 100 صف فقط.",
        "start_batch": "بدء تحليل الملف (Start)",
        "batch_spinner": "جاري تحليل التغريدات...",
        "batch_ok": "تم التحليل بنجاح!",
        "details": "تفاصيل النتائج:",
        "download": "تحميل النتائج كملف CSV (Download)",
        "lang_btn": "🌐 English",
    },
    "en": {
        "title": "💬 Arabic Tweet Sentiment Classifier",
        "subtitle": "MARBERTv2 - Educational, non-commercial project",
        "about": "About the project",
        "about_text": "An educational project for sentiment analysis of Arabic (Levantine) tweets using MARBERTv2.",
        "limits": "⚠️ Model limitations",
        "limits_list": [
            "It classifies text into two classes only (positive / negative) and does not support the **neutral** class.",
            "Accuracy may drop with **sarcasm** and sentences containing mixed sentiment.",
            "**Your privacy:** the app does not store or share the text you enter; it is processed instantly to produce the result only. (English text is sent to an external translation service before analysis.)",
        ],
        "links": "🔗 **Links:**",
        "loading": "Loading the model... (may take a minute the first time)",
        "tab1": "📝 Single Text (نص واحد)",
        "tab2": "📊 Batch CSV (رفع ملف)",
        "input_label": "Enter the tweet or text here (Arabic or English):",
        "placeholder": "Type a sentence here...",
        "analyze": "Analyze (تحليل)",
        "clear": "Clear (مسح)",
        "empty": "Please enter some text first.",
        "translated": "English text was translated to:",
        "analyzing": "Analyzing...",
        "result": "Result:",
        "pos": "Positive",
        "neg": "Negative",
        "conf": "Confidence",
        "low_conf": "⚠️ **Low confidence:** the model is unsure. The sentence may be sarcastic or contain mixed words.",
        "latency": "⏱️ Inference time",
        "csv_hint": "Upload a CSV file with a column named **`Tweet`** (Arabic only).",
        "upload": "Choose a CSV file",
        "no_col": "The file must contain a column named 'Tweet'.",
        "too_many": "The file has {n} rows. Only the first 100 will be analyzed.",
        "start_batch": "Start analysis (بدء)",
        "batch_spinner": "Analyzing tweets...",
        "batch_ok": "Analysis completed!",
        "details": "Result details:",
        "download": "Download results as CSV (تحميل)",
        "lang_btn": "🌐 العربية",
    },
}

if "lang" not in st.session_state:
    st.session_state["lang"] = "ar"

lang = st.session_state["lang"]
t = T[lang]
is_rtl = lang == "ar"
direction = "rtl" if is_rtl else "ltr"
align = "right" if is_rtl else "left"

# ------------------------------------------------------------------
# CSS
# ------------------------------------------------------------------
st.markdown(
    f"""
<style>
    /* الاتجاه على المحتوى فقط (مو على .stApp) عشان القائمة الجانبية ما تخرب عل موبايل */
    section.main, [data-testid="stMain"], [data-testid="stSidebarContent"] {{ direction: {direction}; }}
    .main-title {{ text-align:center; font-size: clamp(1.4rem, 6vw, 2.4rem); line-height:1.4; margin: 0.2rem 0; font-weight:700; }}
    @media (max-width: 640px) {{
        .stButton > button {{ font-size: 1rem; }}
        .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
    }}
    textarea {{ direction: {direction} !important; text-align: {align} !important; }}
    h2, h3, p, li, label {{ text-align: {align}; }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(90deg, #f97316, #fbbf24);
        border: none; color: white; font-weight: 700;
    }}
    .bar-row {{ margin: 8px 0; direction: {direction}; }}
    .bar-label {{ display:flex; justify-content:space-between; font-weight:600; margin-bottom:4px; }}
    .bar-bg {{ background: rgba(128,128,128,0.25); border-radius: 8px; height: 14px; overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: 8px; }}
</style>
""",
    unsafe_allow_html=True,
)


def bar(label: str, p: float, color: str) -> str:
    return f"""
    <div class="bar-row">
      <div class="bar-label"><span>{label}</span><span>{p:.1%}</span></div>
      <div class="bar-bg"><div class="bar-fill" style="width:{p*100:.1f}%; background:{color};"></div></div>
    </div>
    """


# ------------------------------------------------------------------
# القائمة الجانبية
# ------------------------------------------------------------------
with st.sidebar:
    if st.button(t["lang_btn"], use_container_width=True):
        st.session_state["lang"] = "en" if lang == "ar" else "ar"
        st.rerun()
    st.markdown(
        """
        <div style="text-align:center; margin-top:10px;">
            <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="60">
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<h2 style='text-align:center; margin-top:8px;'>{t['about']}</h2>",
        unsafe_allow_html=True,
    )
    st.info(t["about_text"])
    st.markdown(f"### {t['limits']}")
    st.markdown("\n".join(f"- {x}" for x in t["limits_list"]))
    st.markdown("---")
    st.markdown(f"{t['links']} [GitHub](https://github.com/fadimosha7-coder/arabic-sentiment-api)")


# ------------------------------------------------------------------
# الموديل
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
    repo_id = st.secrets["MODEL_REPO"]
    local_path = snapshot_download(repo_id=repo_id, token=os.environ["HF_TOKEN"])
    return SentimentModel(local_path)


def translate_if_english(text):
    if re.search(r"[\u0600-\u06FF]", text):
        return text, False
    try:
        translated = GoogleTranslator(source="auto", target="ar").translate(text)
        return translated, True
    except Exception:
        return text, False


def clear_text():
    st.session_state["text_input"] = ""
    st.session_state.pop("single_result", None)


# ------------------------------------------------------------------
# الواجهة الرئيسية
# ------------------------------------------------------------------
st.markdown(f"<div class='main-title'>{t['title']}</div>", unsafe_allow_html=True)
st.markdown(
    f"<p style='text-align: center; color: gray;'>{t['subtitle']}</p>",
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs([t["tab1"], t["tab2"]])

# ==========================================
# التبويب الأول: نص واحد
# ==========================================
with tab1:
    text_input = st.text_area(
        t["input_label"],
        height=150,
        max_chars=MAX_CHARS,
        placeholder=t["placeholder"],
        key="text_input",
    )

    c1, c2 = st.columns([3, 1])
    analyze_clicked = c1.button(t["analyze"], type="primary", use_container_width=True)
    c2.button(t["clear"], on_click=clear_text, use_container_width=True)

    if analyze_clicked:
        text = text_input.strip()
        if not text:
            st.error(t["empty"])
        else:
            processed_text, is_translated = translate_if_english(text)
            with st.spinner(t["analyzing"]):
                svc = load_model()
                start_time = time.time()
                result = svc.predict([processed_text])[0]
                latency = (time.time() - start_time) * 1000
            st.session_state["single_result"] = {
                "processed": processed_text,
                "translated": is_translated,
                "label": result["label"],
                "conf": result["confidence"],
                "latency": latency,
            }

    # عرض النتيجة (تبقى موجودة حتى لو غيّرت اللغة)
    res = st.session_state.get("single_result")
    if res:
        if res["translated"]:
            st.info(f"{t['translated']} **{res['processed']}**")

        label, conf = res["label"], res["conf"]
        is_pos = label == "pos"
        p_pos = conf if is_pos else 1 - conf
        p_neg = 1 - p_pos

        st.markdown(f"### {t['result']}")
        main_color = GREEN if is_pos else RED
        main_text = t["pos"] if is_pos else t["neg"]
        st.markdown(
            f"""
            <div style="background:#ffffff; border:1px solid #e5e7eb; border-radius:12px;
                        padding:14px; text-align:center; margin-bottom:12px;">
                <h2 style="color:{main_color}; margin:0; text-align:center;">{main_text}</h2>
                <div style="color:gray;">{t['conf']}: {conf:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # الشريطين: سلبي + إيجابي (الأعلى أولاً)
        rows = [(t["neg"], p_neg, RED), (t["pos"], p_pos, GREEN)]
        rows.sort(key=lambda r: r[1], reverse=True)
        st.markdown("".join(bar(n, p, c) for n, p, c in rows), unsafe_allow_html=True)

        if conf < LOW_CONF:
            st.warning(t["low_conf"])
        st.caption(f"{t['latency']}: {res['latency']:.0f} ms")

# ==========================================
# التبويب الثاني: رفع ملف CSV
# ==========================================
with tab2:
    st.markdown(t["csv_hint"])
    uploaded_file = st.file_uploader(t["upload"], type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if "Tweet" not in df.columns:
            st.error(t["no_col"])
        else:
            if len(df) > 100:
                st.warning(t["too_many"].format(n=len(df)))
                df = df.head(100)

            if st.button(t["start_batch"], type="primary"):
                with st.spinner(t["batch_spinner"]):
                    svc = load_model()
                    texts = df["Tweet"].astype(str).tolist()
                    results = svc.predict(texts)
                    df["Label"] = [r["label"] for r in results]
                    df["Confidence"] = [r["confidence"] for r in results]
                st.session_state["batch_df"] = df

    bdf = st.session_state.get("batch_df")
    if bdf is not None:
        out = bdf.copy()
        out["Prediction"] = out["Label"].map(lambda l: t["pos"] if l == "pos" else t["neg"])
        pos_count = int((bdf["Label"] == "pos").sum())
        neg_count = len(bdf) - pos_count

        st.success(t["batch_ok"])
        col1, col2 = st.columns(2)
        col1.metric(f"{t['pos']} 🟢", f"{pos_count} ({pos_count/len(bdf)*100:.1f}%)")
        col2.metric(f"{t['neg']} 🔴", f"{neg_count} ({neg_count/len(bdf)*100:.1f}%)")

        st.markdown(f"### {t['details']}")
        st.dataframe(out[["Tweet", "Prediction", "Confidence"]])

        csv = out[["Tweet", "Prediction", "Confidence"]].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label=t["download"],
            data=csv,
            file_name="sentiment_results.csv",
            mime="text/csv",
        )
