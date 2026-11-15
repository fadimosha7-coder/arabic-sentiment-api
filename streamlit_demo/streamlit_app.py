import os
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sentiment_api.model import SentimentModel, clean_text  # noqa: E402

MAX_CHARS = 1000
LOW_CONF = 0.8

st.set_page_config(page_title="Arabic sentiment demo", page_icon="💬")
st.title("تصنيف مشاعر التغريدات العربية (شامي)")
st.caption("MARBERTv2 معدّل للمهمة. مشروع تعليمي غير تجاري.")


@st.cache_resource(show_spinner="جاري تحميل الموديل (أول مرة بياخد وقت)...")
def load_model():
    local = os.environ.get("MODEL_DIR")                  # للتجربة المحلية بدون تحميل
    if local:
        return SentimentModel(local)
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id=st.secrets["MODEL_REPO"], token=st.secrets["HF_TOKEN"])
    return SentimentModel(path)


svc = load_model()
text = st.text_area("النص", height=120, max_chars=MAX_CHARS, placeholder="اكتب تغريدة...")

if st.button("صنّف", type="primary"):
    if not text.strip():
        st.error("اكتب نص أولاً.")
    elif not clean_text(text):
        st.error("النص فاضي بعد حذف الروابط والإشارات (@).")
    else:
        t0 = time.perf_counter()
        r = svc.predict([text.strip()])[0]
        ms = (time.perf_counter() - t0) * 1000
        pos = r["confidence"] if r["label"] == "pos" else 1 - r["confidence"]
        st.subheader(f"{'إيجابي' if r['label'] == 'pos' else 'سلبي'} (ثقة {r['confidence']:.2f})")
        st.write(f"إيجابي {pos:.2f} | سلبي {1 - pos:.2f}")
        if r["confidence"] < LOW_CONF:
            st.warning(f"ثقة منخفضة ({r['confidence']:.2f}): الموديل متردد.")
        st.caption(f"زمن التنبؤ: {ms:.0f} ms")

st.markdown("""
**قبل ما تعتمد على النتيجة:**
- الموديل **ثنائي** (إيجابي أو سلبي) وما بيعرف المحايد، وأحياناً بيصنّف الجمل المحايدة بثقة عالية جداً.
- **الثقة مو احتمال إن الجملة فعلاً إيجابية أو سلبية.**
- اتقيّم على تغريدات عربية (شامية وأردنية) فقط. دقته على اختبار من 329 تغريدة حوالي 91% (±3.5 نقطة).
- **ما تكتب معلومات شخصية.**
- مشروع تعليمي، مو لاتخاذ قرارات عن ناس.
""")