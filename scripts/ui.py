import os
import sys
import time
from pathlib import Path

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr
from sentiment_api.model import SentimentModel, clean_text
from sentiment_api.schemas import MAX_CHARS

MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models" / "sentiment_model"))
LOW_CONF = 0.8
svc = SentimentModel(MODEL_DIR)


def classify(text):
    text = (text or "").strip()
    if not text:
        raise gr.Error("اكتب نص أولاً.")
    if len(text) > MAX_CHARS:
        raise gr.Error(f"النص أطول من {MAX_CHARS} حرف.")
    if not clean_text(text):
        raise gr.Error("النص فاضي بعد حذف الروابط والإشارات (@).")
    t0 = time.perf_counter()
    r = svc.predict([text])[0]
    ms = (time.perf_counter() - t0) * 1000
    conf = r["confidence"]
    pos = conf if r["label"] == "pos" else 1 - conf
    note = f"زمن التنبؤ: {ms:.0f} ms."
    if conf < LOW_CONF:
        note = f"⚠️ ثقة منخفضة ({conf:.2f}): الموديل متردد. " + note
    return {"إيجابي": pos, "سلبي": 1 - pos}, note


DISCLAIMER = """
**قبل ما تعتمد على النتيجة:**
- الموديل **ثنائي** (إيجابي أو سلبي) وما بيعرف المحايد: الجمل المحايدة بتنصنّف كمان، وأحياناً بثقة عالية جداً.
- **الثقة مو احتمال إن الجملة فعلاً إيجابية أو سلبية.** هي بس أقرب فئة من الاتنين.
- اتقيّم على تغريدات عربية (شامية وأردنية) فقط، مو على مراجعات أو أخبار أو لهجات تانية.
- دقته على اختبار من 329 تغريدة: حوالي 88% (هامش ±3.5 نقطة).
- مشروع تعليمي، مو لاتخاذ قرارات عن ناس.
"""

EXAMPLES = [
    ["المبارة كانت روعة"],
    ["الوضع صار صعب"],
    ["الطقس اليوم متقلب في العديد من المناطق"],
    ["يا سلام، يعني بدك تقنعني إنو الخدمة ممتازة؟"],
]

with gr.Blocks(title="Arabic sentiment demo") as demo:
    gr.Markdown("# تصنيف مشاعر التغريدات العربية (شامي)\nموديل MARBERTv2 معدّل للمهمة. للتعلّم والعرض فقط.")
    txt = gr.Textbox(label="النص", lines=3, max_lines=6, rtl=True, placeholder="اكتب تغريدة...")
    with gr.Row():
        btn = gr.Button("صنّف", variant="primary")
        gr.ClearButton([txt])
    out = gr.Label(label="النتيجة", num_top_classes=2)
    note = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=txt)
    gr.Markdown(DISCLAIMER)
    btn.click(classify, txt, [out, note])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)