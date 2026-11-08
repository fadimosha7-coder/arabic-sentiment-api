from sentiment_api.model import clean_text


def test_clean_removes_urls_and_mentions():
    assert clean_text("مرحبا @user https://t.co/abc  كيفك") == "مرحبا كيفك"


def test_output_structure(svc):
    res = svc.predict(["المبارة كانت روعة", "الوضع صار صعب"])
    assert len(res) == 2
    for r in res:
        assert r["label"] in {"neg", "pos"}
        assert 0.5 <= r["confidence"] <= 1.0


def test_clear_examples(svc):
    pos, neg = svc.predict(["المبارة كانت روعة", "الوضع صار صعب"])
    assert pos["label"] == "pos"
    assert neg["label"] == "neg"


def test_empty_and_very_long_inputs_do_not_crash(svc):
    res = svc.predict(["", "كلمة " * 5000])
    assert len(res) == 2


def test_batch_matches_single(svc):
    texts = ["المبارة كانت روعة", "الوضع صار صعب", "مرات اقعد بيني وبين حالي ما بعرف شو اعمل"]
    together = svc.predict(texts)
    alone = [svc.predict([t])[0] for t in texts]
    for a, b in zip(together, alone):
        assert a["label"] == b["label"]
        assert abs(a["confidence"] - b["confidence"]) < 0.01