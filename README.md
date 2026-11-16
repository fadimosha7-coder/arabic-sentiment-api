![tests](https://github.com/fadimosha7-coder/arabic-sentiment-api/actions/workflows/tests.yml/badge.svg)

# Arabic sentiment API

A small FastAPI service that classifies Arabic (Levantine) tweets as positive or negative, using a fine-tuned MARBERTv2. Built to practice serving, testing and containerizing a model. Not a production system.

## Endpoints

- `POST /predict`: 1 to 16 texts (max 1,000 characters each, not empty after removing links and @mentions) -> label (`pos`/`neg`), confidence, latency in ms. Invalid input returns 422.
- `GET /health`: whether the model is loaded.
- Interactive docs at `/docs`.

Binary only: there is no neutral class. Neutral sentences are forced into pos or neg, sometimes with confidence above 0.99, so the confidence is NOT the probability that a sentence is really positive or negative.

## Example

    import httpx
    r = httpx.post("http://127.0.0.1:8000/predict", json={"texts": ["المبارة كانت روعة"]})
    print(r.json())   # {'results': [{'label': 'pos', 'confidence': 0.93}], 'latency_ms': 85.2}  (illustrative values)

## Run locally

    python -m venv .venv
    .venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
    pip install -r requirements.txt
    uvicorn sentiment_api.main:app --app-dir src --port 8000

Put the model folder at `models/sentiment_model/` or set `MODEL_DIR` to its path.

## Tests

    pip install -r requirements-dev.txt
    pytest -q          # 15 tests

## Docker

    docker build -t sentiment-api:0.1 .
    docker run --rm -p 8000:8000 -v "<absolute path>/models/sentiment_model:/models/sentiment_model:ro" sentiment-api:0.1

The model is mounted, not baked into the image. Image size: 381 MB.

## The model (not included - licensing restrictions)

Folder with `model.safetensors`, tokenizer files, `config.json` and `serving_config.json` (654 MB). It is not published: it was trained on ArSenTD-LEV (no known license) from UBC-NLP/MARBERTv2 (license not verified). Training notebooks: link TBD.
Quality (same recipe, seed 42): val 89.6%, test 86.6%, external AJGT 91.3%. Mean of 3 seeds: test 88.0%, AJGT 91.1%. Majority-class baselines: about 60% and 50%. Binary task, tweets only, test set of 329 tweets (about +/-3.5 points): do not extrapolate.

## Performance (my laptop, CPU, one request at a time, fp32)

About 81 ms for one text, about 1.0 s for 16 texts. Not load-tested.
Dynamic int8 quantization was tried and rejected by a rule written before measuring: 395 MB, 37 ms per text, but validation accuracy 85.9% vs 89.6% (16 predictions went from correct to wrong, 4 the other way).

## Limits

No authentication or rate limiting: local use only. Single process. English or MSA text was not evaluated.

## Documentation

Model card: MODEL_CARD.md | Data card: DATA_CARD.md | Licenses: LICENSES.md

## Local demo (Gradio)

    pip install -r requirements-ui.txt
    python scripts/ui.py        # http://127.0.0.1:7860, local only (share=False on purpose)

The page shows the prediction, the confidence and the model's limits side by side.

## Live demo (free hosting, may be asleep)

https://arabic-sentiment-api-3ehcy8caad8bgwvjghkux2.streamlit.app/
First load can take a few minutes: the app sleeps after 12 hours without visitors and downloads the model on start.
The model weights are NOT published: the demo loads them at runtime from a private repository. Inference only.
Do not type personal information. The demo does not store inputs on purpose (I do not control the host's server logs).
Non-commercial educational project. Sources: MARBERTv2 (ARBERT & MARBERT, ACL 2021), ArSenTD-LEV (arXiv:1906.01830), AJGT.
If you are a rights holder and want it removed, contact me via GitHub and I will take the demo down.
