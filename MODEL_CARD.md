# Model card: Arabic (Levantine) tweet sentiment, MARBERTv2 fine-tune v0.1

## Model

UBC-NLP/MARBERTv2 (BERT-base, about 163M parameters) fine-tuned for binary sentiment (neg/pos): 4 epochs, lr 2e-5, batch 16, max length 96, seed 42, no best-epoch selection.

## Intended use

Learning and portfolio demo: classify short Arabic tweets (Levantine / Jordanian style) as positive or negative.

## Out of scope

- Neutral or mixed sentiment: the model has no neutral class. Neutral sentences are forced into pos or neg, in my 2 informal tests with confidence about 0.99.
- Text types not evaluated: reviews, news, formal Arabic, other dialects (Egyptian, Gulf, Maghrebi).
- Any decision about people, moderation, or production use.

## Training data

ArSenTD-LEV, binary subset: 1,523 train tweets (see DATA_CARD.md).

## Evaluation (decision rules written before running; test and external set opened once)

|                                     | Val (326) | Test (329)        | AJGT (1,800)      |
| ----------------------------------- | --------- | ----------------- | ----------------- |
| Majority class                      | 59.5%     | 59.9%             | 50.0%             |
| TF-IDF + logistic regression        | 74.5%     | 74.5%             | 74.8%             |
| This model family (mean of 3 seeds) | 90.5%     | 88.0% (86.6-89.1) | 91.1% (90.3-91.6) |
| Seed 42                             | 89.6%     | 86.6%             | 91.3%             |

The saved model is a retrain of seed 42 with the same recipe (val 89.6%, identical); test and AJGT were not re-measured on the saved files.
Test margin of error is about +/-3.5 points (329 tweets).

## Known behavior

- Errors are balanced between classes (19 neg->pos, 15 pos->neg on val); no confirmed pattern by topic, country or length.
- Training without politics tweets cost about 2 points on politics tweets (not confirmed).
- Confidence is weak for filtering: at threshold 0.95 the model still answers 95% of test tweets (89.4% vs 86.6% accuracy).
- Qualitative impression from the author (not counted): errors include sarcasm, negation and ambiguous labels.

## Ethical notes

Sentiment labels are annotators' judgments and can be ambiguous. Tweets are personal opinions: do not use for profiling individuals.

## Performance

CPU laptop, fp32: about 81 ms per text. Int8 quantization was rejected (val 85.9%).

## Licenses and provenance

See LICENSES.md. Status: No explicit commercial license found. Educational/research use only.
