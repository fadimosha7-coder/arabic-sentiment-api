import os, sys, psutil
sys.path.insert(0, "src")
from sentiment_api.model import SentimentModel

p = psutil.Process(os.getpid())
print("before:", round(p.memory_info().rss / 1e6), "MB")
svc = SentimentModel("models/sentiment_model")
svc.predict(["المبارة كانت روعة"])
print("after:", round(p.memory_info().rss / 1e6), "MB")