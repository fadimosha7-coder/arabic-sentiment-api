import time, httpx

URL = "http://127.0.0.1:8000/predict"
one = {"texts": ["المبارة كانت روعة"]}
many = {"texts": ["المبارة كانت روعة", "الوضع صار صعب"] * 8}   # 16 نص

def call(payload):
    t0 = time.perf_counter()
    r = httpx.post(URL, json=payload, timeout=60)
    total = (time.perf_counter() - t0) * 1000
    return r.json()["latency_ms"], total

print("single text:")
for i in range(6):
    inside, total = call(one)
    print(f"  request {i + 1}: inside {inside:.0f} ms | total {total:.0f} ms")

print("16 texts:")
for i in range(3):
    inside, total = call(many)
    print(f"  request {i + 1}: inside {inside:.0f} ms | total {total:.0f} ms")