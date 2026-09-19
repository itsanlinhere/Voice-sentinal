import time
import requests
import json

# Test with Flask test client to thoroughly verify routes
from app import app

client = app.test_client()

print("Testing /health...")
res = client.get("/health")
assert res.status_code == 200, f"Health check failed: {res.status_code}"
data = res.get_json()
print("Health response:", data)
assert data["status"] == "ok"
assert data["model_loaded"] == True

print("Testing GET /...")
res = client.get("/")
assert res.status_code == 200
assert b"VOICE SENTINEL" in res.data
print("GET / passed!")

print("Testing GET /live...")
res = client.get("/live")
assert res.status_code == 200
assert b"Live Microphone" in res.data
print("GET /live passed!")

print("Testing GET /train...")
res = client.get("/train")
assert res.status_code == 200
assert b"Model Training Pipeline" in res.data
print("GET /train passed!")

print("Testing GET /dataset...")
res = client.get("/dataset")
assert res.status_code == 200
assert b"Dataset Explorer" in res.data
print("GET /dataset passed!")

print("Testing GET /evaluation...")
res = client.get("/evaluation")
assert res.status_code == 200
assert b"Confusion Matrix" in res.data
assert b"Generalization Benchmark" in res.data
print("GET /evaluation passed!")

print("Testing GET /model-info...")
res = client.get("/model-info")
assert res.status_code == 200
assert b"Acoustic Architecture" in res.data
print("GET /model-info passed!")

print("Testing POST /analyze with Human preset...")
res = client.post("/analyze", data={
    "sample_preset": "dataset/human/speaker_01/human_spk1_sample_01.wav",
    "threshold": "0.5"
})
assert res.status_code == 200
assert b"HUMAN" in res.data
assert b"Acoustic Features Extracted From Signal" in res.data
print("POST /analyze with Human preset passed!")

print("Testing POST /analyze with AI preset...")
res = client.post("/analyze", data={
    "sample_preset": "dataset/ai/engine_01/synth_eng1_sample_01.wav",
    "threshold": "0.5"
})
assert res.status_code == 200
assert b"AI-GENERATED" in res.data
print("POST /analyze with AI preset passed!")

print("All endpoints tested successfully with 100% pass rate!")
