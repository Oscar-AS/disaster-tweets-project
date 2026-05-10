---
title: Disaster Tweets API
emoji: "🔥"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

## FastAPI inference proxy

This Space runs a FastAPI server that:

- cleans and (optionally) translates input text
- calls the Hugging Face Inference API for your model
- returns `is_disaster`, `confidence`, and basic word-importance scores

### Endpoints

- `GET /` : status message
- `GET /health` : service health + HF config
- `POST /predict` : prediction

### Environment variables (Space Settings → Variables/Secrets)

- `HF_API_URL` (optional if you set `HF_MODEL_ID`): example `https://api-inference.huggingface.co/models/Oscarkaf/disaster-tweets-bert`
- `HF_MODEL_ID` (optional if `HF_API_URL` already contains `/models/...`): example `Oscarkaf/disaster-tweets-bert`
- `HF_TOKEN` (**Secret**, only needed if model is private)

### Test quickly

Open the interactive docs:

- `/docs`
