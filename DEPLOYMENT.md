# OfflineRAG Deployment Guide

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Vercel        │────▶│   Railway/Render │────▶│   Ollama        │
│   (Frontend)    │     │   (Backend)      │     │   (LLM Server)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Option 1: Deploy Backend to Railway (Recommended)

### Step 1: Deploy Backend to Railway

1. **Create Railway Account**: https://railway.app

2. **Install Railway CLI**:
   ```bash
   npm install -g @railway/cli
   railway login
   ```

3. **Initialize Project** (in backend folder):
   ```bash
   cd backend
   railway init
   ```

4. **Set Environment Variables**:
   ```bash
   railway variables set DEBUG=false
   railway variables set OLLAMA_HOST=https://your-ollama-server.com
   railway variables set CORS_ORIGINS=https://your-app.vercel.app
   ```

5. **Deploy**:
   ```bash
   railway up
   ```

6. **Get your Backend URL**: 
   ```bash
   railway domain
   ```
   Example: `https://offlinerag-backend.up.railway.app`

### Step 2: Deploy Frontend to Vercel

1. **Install Vercel CLI**:
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**:
   ```bash
   vercel login
   ```

3. **Deploy Frontend** (in frontend folder):
   ```bash
   cd frontend
   vercel
   ```

4. **Set Environment Variables in Vercel Dashboard**:
   - Go to your project settings
   - Add: `VITE_API_URL` = `https://your-railway-backend.up.railway.app`

5. **Redeploy** to apply environment variables:
   ```bash
   vercel --prod
   ```

---

## Option 2: Deploy Backend to Render

### Step 1: Create render.yaml

Already created in `backend/render.yaml`

### Step 2: Deploy

1. Go to https://render.com
2. Click "New Web Service"
3. Connect your GitHub repo
4. Select the backend folder
5. Set environment variables:
   - `OLLAMA_HOST`: Your Ollama server URL
   - `CORS_ORIGINS`: Your Vercel frontend URL

---

## Ollama Deployment Options

Since Ollama requires GPU for reasonable performance, you have these options:

### Option A: Use Ollama Cloud API (Easiest)
```
OLLAMA_HOST=https://api.ollama.com
OLLAMA_API_KEY=your-key
```

### Option B: Self-host on GPU Server
- Deploy on RunPod, Lambda Labs, or a VPS with GPU
- Run: `ollama serve --host 0.0.0.0`

### Option C: Use Alternative LLM Providers
Modify backend to use:
- OpenAI API (not offline)
- Anthropic Claude (not offline)
- Local CPU inference (very slow)

---

## Environment Variables Reference

### Backend (.env)
```env
DEBUG=false
OLLAMA_HOST=https://your-ollama-server
CORS_ORIGINS=https://your-app.vercel.app
DATA_DIR=/app/data
WHISPER_MODEL=base
LLM_MODEL=llama3.2
```

### Frontend (.env)
```env
VITE_API_URL=https://your-backend.railway.app
```

---

## Important Notes

1. **ML Models**: The backend downloads models on first run (~2-3GB). Railway/Render may timeout. Consider:
   - Pre-building a Docker image with models
   - Using smaller models (whisper tiny, smaller embeddings)

2. **Persistent Storage**: 
   - Railway: Add a volume for `/app/data`
   - Render: Use Render Disks

3. **Cold Starts**: Serverless platforms have cold starts. The app needs ~30-60s to load models.

4. **Cost Estimates**:
   - Railway: ~$5-20/month for backend
   - Render: ~$7-25/month for backend
   - Vercel: Free for frontend (hobby tier)

---

## Quick Deploy Commands

### One-Command Backend Deploy (Railway)
```bash
cd backend
railway login
railway init
railway variables set DEBUG=false CORS_ORIGINS=https://your-vercel-app.vercel.app
railway up
railway domain
```

### One-Command Frontend Deploy (Vercel)
```bash
cd frontend
vercel login
vercel env add VITE_API_URL
# Enter your Railway backend URL
vercel --prod
```
