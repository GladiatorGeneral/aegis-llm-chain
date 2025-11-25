# 🚀 Quick Start - AEGIS Blockchain Module

## Backend Setup (Python FastAPI)

### 1. Setup Virtual Environment
```powershell
cd E:\Projects\aegis-llm-chain\backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements\base.txt
```

### 2. Configure Environment (if not already done)
```powershell
# Edit .env and add:
# HF_TOKEN=your_token
# DEEPSEEK_API_KEY=your_key
# OPENAI_API_KEY=your_key (optional)
```

### 3. Start Backend Server
```powershell
cd E:\Projects\aegis-llm-chain\backend\src
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Backend URL:** http://localhost:8000  
**API Docs:** http://localhost:8000/docs

---

## Frontend Setup

### 1. Start Frontend Server (new terminal)
```powershell
cd E:\Projects\aegis-llm-chain
npm run dev
```

**Frontend URL:** http://localhost:8080

### 2. Test Blockchain UI
Open: http://localhost:8080/domains/chains.html

---

## Blockchain API Endpoints

All endpoints are under `/api/v1/blockchain/`:

- `POST /compile` - Compile Solidity contracts
- `POST /deploy` - Deploy contracts to networks
- `POST /execute` - Execute contract functions
- `GET /contracts` - List all deployed contracts
- `GET /contracts/{address}` - Get contract details
- `POST /verify` - Verify contract on explorer
- `GET /networks` - List supported networks
- `GET /shard-status` - Get AEGIS shard status

---

## Quick Test

### Test Backend Health
```powershell
curl http://localhost:8000/health
```

### Test Blockchain Endpoint
```powershell
curl http://localhost:8000/api/v1/blockchain/networks
```

### Test Shard Status
```powershell
curl http://localhost:8000/api/v1/blockchain/shard-status
```

---

## Supported Networks

1. **Polygon Mumbai** (testnet) - Recommended for testing
2. **Polygon PoS** (mainnet)
3. **Clover Testnet**
4. **Clover Mainnet**
5. **AEGIS Shard 0** - AI Inference (local)
6. **AEGIS Shard 1** - Data Storage (local)
7. **AEGIS Shard 2** - Governance (local)

---

## Files Created/Modified

✅ **E:\Projects\aegis-llm-chain\backend\src\api\v1\blockchain.py** - New blockchain API  
✅ **E:\Projects\aegis-llm-chain\backend\src\main.py** - Added blockchain router  
✅ **Frontend API client** - Updated to use port 8000

---

## Troubleshooting

**Port 8000 already in use:**
```powershell
uvicorn main:app --reload --port 8001
# Then update frontend API client baseUrl
```

**Module import errors:**
```powershell
# Make sure you're in backend/src directory
cd E:\Projects\aegis-llm-chain\backend\src
```

**Virtual environment not activated:**
```powershell
E:\Projects\aegis-llm-chain\backend\venv\Scripts\Activate.ps1
```

---

Ready to start! Run the commands above and visit http://localhost:8080/domains/chains.html
