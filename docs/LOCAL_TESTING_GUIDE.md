# Local Testing Guide - Step by Step

This guide will help you test each component of the project individually on your local machine before Docker deployment.

---

## 📋 Prerequisites

### 1. Check Python Version
```bash
python --version
# Should be Python 3.10+
```

### 2. Verify Virtual Environment
```bash
cd "c:\Users\ASUS\Desktop\Projects\End-To-End-Medical-Assistant-main\End-To-End-Medical-Assistant-main"

# Check if venv exists
dir .venv

# If not, create it
python -m venv .venv
```

### 3. Activate Virtual Environment
```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows CMD
.venv\Scripts\activate.bat
```

### 4. Install/Update Dependencies
```bash
# Install base requirements
pip install -r server/requirements.txt

# Install additional dependencies for new modules
pip install pyyaml
```

---

## ✅ Step 1: Test Environment Setup

### 1.1 Verify .env File
```bash
# Check if .env exists
type .env

# Ensure these keys are present:
# MONGO_URI=...
# PINECONE_API_KEY=...
# GOOGLE_API_KEY=...
# GROQ_API_KEY=...
```

**Expected**: All API keys should be populated

---

## ✅ Step 2: Test Basic Backend (Existing Functionality)

### 2.1 Test Basic Import
```bash
# Test if server can be imported
python -c "from server.main import app; print('✅ Import successful')"
```

**Expected Output**: `✅ Import successful`

**If Error**: Note the error message and we'll fix it

### 2.2 Start Backend Server
```bash
# Start server
uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
```

**Expected Output**:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
```

**Keep this terminal open** and open a new terminal for testing

### 2.3 Test Health Endpoint
```bash
# In a NEW terminal (keeping server running)
curl http://localhost:8000/health
```

**Expected Output**:
```json
{"status":"ok","service":"medical-assistant-api"}
```

### 2.4 Test API Documentation
Open browser: `http://localhost:8000/docs`

**Expected**: Swagger UI with all endpoints listed

---

## ✅ Step 3: Test New Monitoring Module

### 3.1 Test Metrics Endpoint
```bash
# With server running
curl http://localhost:8000/metrics
```

**Expected Output**: JSON with metrics like:
```json
{
  "total_requests": 1,
  "total_errors": 0,
  "error_rate": 0.0,
  "avg_request_duration_ms": ...,
  "timestamp": "..."
}
```

### 3.2 Test Monitoring Import (Standalone)
```bash
# Stop server (CTRL+C) and test monitoring module directly
python -c "from server.monitoring.monitoring import metrics_collector; print('✅ Monitoring module works'); print(metrics_collector.get_metrics())"
```

**Expected**: Should print metrics dict without errors

---

## ✅ Step 4: Test Evaluation Module

### 4.1 Test Metrics Import
```bash
python -c "from server.evaluation.metrics import RAGEvaluator; print('✅ Metrics module works')"
```

### 4.2 Test Metric Calculation
Create a test file: `test_metrics.py`

```python
from server.evaluation.metrics import RAGEvaluator

evaluator = RAGEvaluator()

# Test retrieval metrics
retrieved = ["doc1", "doc2", "doc3"]
relevant = ["doc1", "doc3", "doc4"]

results = evaluator.evaluate_retrieval(retrieved, relevant, k_values=[3])
print("✅ Retrieval Metrics:", results)

# Test generation metrics
reference = "The symptoms of diabetes include increased thirst and frequent urination."
hypothesis = "Diabetes symptoms are increased thirst, frequent urination, and fatigue."

gen_results = evaluator.evaluate_generation(reference, hypothesis)
print("✅ Generation Metrics:", gen_results)
```

Run it:
```bash
python test_metrics.py
```

**Expected**: Should print metric scores without errors

### 4.3 Test Test Dataset
```bash
python -c "from server.evaluation.test_dataset import MEDICAL_TEST_CASES; print(f'✅ Loaded {len(MEDICAL_TEST_CASES)} test cases')"
```

**Expected**: `✅ Loaded 10 test cases`

---

## ✅ Step 5: Test Experiment Framework

### 5.1 Test Experiment Tracker Import
```bash
python -c "from experiments.experiment_tracker import ExperimentConfig; print('✅ Experiment tracker works')"
```

### 5.2 Test Configuration Loading
```bash
python -c "from experiments.experiment_tracker import ExperimentConfig; config = ExperimentConfig.load('experiments/configs/baseline_minilm.yaml'); print('✅ Config loaded:', config.experiment_name)"
```

**Expected**: `✅ Config loaded: baseline_minilm`

### 5.3 Run Example Experiment
```bash
cd experiments
python run_experiment.py
```

**Expected Output**:
```
✅ Saved experiment configuration
✅ Logged experiment results
📊 Experiment Summary (1 runs)
...
```

---

## ✅ Step 6: Test Authentication Flow (Existing)

### 6.1 Ensure Server is Running
```bash
# Start server if not running
uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
```

### 6.2 Test Signup
```bash
# In new terminal
curl -X POST http://localhost:8000/signup ^
  -H "Content-Type: application/json" ^
  -d "{\"username\":\"testuser\",\"password\":\"testpass123\",\"role\":\"patient\"}"
```

**Expected Output**:
```json
{"user_id":"...","username":"testuser","role":"patient"}
```

**If Error**: "Username already exists" - use different username

### 6.3 Test Login
```bash
curl -X GET http://localhost:8000/login ^
  -u testuser:testpass123
```

**Expected Output**:
```json
{"user_id":"...","username":"testuser","role":"patient"}
```

---

## ✅ Step 7: Test Document Upload (Existing)

### 7.1 Create Admin User
```bash
curl -X POST http://localhost:8000/signup ^
  -H "Content-Type: application/json" ^
  -d "{\"username\":\"admin\",\"password\":\"admin123\",\"role\":\"admin\"}"
```

### 7.2 Upload Test Document
```bash
# Create a simple test PDF or use existing one
curl -X POST http://localhost:8000/upload_docs ^
  -u admin:admin123 ^
  -F "files=@uploaded_docs/sample.pdf" ^
  -F "role=public" ^
  -F "doc_id=test_doc_001"
```

**Expected**: Success message with uploaded file info

**If Error**: Check if `uploaded_docs/` has PDF files

---

## ✅ Step 8: Test Chat/Query (Existing)

### 8.1 Test Simple Query
```bash
curl -X POST http://localhost:8000/chat ^
  -u testuser:testpass123 ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What are the symptoms of diabetes?\"}"
```

**Expected Output**:
```json
{
  "answer": "...",
  "sources": ["document.pdf"]
}
```

**If Takes Long**: First query may take time for model loading

---

## ✅ Step 9: Verify Monitoring is Tracking

### 9.1 Check Metrics After Requests
```bash
curl http://localhost:8000/metrics
```

**Expected**: 
- `total_requests` should be > 0
- `endpoint_counts` should show `/health`, `/metrics`, `/chat`, etc.
- `avg_request_duration_ms` should have values

---

## 🔍 Troubleshooting Common Issues

### Issue 1: Import Errors
**Error**: `ModuleNotFoundError: No module named 'server'`

**Fix**:
```bash
# Run from project root
cd "c:\Users\ASUS\Desktop\Projects\End-To-End-Medical-Assistant-main\End-To-End-Medical-Assistant-main"

# Try running with python -m
python -m uvicorn server.main:app --reload
```

### Issue 2: Missing Dependencies
**Error**: `ModuleNotFoundError: No module named 'yaml'`

**Fix**:
```bash
pip install pyyaml
```

### Issue 3: Monitoring Import Error
**Error**: Problems importing monitoring module

**Fix**: Check if `__init__.py` exists in `server/monitoring/`
```bash
# Should exist
type server\monitoring\__init__.py
```

### Issue 4: Port Already in Use
**Error**: `Address already in use`

**Fix**:
```bash
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID)
taskkill /PID <PID> /F

# Or use different port
uvicorn server.main:app --reload --port 8001
```

### Issue 5: Environment Variables Not Loaded
**Error**: `KeyError: 'PINECONE_API_KEY'`

**Fix**: Ensure `.env` file is in project root and contains all keys

---

## 📝 Testing Checklist

After completing all steps, verify:

- [ ] ✅ Server starts without errors
- [ ] ✅ Health endpoint returns 200
- [ ] ✅ Metrics endpoint returns JSON
- [ ] ✅ API docs accessible at `/docs`
- [ ] ✅ Monitoring module imports successfully
- [ ] ✅ Evaluation module works standalone
- [ ] ✅ Experiment tracker runs without errors
- [ ] ✅ Signup/Login works
- [ ] ✅ Document upload succeeds (admin only)
- [ ] ✅ Chat query returns answer
- [ ] ✅ Metrics show request tracking

---

## 🎯 What to Report Back

For each step, let me know:
1. ✅ **Passed** - worked as expected
2. ❌ **Failed** - include the error message
3. ⚠️ **Partial** - works but with warnings

Once all steps pass, we'll move to Docker deployment testing!

---

## 🚀 Next Steps After Local Testing

1. Fix any errors found during testing
2. Create test script for automated testing
3. Proceed to Docker deployment
4. Test Docker container locally
5. Deploy to production

---

**Start with Step 1** and let me know the results!
