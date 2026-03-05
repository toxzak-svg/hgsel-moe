# Google Colab Pro Execution Guide

**Goal:** Validate Phase 4 on Colab Pro with minimal cost and maximum efficiency.

---

## Quick Start (5 minutes)

1. **Upload code to GitHub:**
   ```bash
   cd c:/dev/research/214/hgsel-moe
   git add .
   git commit -m "Phase 4 ready for Colab validation"
   git push origin main
   ```

2. **Open Colab notebook:**
   - Go to: https://colab.research.google.com/
   - Upload: `notebooks/phase4_colab_validation.ipynb`
   - Or: File → Upload Notebook

3. **Select GPU runtime:**
   - Runtime → Change runtime type
   - Hardware accelerator: **GPU**
   - GPU type: **A100** (if available) or **V100**
   - Click Save

4. **Run all cells:**
   - Runtime → Run all
   - Estimated time: 2-4 hours

---

## What Gets Validated

| Experiment | Runtime | What It Proves |
|------------|---------|----------------|
| GPU Baseline | 20-30 min | HGSEL isn't slower than dense |
| Convergence Test | 30-45 min | Training works on GPU |
| Token Exchange | 15-20 min | Distributed primitives work (if 2+ GPUs) |
| Memory Profile | 10-15 min | Memory usage is reasonable |

**Total: ~2-4 hours** depending on GPU

---

## Expected Results

### GPU Baseline (Good Sign ✅)
```
DENSE:
  Throughput: 2500 tokens/sec
  Peak Memory: 1200 MB
  
HGSEL:
  Throughput: 2200 tokens/sec  (0.88x – acceptable due to routing overhead)
  Peak Memory: 1400 MB
```

### Convergence (Good Sign ✅)
```
Epoch 1 Loss: 5.2 → 4.8
Epoch 2 Loss: 4.8 → 4.4
Epoch 3 Loss: 4.4 → 4.1

Expert Entropy: 0.95 (balanced)
```

### Memory (Good Sign ✅)
```
Total: <8GB (fits on V100/A100 with headroom)
Params: ~70MB
Activations: ~200MB
Optimizer: ~140MB
```

---

## Cost Breakdown

**Colab Pro compute units:**
- A100: ~2 units/hour
- V100: ~1 unit/hour
- You get ~100 units/month

**Phase 4 validation cost:**
- 4 hours on A100: ~8 units (~8% of monthly quota)
- 4 hours on V100: ~4 units (~4% of monthly quota)

**After Phase 4, you still have 90+ units for:**
- Phase 5 benchmarking
- Debugging/iterations
- Paper experiments

---

## Tips for Maximizing Your $10

### 1. Use Sessions Efficiently
- Start with quick experiments (<1 hour)
- Only run long experiments after smoke tests pass
- Use background execution for overnight runs

### 2. Save Checkpoints Frequently
```python
# In notebook cells, add checkpoint saving
!mkdir -p /content/drive/MyDrive/checkpoints
!cp -r checkpoints/* /content/drive/MyDrive/checkpoints/
```

### 3. Download Results Immediately
```python
from google.colab import files
files.download('results/gpu_baseline/colab_baseline.json')
```

### 4. Use High-Priority GPU When Available
- Runtime → Change runtime type
- GPU type: **Premium** (A100)
- Only use when running important experiments

### 5. Monitor Compute Units
- Hamburger menu → Manage usage
- Track remaining units
- Stop experiments early if results look bad

---

## Alternative: Kaggle (Free Backup)

If you run out of Colab units, Kaggle gives **30 GPU hours/week free**:

1. Go to kaggle.com
2. Create notebook
3. Settings → Accelerator → GPU T4 x2
4. Upload code as dataset
5. Run same experiments

**Pro:** Free  
**Con:** 12-hour session limit, need to save checkpoints

---

## Decision Tree

```
Run GPU Baseline (30 min)
├─ Throughput ≥0.8x dense? 
│  ├─ YES → Continue to convergence test
│  └─ NO → Debug or reassess (don't waste more time)
│
Run Convergence Test (45 min)
├─ Loss decreasing?
│  ├─ YES → Run memory profile
│  └─ NO → Debug training loop
│
Memory Profile (15 min)
├─ Memory <80% GPU capacity?
│  ├─ YES → Phase 4 PASS ✅
│  └─ NO → Reduce batch size and retest
```

**Stop at any failure** → Don't waste compute units on broken experiments

---

## Success Criteria (Phase 4 Pass)

✅ HGSEL throughput ≥ 0.8x dense  
✅ Training loss decreases over epochs  
✅ Expert entropy > 0.8 (balanced load)  
✅ Memory usage < 80% of GPU  
✅ No crashes or OOM errors  

**If all pass:** You have proof HGSEL works! Update repo and consider Phase 5.

**If any fail:** Debug specific issue before continuing.

---

## After Phase 4 Validation

### Option A: Stop Here (Conservative)
- Document results in README
- Label Phase 4 as "validated on Colab Pro"
- Use remaining compute units for other projects
- **Cost: $10 total**

### Option B: Continue to Phase 5 (Worth It)
- Run full benchmark sweeps
- Performance optimization
- Generate publication-quality plots
- **Cost: Still just $10** (you have 90+ units left)

### Option C: Deep Dive (Risky)
- Full Phase 5-9 implementation
- Might need another month ($10) or Vast/Lambda
- Only if Phase 4 shows strong results

---

## Troubleshooting

### "Runtime disconnected"
- Enable background execution
- Or: Use `tmux` equivalent (Colab keeps running)

### "GPU out of memory"
- Reduce `--batch-size` from 16 to 8
- Reduce `--d-ff` from 1024 to 768
- Reduce `--num-layers` from 4 to 2

### "Can't access GitHub repo"
- Make repo public temporarily
- Or: Upload code as ZIP to Google Drive
- Or: Use personal access token for private repos

### "Results not saving"
- Check Drive mounted: `drive.mount('/content/drive')`
- Verify path exists: `!ls /content/drive/MyDrive/`
- Manually download: `files.download('results.json')`

---

## Summary

**Total investment:**
- Time: 2-4 hours of GPU runtime (mostly unattended)
- Money: $10 for the month
- Compute: ~10% of your monthly Colab Pro quota

**What you get:**
- Proof HGSEL scales to GPU
- Benchmark numbers for portfolio/paper
- Decision point for whether to continue

**Risk:**
- Minimal – only $10
- Keep 90% of compute for other work
- Can cancel anytime

This is a **high-value, low-risk** validation path. Run the notebook and see what happens!
