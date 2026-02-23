# Advanced Benchmarking Guide: Working Sets, Tail Latency & Interference

## Overview

Three new benchmarking experiments for HGSEL that test production-readiness hypotheses:

| Experiment | Hypothesis | Falsification | Use Case |
|-----------|-----------|--------------|----------|
| **Trace-Driven Workset** | Working sets are predictable, grow sublinearly | Working sets are random, caching impossible | Compiled routing, cache design |
| **Tail-Latency Decomp** | One component dominates, tail is predictable | All components equal, tail is random | Architecture optimization, SLO planning |
| **Expert Interference** | Interference < 5%, multi-tenancy safe | Interference catastrophic, need isolation | Multi-user deployment, QoS design |

---

## 1. Trace-Driven Expert Working-Set Modeling

### What It Tests

For each context length, traces which expert IDs are routed to and measures:
- **Working set size:** How many unique experts are actually used?
- **Utilization:** What fraction of the total expert pool is active?
- **Predictability:** Does this stay consistent across different contexts?

### Why It Matters

If working sets are small and predictable:
- ✓ Can compile expert dispatch (route decisions known ahead of time)
- ✓ Can pre-fetch expert weights into L1/L2 cache
- ✓ Can partition experts on multiple GPUs predictably
- ✓ Enables "cold expert" standby and activation on demand

If working sets are large or random:
- ✗ Need full expert bank in cache
- ✗ Cannot compile patterns (dynamic routing required)
- ✗ Expert affinity becomes meaningless

### How to Run

```bash
cd hgsel-moe
python experiments/trace_driven_workset.py
```

### Interpreting Results

**Coefficient of Variation (CV):**
- CV < 0.15: **HIGHLY PREDICTABLE** → Compile expert dispatch
- CV 0.15-0.30: **MODERATELY PREDICTABLE** → Adaptive caching
- CV > 0.30: **UNPREDICTABLE** → Full cache or no caching

**Working Set Size vs Context Length:**
- Sublinear growth → Cache hit rates improve with longer contexts
- Linear/superlinear → Cache hit rates degrade (interference increases)

---

## 2. Tail-Latency Decomposition

### What It Tests

Profiles the forward pass and measures latency (per-token, microseconds) for:
- **p50 (median):** Typical-case latency
- **p99 (tail):** 1% of requests are slower
- **p999 (extreme):** 0.1% of requests are even slower

### Why It Matters

Understanding latency tails is critical for:
- **SLO negotiation:** Can you guarantee p99 < 100µs per token?
- **Architecture decisions:** Is the bottleneck hardware (kernel) or software (routing, dispatch)?
- **Optimization ROI:** Where should you spend engineering effort?

If one component dominates the tail:
- ✓ Clear optimization target (batch that component, pipeline it, specialize for it)
- ✓ Predictable: Can make SLO commitments

If tail is unpredictable (high coefficient of variation):
- ✗ Hardware randomness dominates (power throttling, cache contention, network jitter)
- ✗ May need isolation (dedicated resources, priority scheduling)

### How to Run

```bash
cd hgsel-moe
python experiments/tail_latency_decomposition.py
```

### Interpreting Results

**Coefficient of Variation (CV):**
- CV < 0.2: **PREDICTABLE** → Workload co-scheduling safe, can aggregate SLOs
- CV 0.2-0.5: **MODERATE** → Use timeout-based scheduling
- CV > 0.5: **UNPREDICTABLE** → Need architecture redesign (isolation, guaranteed resources)

**p99 vs p50 ratio:**
- p99/p50 < 1.5: Tight distribution (well-understood bottleneck)
- p99/p50 1.5-3: Moderate tail (some variability)
- p99/p50 > 3: Severe tail (need investigation)

---

## 3. Expert Interference Benchmarks

### What It Tests

Runs two concurrent workloads with different token distributions and measures:
- **Latency blowup:** How much slower does each workload get?
- **Interference percentage:** (interleaved_latency - baseline) / baseline × 100
- **Worst-case scenarios:** Alternating (cache thrashing) vs random (realistic)

**Workloads:**
- **Workload A:** "Coding" (skewed towards first half of vocabulary)
- **Workload B:** "Math" (skewed towards second half of vocabulary)

### Why It Matters

In production, many users share expert capacity. Questions:
- Can we safely co-schedule workloads? (multi-tenancy)
- Do experts trained for one task degrade on another? (quality cross-talk)
- How much L1/L2 cache interference occurs? (hardware contention)

Results determine deployment strategy:
- **Small interference (< 5%):** Oversubscribe experts, use lightweight scheduling
- **Moderate interference (5-20%):** Use priority scheduling, soft partitioning
- **Severe interference (> 20%):** Need strict partitioning, dedicated expert pools

### How to Run

```bash
cd hgsel-moe
python experiments/expert_interference_benchmark.py
```

### Interpreting Results

**Three scenarios are measured:**

1. **Isolated (baseline):** Each workload runs alone
   - Workload A: X µs/token
   - Workload B: Y µs/token

2. **Alternating:** A-B-A-B pattern (worst-case L1/L2 pressure)
   - Interference_A = (Latency_A_alt - X) / X × 100
   - Interference_B = (Latency_B_alt - Y) / Y × 100

3. **Random:** Random interleaving (realistic worst-case)
   - Interference_A = (Latency_A_rand - X) / X × 100
   - Interference_B = (Latency_B_rand - Y) / Y × 100

**Decision Matrix:**

| Max Interference | Verdict | Action |
|-----------------|---------|--------|
| < 5% | Negligible | Multi-tenancy safe, oversubscribe |
| 5-20% | Moderate | Use soft partitioning (priorities) |
| > 20% | Severe | Strict partitioning needed |

---

## Running All Benchmarks Together

Create a test suite that runs all three:

```bash
#!/bin/bash
cd hgsel-moe

echo "Running advanced benchmarks..."
python experiments/trace_driven_workset.py > ../results/workset.log 2>&1
python experiments/tail_latency_decomposition.py > ../results/tail_latency.log 2>&1
python experiments/expert_interference_benchmark.py > ../results/interference.log 2>&1

echo "All benchmarks complete. Results in results/ directory."
ls -la ../results/
```

---

## Outputs

All benchmarks generate:
- **Console output:** Immediate pass/fail summaries
- **PNG plots:** `results/workset_curve.png`, `results/tail_latency_decomp.png`, `results/expert_interference.png`
- **Log files:** Detailed measurements for post-processing

---

## Common Questions

### Q: Why these specific tests?

**A:** They test the three main assumptions for production deployment:

1. **Working sets:** Can we compile and cache expert decisions?
2. **Tail latency:** Can we meet SLOs predictably?
3. ** Interference:** Can we safely share experts across users?

### Q: What if results contradict each other?

**Example:** Small working sets (good for caching) but high interference (bad for sharing).

**A:** This is valuable information! It means:
- Experts ARE specialized (good: working sets are small)
- Experts COMPETE for cache (bad: sharing causes thrashing)

**Solution:** Use soft partitioning (allocate experts to workload groups) or intelligent prefetching (preload experts for upcoming requests).

### Q: Can I modify these benchmarks?

**A:** Yes! Common extensions:

1. **Change workload characteristics:** Modify `WorkloadGenerator` to match your domain
2. **Test more experts:** Edit `n_experts` and `k_active` parameters
3. **Vary context length:** Extend `context_lengths` list in `trace_driven_workset.py`
4. **Add more concurrent workloads:** Duplicate workload classes for >2 tasks

### Q: How do these relate to Phase 3 training?

**A:** Phase 3 trains load-balanced experts. These benchmarks validate that:

- ✓ Load balancing works (experts are utilized predictably)
- ✓ Experts specialize (working sets are small)
- ✓ System is production-ready (tail latency is predictable, interference acceptable)

If any test fails, debug training (salt tuning, auxiliary loss weight, expert count).

---

## Next Steps

After running these benchmarks:

1. **Predictable working sets?** → Design routing compiler (Phase 4+)
2. **Long tail latency?** → Profile individual components, pipeline operations
3. **High interference?** → Add expert affinity metadata, implement QoS scheduling
4. **All pass?** → Ready for multi-GPU and production deployment

---

**See also:** [HGSEL_BUILD_PLAN.md](./HGSEL_BUILD_PLAN.md) for Phase roadmap.
