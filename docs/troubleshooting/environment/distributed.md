# Distributed Training Issues

Multi-GPU and multi-node training problems.

## DDP Hangs or Deadlocks

```python
# 1. Set environment variables
import os
os.environ["NCCL_DEBUG"] = "INFO"  # Debug NCCL issues

# 2. Use timeout to detect hangs
trainer = AutoTrainer(
    accelerator="gpu",
    devices=2,
    strategy="ddp",
    plugins=[
        {"timeout": 1800}  # 30 minute timeout
    ],
)

# 3. If still hangs, try ddp_spawn
trainer = AutoTrainer(
    strategy="ddp_spawn",
    devices=2,
)
```

## Multi-Node Training Issues

```bash
# Set master node address
export MASTER_ADDR=192.168.1.1
export MASTER_PORT=29500

# Run on each node
python train.py --num_nodes=2 --node_rank=0  # Master node
python train.py --num_nodes=2 --node_rank=1  # Worker node
```

```python
# In code
trainer = AutoTrainer(
    accelerator="gpu",
    devices=4,
    num_nodes=2,
    strategy="ddp",
)
```

## Related Issues

- [Device Errors](device-errors.md) - CUDA and GPU problems
- [OOM Errors](../performance/oom-errors.md) - Memory issues with multiple GPUs
- [Slow Training](../performance/slow-training.md) - Multi-GPU performance
