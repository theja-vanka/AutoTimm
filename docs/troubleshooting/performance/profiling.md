# Performance Profiling

Identifying bottlenecks in training and inference.

## Identifying Bottlenecks

```python
import torch.profiler

# Profile training
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
) as prof:
    # Run a few training steps
    for i, batch in enumerate(train_loader):
        if i >= 10:  # Profile 10 batches
            break
        outputs = model(batch["images"])
        loss = criterion(outputs, batch["labels"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export for visualization
prof.export_chrome_trace("trace.json")
# View in chrome://tracing
```

## Data Loading Profiling

```python
import time

# Measure data loading time
loader = datamodule.train_dataloader()
times = []

for i, batch in enumerate(loader):
    start = time.time()
    # Just iterate, don't process
    end = time.time()
    times.append(end - start)
    if i >= 100:
        break

print(f"Average batch load time: {sum(times)/len(times):.4f}s")
print(f"Max batch load time: {max(times):.4f}s")

# If slow, increase num_workers
data = ImageDataModule(
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
)
```

## Memory Profiling

```python
import torch

# Track memory allocations
torch.cuda.memory._record_memory_history()

# Run training
trainer.fit(model, datamodule=data)

# Get memory snapshot
snapshot = torch.cuda.memory._snapshot()
torch.cuda.memory._dump_snapshot("memory_snap.pickle")

# Analyze with https://pytorch.org/memory_viz
```

## Related Issues

- [Slow Training](slow-training.md) - Performance optimization tips
- [OOM Errors](oom-errors.md) - Memory issues
- [Device Errors](../environment/device-errors.md) - Hardware problems
