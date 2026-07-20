# Checkpoint: LSPG Calibration Construction Failure v1

## Attempt

The bounded watcher reached an idle GPU after 5,304 seconds and launched
`LSAD-C-K2L8-calibration-S0-s397` through the registered wrapper.

## Failure

PyTorch rejected the index-free `torch.device("cuda")` value passed to
`set_per_process_memory_fraction`. The cell exited before model allocation or
training with:

```text
ValueError: Expected a torch.device with a specified index or an integer, but got: cuda
```

The wrapper receipt records `process_exit_1`, 431.930 MB peak RAM, 0 MB peak
VRAM, 18.032 MB/s peak I/O, no lingering process, and successful cleanup. No
task metric or calibration measurement was produced.

## Correction

Device selection now returns `cuda:<current_device>` and has a regression test
that requires the explicit index. The registered architecture, data, resource
caps, calibration count, and promotion rules are unchanged.

## Next Step

Commit this construction failure and correction, then retry calibration from
S0 while the GPU remains idle.
