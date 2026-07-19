# Invalid Stage A launch

The wrapper assigned the child to the registered RAM/CPU Job Object, then
encountered `[N/A]` in the WDDM `nvidia-smi` compute-app memory field and
attempted to cast it to a number. The monitor stopped the owned process after
five samples. No proposal completed and no partial screening record exists.

This is a telemetry construction failure, not a scientific outcome. The
correction accepts only numeric per-process VRAM fields and leaves all frozen
proposal/config hashes unchanged.
