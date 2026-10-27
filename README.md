Photonic Organ

Plays “Ode to Joy” as a drawbar organ, with the sound computed by Q.ANT’s Native Computing Toolkit.

A drawbar organ tone is a sum of cosines, A·cos(k·x), which is exactly what the KAN layer computes. The tone is computed in two ways.

kan: one calc_kan_layer_fprop call per note. The phase is rounded to bfloat16 before the chip multiplies it by k, so the error grows on high drawbars.

single: the CPU computes k·x first, then one calc_scaled_periodic_nl_fprop call per drawbar.

For both, the program writes a WAV file and measures its signal-to-error ratio (SINAD) against an exact float64 version.

In the simulator, all errors come from bfloat16 rounding. On real hardware, more errors might be added on top, and since the SINAD is measured from the WAV files, they would show up directly.

Run: python3.12 organ.py (needs numpy, ml_dtypes and the Q.ANT toolkit)
