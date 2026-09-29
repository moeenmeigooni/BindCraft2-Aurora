# BindCraft2 on ALCF Aurora

## Clone this Aurora port

```bash
git clone https://github.com/moeenmeigooni/BindCraft2-Aurora.git
cd BindCraft2-Aurora
```

## Port status

This is an Aurora/XPU port. It is a JAX/AlphaFold/ProteinMPNN workflow, not a
PyTorch workflow, so Intel Extension for PyTorch is not applicable. BC2's
declared `jax[oneapi]` extra is the intended accelerator route. Run the
included XPU smoke gate on an allocated tile before starting a campaign.

## PE 26.1810 runtime compatibility

The Intel JAX plugin wheel uses the oneAPI 2025 runtime ABI. On Aurora's
current PE 26.181.0 image, load the rebuilt PE 26.26.0 runtime with
`module load oneapi/release/2025.3.1` before running JAX workflows.
`aurora/runtime_env.sh` loads it by default; set `AURORA_JAX_ONEAPI_MODULE` to
override the module name. The launcher also adds the venv's `lib` directory so
the plugin can load the MKL/SYCL libraries installed with its Python packages.

The overlay makes three deliberate changes:

- Pins JAX 0.11.1 and Intel's matching 0.11.1-dev oneAPI PJRT plugin.  This is
  currently an Intel pre-release.  It does not lower JAX's `eigh` primitive,
  so AlphaFold's rotation-to-quaternion conversion uses an equivalent
  largest-component closed form built from basic XLA operations. Kabsch uses a
  Davenport-quaternion solver instead of SVD, and the opt-in
  collective-softness loss uses modified Gram--Schmidt plus a Newton--Schulz
  inverse instead of QR/LU. Aurora's current Level Zero stack does not accept
  the plugin's `SPV_KHR_bfloat16` output, so AlphaFold uses its supported fp32
  path on oneAPI. The wrapper also supplies an empty command-buffer enum
  override (the XLA representation for disabling it), because Aurora's Level
  Zero runtime does not implement command buffers. Aurora also uses the stock
  attention path instead of the CUDA fused-attention auto-probe, whose bf16
  probe would otherwise trigger the same SPIR-V incompatibility. CUDA CUB sort
  custom calls are replaced by stable residue-sized ranking expressions for
  contact selection, ProteinMPNN decoding, and neighbor selection. The XPU
  smoke gate exercises all fallback paths before a design is attempted.
- Keeps Aurora's `ZE_AFFINITY_MASK` untouched and uses local oneAPI device 0
  after the endpoint's tile mask.  The CUDA fan-out code now launches one
  worker per assigned XPU tile rather than querying `nvidia-smi` or packing
  seven processes onto an XPU.
- Redirects `HOME`, XDG, uv/pip, JAX compilation, Python-user, Matplotlib and
  model-weight state to sibling project-storage directories of the virtual
  environment.  The package itself now uses a temporary-directory cache rather
  than a home-directory fallback if launched outside this wrapper.

## Requested project deployment

Set `AURORA_PROJECT_ROOT` to a shared project directory visible from both the
login and compute nodes. By default, the scripts use
`/lus/flare/projects/FRAME-IDP/$USER`. The checkout can live anywhere on shared
storage; the environment defaults to
`${AURORA_PROJECT_ROOT}/envs/bindcraft2-aurora`.

From the checkout root, install on an Aurora login node with:

```bash
export AURORA_PROJECT_ROOT="/lus/flare/projects/FRAME-IDP/${USER}"
bash aurora/bootstrap_aurora.sh
```

The default weight path is project storage under
`${AURORA_PROJECT_ROOT}/envs/.bindcraft2-aurora-cache/weights`;
set `BINDCRAFT2_CACHE_ROOT` before the command to select another project-visible
location.  Use `--no-weights` for a software-only install.

Submit compute work using the supplied PBS scripts. They load `frameworks`,
preserve the endpoint's tile binding, and enforce the cache policy:

```bash
export AURORA_PROJECT_ROOT="/lus/flare/projects/FRAME-IDP/${USER}"
qsub aurora/smoke_xpu.pbs

# After the smoke log says "BindCraft2 Aurora XPU smoke passed."
qsub -v BINDCRAFT2_PROJECT_FOLDER=${AURORA_PROJECT_ROOT}/bindcraft2-runs/pdl1-first \
  aurora/design_one_trajectory.pbs
```

The smoke test must show a `oneapi` or `sycl` JAX device and pass its matrix,
quaternion-frame, Kabsch-alignment, orthonormal-basis, and inverse checks
before a design run. First use a bounded one-trajectory campaign; a
low-confidence or rejected candidate is a valid integration result, not grounds
for lowering BC2's filters.

The campaign script defaults to a job-specific directory under
`${AURORA_PROJECT_ROOT}/bindcraft2-runs` when
`BINDCRAFT2_PROJECT_FOLDER` is not supplied. Model state, compilation cache,
temporary files, PBS logs, and package caches remain under the project-scoped
environment cache, never in the real home directory.

For implementation testing only, `design_one_trajectory.pbs` accepts one
additional `--set` assignment through `BINDCRAFT2_EXTRA_SET`. For example,
`min_iptm_final=0` relaxes the *post-refold* i_pTM gate for a known trajectory,
so the downstream ProteinMPNN, refolding, filtering, and ranking writers can
be tested. Never use a result from a relaxed test as a binder-design claim.
