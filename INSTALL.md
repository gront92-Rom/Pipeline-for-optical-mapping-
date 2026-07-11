# Installation Guide — Cardiac Optical Mapping Pipeline v3

Quick-start to run `run_cardiac.sh` on a new machine.

---

## 1. Prerequisites

- Linux / WSL / macOS (bash + coreutils)
- Python 3.12 (tested on 3.12.3)
- `git`, `python3-dev`, `g++` (needed to build optimap C++ extension)
- A C compiler and `numpy` headers must be present before building optimap

### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install -y git python3.12 python3.12-dev python3-pip build-essential
```

---

## 2. Clone / copy the repository

The repo must keep its internal layout:

```
cardiac_pipeline_v3/
├── run_cardiac.sh
├── requirements.txt
├── install_optimap_dev.sh
├── src/              # Python package (auto-added to PYTHONPATH)
├── config/           # default.yaml and overrides
├── data/             # put recordings here
└── results/          # created automatically
```

No absolute paths are required; the script resolves its own directory.

---

## 3. Install Python dependencies

### 3.1. Exact optimap dev build (REQUIRED)

The pipeline needs a pre-release `optimap` snapshot that is **not** on PyPI.

```bash
cd cardiac_pipeline_v3
chmod +x install_optimap_dev.sh
./install_optimap_dev.sh
```

This installs commit `52c156ebb` (version `0.3.2.dev47+g52c156ebb.d20260501`).

### 3.2. Remaining pinned dependencies

```bash
python3 -m pip install -r requirements.txt
```

---

## 4. Place data

Create a subdirectory under `data/` named by sample ID and put the MiCAM files
there. The script auto-detects `.rsh`, `.gsh`, `.rsd`, and `.gsd` files.

Example:

```bash
mkdir -p data/004A
cp /mnt/d/your_data/004A/*.rsh /mnt/d/your_data/004A/*.rsd data/004A/
# For partitioned recordings, copy ALL chunks + the .gsd companion file.
```

---

## 5. Run

```bash
./run_cardiac.sh 004A
./run_cardiac.sh --debug 004A    # full traceback + debug_report.md on failure
```

Results go to `results/004A/`.

---

## 6. Portability notes / known constraints

| Item | Status | Notes |
|---|---|---|
| Repo-relative paths | ✅ Portable | Script locates itself via `BASH_SOURCE` |
| Temp driver script | ✅ Fixed | Now written to repo-local `tmp/`, not `/tmp` |
| Python dependencies | ⚠️ Pinned | `requirements.txt` + exact optimap commit |
| MiCAM file format | ⚠️ Hardcoded | Assumes ULTIMA sensor width 128 px, crop 20/8, pixel 0.085 mm |
| skimage deprecation | ⚠️ Future risk | `binary_opening`/`binary_closing` deprecated in 0.28 |
| Windows native CMD | ❌ Not supported | Use WSL2 or Docker |

If you move to a different microscope or camera, update `config/default.yaml`
(`loader.crop_left`, `loader.crop_right`, `pixel_size_mm`).

---

## 7. Docker (fully reproducible)

If you want zero host dependency issues, build the Docker image:

```bash
docker build -t cardiac-pipeline-v3 .
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/results:/app/results" cardiac-pipeline-v3 004A
```

See `Dockerfile` for details.

---

## 8. Verify installation

After setup, run the bundled smoke test:

```bash
./run_cardiac.sh --debug 004A
```

Expected: 9/9 stages PASS, total ~10 s on sample 004A.
