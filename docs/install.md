# Installing rvfitter

rvfitter v2 is installed with `pip` into a conda environment.
It is **not yet on PyPI**, so use the GitHub command marked "now" below.
When it is released, use the "after release" command instead; everything else is identical.

| | Command |
|---|---|
| Now (development branch) | `pip install "rvfitter[all] @ git+https://github.com/Will-Cooper/rvfitter@v2-modernise"` |
| After release on PyPI | `pip install "rvfitter[all]"` |

`[all]` adds the GUI (PySide6) and the MCMC method (emcee, corner).
Leave it out for a smaller, command-line-only install.

## Option 1 (recommended): Miniforge + conda environment

Miniforge is a free Python installer that needs no admin rights and does not touch your system Python.

### 1. Install Miniforge

Download the installer for your system, then run it.

- **Windows:** <https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe>.
  Double-click it, accept the defaults, then open **Miniforge Prompt** from the Start menu.
  Use that prompt for every command below.
- **macOS, Apple Silicon (M1 or later):** <https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh>
- **macOS, Intel:** <https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh>
- **Linux, x86-64:** <https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh>
- **Linux, ARM:** <https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh>

On macOS and Linux, open a terminal in the folder with the download and run (use the file name you downloaded):

```bash
bash Miniforge3-Linux-x86_64.sh
```

Press Enter through the licence, answer `yes` to the questions (including "initialize conda"), then **close and reopen the terminal**.
The prompt now starts with `(base)`.

### 2. Create the environment

Windows (Miniforge Prompt), macOS and Linux:

```bash
conda create -n rvfitter -c conda-forge python=3.12
```

Type `y` when asked.
Expected: the last lines say `# To activate this environment, use #     $ conda activate rvfitter`.

```bash
conda activate rvfitter
```

Expected: the prompt changes from `(base)` to `(rvfitter)`.
Repeat this `conda activate` line in every new terminal before using rvfitter.

### 3. Install rvfitter

```bash
pip install "rvfitter[all] @ git+https://github.com/Will-Cooper/rvfitter@v2-modernise"
```

This takes a few minutes and ends with `Successfully installed ... rvfitter-2.0.0.dev0 ...`.

### 4. Check it

```bash
rvfitter --version
rvfitter doctor
```

Expected:

```text
rvfitter 2.0.0.dev0
OK    Python >= 3.11           3.12.15
OK    import numpy             2.5.4
...
OK    import PySide6           6.9.2
OK    import emcee             3.1.6
OK    import corner            2.3.0
OK    Qt platform plugin       eglfs, linuxfb, minimal, ..., offscreen, xcb
OK    Template cache writable  /home/you/.cache/rvfitter/templates
OK    SVO reachable            https://svo2.cab.inta-csic.es/theory/newov2/index.php
OK    Zenodo reachable         https://zenodo.org
All checks passed
```

Versions and the plugin list differ by system.
Each failing line prints `FAIL ... -> fix: <what to run>`, and the command exits with status 1.
Add `--offline` to skip the two network checks.

## Option 2: pipx (command line only)

Use this if you do not need the GUI and already have Python 3.11 or later.
pipx puts each tool in its own hidden environment.

```bash
python -m pip install --user pipx
python -m pipx ensurepath
```

Close and reopen the terminal, then:

```bash
pipx install "rvfitter @ git+https://github.com/Will-Cooper/rvfitter@v2-modernise"
rvfitter --version
```

To include the GUI and MCMC, use `"rvfitter[all] @ git+..."` instead.
After release: `pipx install rvfitter`.
On Windows, use `py -m pip` in place of `python -m pip` if `python` is not found.

## Your first RV in 3 commands

Activate the environment first (`conda activate rvfitter`).

**1. Download template spectra** (model atmospheres; only needed for the `template_fit`, `ccf` and `forward_mcmc` methods):

```bash
rvfitter templates download --grid bt-settl-cifist --teff 1500 2500 --logg 4.5 5.5
```

The flags are `--grid` (one of `bt-settl-agss2009`, `bt-settl-cifist`, `lowz`, `sand`, `sonora-bobcat`, `sonora-diamondback`, `sonora-elfowl`), `--teff MIN MAX` [K], `--logg MIN MAX`, `--met MIN MAX`, `--alpha MIN MAX`, `--range AXIS MIN MAX` for any other axis, `--dir DIR` (default: the cache, see `rvfitter doctor`) and `--source {auto,svo,zenodo}`.
The defaults are `--teff 1200 3000 --logg 4.5 5.5`, `auto` meaning SVO for the BT-Settl grids and Zenodo for the others.
Check what you have with `rvfitter templates list`.
Zenodo grids are large single archives (the download resumes if repeated), so choose a narrow range.
If this command fails, see the FAQ below; you can still run the `linecentre` method, which needs no templates.

**2. Measure.** `my_spectra.zip` can equally be one spectrum file or a folder:

```bash
rvfitter measure my_spectra.zip --out results.csv
```

Expected (abridged): one block per object, ending with

```text
2MASSJ0453-1751_tellcorr: adopted RV 16.26 ± 8.73 km/s
Wrote results.csv
Wrote results_details.csv
Wrote results_config.toml
```

`results.csv` has one row per object (see the README for the columns), `results_details.csv` one row per region and method, and `results_config.toml` the exact settings, which `rvfitter run results_config.toml` repeats.
Useful options: `--methods template_fit,ccf,linecentre`, `--no-plots`, `--telluric-corrected` (the spectra are already telluric-corrected), `--bary {auto,apply,skip}`, `--medium {auto,air,vacuum}`, `--config FILE`, `--n-procs N`, `--format csv,json,tsv,html`.
Exit status is 0 if clean, 1 on an error and 2 if the run finished but a file or method failed.

**3. Open the GUI:**

```bash
rvfitter gui
```

A window opens to load spectra, edit regions and run the methods.

## Troubleshooting

**`rvfitter: command not found` (or "not recognised").**
The environment is not active.
Run `conda activate rvfitter` (Windows: in Miniforge Prompt, not PowerShell or cmd).

**The GUI does not start on a remote or headless machine** (`could not connect to display`, `Could not load the Qt platform plugin "xcb"`).
Over SSH use X forwarding (`ssh -X host`), or run without a display:

```bash
QT_QPA_PLATFORM=offscreen rvfitter gui
```

That only makes sense for testing; on a server use the CLI.
On Linux, `xcb` needs system libraries; if `rvfitter doctor` reports a missing plugin, run `sudo apt install libxcb-cursor0 libxkbcommon-x11-0` (Debian/Ubuntu) or ask your administrator.
On Windows use `set QT_QPA_PLATFORM=offscreen`.

**Proxies, firewalls and offline machines.**
Set the proxy before running (Linux/macOS: `export HTTPS_PROXY=http://proxy:8080`; Windows: `set HTTPS_PROXY=http://proxy:8080`).
With no internet, download the grid files on another machine, copy them to a folder `<cache>/<grid>/` (for example `~/.cache/rvfitter/templates/bt-settl-cifist/`), and check them with `rvfitter templates verify <folder> --grid bt-settl-cifist`.
Set `RVFITTER_CACHE` to use another cache folder.
`rvfitter doctor --offline` skips the network checks.

**Apple Silicon.**
Use the `arm64` Miniforge installer above, not the Intel one; the Intel build runs under emulation and PySide6 may fail to install.
`python -c "import platform; print(platform.machine())"` should print `arm64`.

**Windows paths.**
Quote paths with spaces: `rvfitter measure "C:\My Data\spectra.zip" --out "C:\My Data\results.csv"`.
Avoid trailing backslashes inside quotes.
The cache is under `%LOCALAPPDATA%\rvfitter`.
Long paths (over 260 characters) can fail; unpack the data near the drive root.

**Template sources (SVO and Zenodo).**
The BT-Settl grids come from the Spanish Virtual Observatory theory server (SVO, https://svo2.cab.inta-csic.es/theory/newov2/), one spectrum per request.
Sonora Bobcat, Diamondback, Elf Owl and SAND come from Zenodo as whole files (CC-BY-4.0; cite the papers in `rvfitter.templates.KNOWN_GRIDS`).
`--source` forces one.
Known issue in this development version: `templates download` from SVO currently stops with `ValueError: unrecognised SSAP response columns`, and `templates list --available` reports the BT-Settl grids as "NOT available", because SVO changed its reply format.
Until that is fixed, use `--grid sonora-bobcat --source zenodo`, or copy a BT-Settl grid obtained elsewhere into `<cache>/bt-settl-cifist/` as above and set `[templates] grid` in your config accordingly.

**Plots mention LaTeX / fonts.** rvfitter bundles its fonts; no LaTeX installation is needed.

**Telluric model.** The optional telluric methods need a one-off, about 90 MB build: `rvfitter templates telluric --build`.

## Update

```bash
conda activate rvfitter
pip install --upgrade --force-reinstall --no-deps "rvfitter @ git+https://github.com/Will-Cooper/rvfitter@v2-modernise"
```

After release: `pip install --upgrade "rvfitter[all]"`.
With pipx: `pipx upgrade rvfitter`.
Then run `rvfitter doctor`.

## Uninstall

```bash
conda deactivate
conda env remove -n rvfitter
```

With pipx: `pipx uninstall rvfitter`.
To delete the downloaded templates, remove the cache folder shown by `rvfitter doctor`.
To remove Miniforge itself, delete its folder (`~/miniforge3`, or use "Uninstall" in the Windows Start menu).
