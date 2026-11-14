# Installing rvfitter

The easiest route is the standalone app: no Python needed.
Use the Python package for scripting, batch runs or the `inject` and `forecast` tools.
rvfitter is **not yet on PyPI**, so the Python routes install it from GitHub; after the PyPI release the same commands take plain `"rvfitter[all]"`.
`[all]` adds the GUI (PySide6) and the MCMC method (emcee, corner); leave it out for a smaller, command-line-only install.
Check any install with `rvfitter doctor`.

## Standalone app

Each [GitHub release](https://github.com/Will-Cooper/rvfitter/releases/latest) has a ready-to-run GUI for each system:

| System | Download | Run |
|---|---|---|
| Windows (64-bit) | `rvfitter-gui-windows-x86_64.zip` | unzip, then double-click `rvfitter-gui\rvfitter-gui.exe` |
| macOS (Apple Silicon) | `rvfitter-gui-macos-arm64.zip` | unzip, then right-click `rvfitter-gui.app` > Open (the first time only: it is not signed). If macOS says it is damaged, run `xattr -dr com.apple.quarantine rvfitter-gui.app` once |
| Linux (x86-64) | `rvfitter-gui-linux-x86_64.tar.gz` | `tar -xzf rvfitter-gui-linux-x86_64.tar.gz && ./rvfitter-gui/rvfitter-gui` |

Each download also contains `examples/` (see `examples/README.md`).

## uv (recommended for Python)

[uv](https://docs.astral.sh/uv/) installs Python itself too.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh                                   # macOS / Linux
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" # Windows

uv tool install "rvfitter[all] @ git+https://github.com/Will-Cooper/rvfitter@main"   # rvfitter on your PATH
uvx --from "rvfitter[all] @ git+https://github.com/Will-Cooper/rvfitter@main" rvfitter gui   # run once
```

Installing from GitHub needs `git`.
If `rvfitter` is not found afterwards, run `uv tool update-shell` and open a new terminal.
`pipx install "rvfitter[all] @ git+https://github.com/Will-Cooper/rvfitter@main"` works the same way.

## pip or conda

Any Python 3.11 or later works.
With conda, [Miniforge](https://github.com/conda-forge/miniforge) needs no admin rights and leaves your system Python alone (on Apple Silicon use the `arm64` installer):

```bash
conda create -n rvfitter -c conda-forge python=3.12
conda activate rvfitter        # repeat in every new terminal (Windows: use Miniforge Prompt)
pip install "rvfitter[all] @ git+https://github.com/Will-Cooper/rvfitter@main"
rvfitter doctor
```

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

**Qt on Wayland (Linux).**
The conda-forge PySide6 build lacks the Qt Wayland plugin, so the GUI runs through XWayland.
For native Wayland, in that environment run `conda remove --force pyside6` (if conda installed it), then `pip install --force-reinstall "PySide6>=6.6"`.
The standalone app already includes the plugin.

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
Each BT-Settl spectrum from SVO is about 60 MB at full resolution, so download only the Teff/log g range you need.
If a server is down, copy a grid obtained elsewhere into `<cache>/<grid>/`, or point to any folder with `local_dir = "/path/to/grid"` under `[templates]` in your config (`rvfitter init spectrum.fits` writes one to edit).

**Plots and LaTeX.** By default (`plot_style = "quick"` under `[output]`) rvfitter draws its figures with bundled Computer Modern fonts; no LaTeX installation is needed.
For paper-ready figures typeset by LaTeX, set `plot_style = "paper"`; this needs `latex` and `dvipng` (TeX Live, MiKTeX or MacTeX), and rvfitter stops with a clear message if they are missing.

**Telluric model.** The optional telluric methods need a one-off, about 90 MB build: `rvfitter templates telluric --build`.

## Update

```bash
conda activate rvfitter
pip install --upgrade --force-reinstall --no-deps "rvfitter @ git+https://github.com/Will-Cooper/rvfitter@main"
```

After the PyPI release: `pip install --upgrade "rvfitter[all]"`.
With uv: `uv tool upgrade rvfitter`. With pipx: `pipx upgrade rvfitter`.
Standalone app: download the new archive from the [latest release](https://github.com/Will-Cooper/rvfitter/releases/latest) and replace the old folder.
Then run `rvfitter doctor`.

## Uninstall

```bash
conda deactivate
conda env remove -n rvfitter
```

With uv: `uv tool uninstall rvfitter`. With pipx: `pipx uninstall rvfitter`. Standalone app: delete its folder.
To delete the downloaded templates, remove the cache folder shown by `rvfitter doctor`.
To remove Miniforge itself, delete its folder (`~/miniforge3`, or use "Uninstall" in the Windows Start menu).
