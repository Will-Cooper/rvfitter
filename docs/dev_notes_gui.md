# GUI dev notes (P4)

- `gui/backend.py`: `Backend` protocol plus `LazyBackend`, the only place that touches `config`, `archives`, `io`, `regions`, `pipeline`, `results` and `plotting`. Tests use `tests/gui_fakes.py`.
- `LazyBackend.run` passes `on_object` and `cancel` to `pipeline.run` when it accepts them, else falls back to one call per object (cancel then acts between objects).
- Settings form (`gui/forms.py`) is generated from the Config dataclass fields and type hints; field `metadata={"help": ..., "choices": [...]}` overrides tooltip and choices. Fields named `methods` and `formats` (list[str]) get check boxes; a nested dataclass of bools also works. `n_procs`, `output_dir`/`out_dir` and `formats` are read from anywhere in the tree by leaf name.
- Region overrides follow Addendum 2: `{method: [{name, kind, wmin, wmax[, centre, continuum]} | {name, remove: true}]}`. The editor diffs the regions in force against the automatic ones (both from `regions.select`) and never re-implements selection.
- Plot functions are called as `plotting.plot_*(result)` (plus `show=False` if accepted); None or an exception gives a placeholder.
- `LazyBackend.group_objects` uses `io.read_spectrum(path)` and `archives.group_by_object(spectra)`; the `Spectrum.source` string maps back to the file.
