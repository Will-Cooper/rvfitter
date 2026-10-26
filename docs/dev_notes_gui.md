# GUI dev notes (P4)

- `gui/backend.py`: `Backend` protocol plus `LazyBackend`, the only place that touches `config`, `archives`, `io`, `regions`, `pipeline`, `results` and `plotting`. Tests use `tests/gui_fakes.py`.
- `pipeline.run(cfg, inputs, *, n_procs)` has no per-object callback, so `LazyBackend.run` calls it once per object group (objects are sequential anyway). Cancel is cooperative, between objects.
- Settings form (`gui/forms.py`) is generated from the Config dataclass fields and type hints; field `metadata={"help": ..., "choices": [...]}` overrides tooltip and choices. Fields named `methods` and `formats` (list[str]) get check boxes; a nested dataclass of bools also works. `n_procs`, `output_dir`/`out_dir` and `formats` are read from anywhere in the tree by leaf name.
- Region overrides contract (needs agreement with P1/P3): `config.regions.overrides` is `dict[str, dict]` keyed by region name: `{"wmin": .., "wmax": ..}` moves an automatic region, `{"remove": true}` drops one, `{"kind": .., "wmin": .., "wmax": ..}` under a new name adds a user region. The editor calls `regions.select(..., overrides=None)` to obtain the automatic regions and applies the overrides itself.
- Plot functions are called as `plotting.plot_*(result)` (plus `show=False` if accepted); None or an exception gives a placeholder.
- `LazyBackend.group_objects` uses `io.read_spectrum(path)` and `archives.group_by_object(spectra)`; the `Spectrum.source` string maps back to the file.
