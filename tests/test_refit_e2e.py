"""End-to-end reproducibility of refit_object: run, refit one object, dump the TOML, rerun from scratch."""
import numpy as np

from rvfitter import config, pipeline, results
from rvfitter import templates as T
from rvfitter.methods import _common as C
from test_grids_nd import _grid4


def _write_object(path, grid, params, rv, seed):
    tpl = C.build_template(C.Candidate('truth', grid=grid, params=params), 3000., (7600., 9000.))
    wave = np.exp(np.arange(np.log(7650.), np.log(8950.), 1. / 3000. / 2.5))
    flux = tpl.at(wave, rv)
    sigma = flux / 150.
    flux = flux + np.random.default_rng(seed).normal(0., 1., len(wave)) * sigma
    np.savetxt(path, np.c_[wave, flux, sigma], delimiter=',', header='wave,flux,err', comments='')
    return path


def test_refit_reproduced_byte_for_byte(tmp_path):
    T.clear_cache()
    grid_dir = _grid4(tmp_path / 'grid')
    grid = T.TemplateGrid(grid_dir)
    files = [str(_write_object(tmp_path / 'objA.csv', grid, (1900., 5.0, 0., 0.), 12., 1)),
             str(_write_object(tmp_path / 'objB.csv', grid, (1900., 5.0, -0.5, 0.), -30., 2))]
    cfg = config.Config()
    cfg.seed = 7
    cfg.input.medium = 'vacuum'
    cfg.input.resolution = 3000.
    cfg.velocity_frame.correction = 'skip'
    cfg.templates.local_dir = str(grid_dir)
    cfg.templates.grid = 'toy4'
    cfg.templates.teff_range = (1800., 2000.)
    cfg.methods.enabled = ['template_fit', 'ccf']
    cfg.methods.options = {'template_fit': {'segment_res': 0.}, 'ccf': {'n_mc': 20}}
    cfg.regions.template_windows = 'full'
    cfg.calibration.wavelength_correction = 'none'

    first = pipeline.run(cfg, files, n_procs=1)
    assert [r.object for r in first] == ['objA', 'objB']
    patch = {'templates': {'met_range': [-1.0, 0.0]}}
    refit_b, cfg2 = pipeline.refit_object(first[1], cfg, patch)
    n_before = len(first[1].methods['template_fit'].measurements[0].diagnostics['template_scores'])
    n_after = len(refit_b.methods['template_fit'].measurements[0].diagnostics['template_scores'])
    assert n_after == 3 * n_before  # met now searched over -1, -0.5, 0
    table_refit = results.to_table([first[0], refit_b]).to_csv(index=False)

    toml = tmp_path / 'refit.toml'
    config.dump(cfg2, toml)
    rerun = pipeline.run(config.load(toml), files, n_procs=1)
    table_rerun = results.to_table(rerun).to_csv(index=False)
    assert table_rerun == table_refit
    # the patch changed only objB; objA is identical to the first run
    assert results.to_table(first).iloc[0].equals(results.to_table(rerun).iloc[0])
    # and objB alone (a different input set) gives the same object result
    alone = pipeline.run(config.load(toml), files[1:], n_procs=1)
    assert results.to_table(alone).to_csv(index=False).splitlines()[1] == table_rerun.splitlines()[2]
