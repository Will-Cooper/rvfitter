"""Shared fixtures: every test sees the tiny synthetic telluric model unless it asks for the real one."""
import os

import pytest

from telluric_synth import write_model


@pytest.fixture(scope='session', autouse=True)
def synthetic_telluric_model(tmp_path_factory):
    """Point rvfitter at the synthetic telluric model (the HITRAN model is built on demand, never shipped)"""
    from rvfitter.calibration import telluric
    path = write_model(tmp_path_factory.mktemp('telluric') / 'telluric_synthetic.npz')
    old = os.environ.get('RVFITTER_TELLURIC_MODEL')
    os.environ['RVFITTER_TELLURIC_MODEL'] = str(path)
    telluric.reload()
    yield path
    if old is None:
        os.environ.pop('RVFITTER_TELLURIC_MODEL', None)
    else:
        os.environ['RVFITTER_TELLURIC_MODEL'] = old
    telluric.reload()


@pytest.fixture
def real_telluric_model(monkeypatch):
    """The HITRAN model built in the user cache (`rvfitter templates telluric --build`); skips if it is absent"""
    from rvfitter.calibration import telluric
    monkeypatch.delenv('RVFITTER_TELLURIC_MODEL', raising=False)
    if not telluric.available():
        pytest.skip(f'the telluric model is not built ({telluric.BUILD_COMMAND})')
    return telluric.model()
