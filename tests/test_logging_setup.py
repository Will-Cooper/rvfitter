import logging

from rvfitter import logging_setup


def test_stdout_and_file(tmp_path, capsys):
    logfile = tmp_path / 'sub' / 'x.log'
    logging_setup.configure(0, logfile)
    lg = logging.getLogger('rvfitter.test')
    lg.info('hello info')
    lg.debug('hidden debug')
    out = capsys.readouterr()
    assert 'hello info' in out.out and 'hidden debug' not in out.out and not out.err
    text = logfile.read_text()
    assert 'hello info' in text and 'hidden debug' in text and text.startswith('20')


def test_levels(capsys):
    logging_setup.configure(1)
    logging.getLogger('rvfitter.t').debug('dbg')
    assert 'dbg' in capsys.readouterr().out
    logging_setup.configure(-1)
    logging.getLogger('rvfitter.t').info('inf')
    logging.getLogger('rvfitter.t').warning('wrn')
    out = capsys.readouterr().out
    assert 'inf' not in out and 'wrn' in out


def test_idempotent_and_quiet(tmp_path):
    import astropy  # noqa: F401
    import matplotlib  # noqa: F401

    for _ in range(3):
        logging_setup.configure(0, tmp_path / 'a.log')
    own = [h for h in logging.getLogger('rvfitter').handlers if getattr(h, '_rvfitter_handler', False)]
    assert len(own) == 2
    assert logging.getLogger('matplotlib').level >= logging.WARNING
    assert logging.getLogger('astropy').level >= logging.WARNING
