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
    assert len(logging.getLogger('rvfitter').handlers) == 2
    assert logging.getLogger('matplotlib').level >= logging.WARNING
    assert logging.getLogger('astropy').level >= logging.WARNING


def test_console_shows_a_repeated_message_once(capsys):
    logging_setup.configure(0)
    lg = logging.getLogger('rvfitter.t')
    for msg in ('a.fits: no unit', 'a.fits: no unit', 'b.fits: no unit'):
        lg.warning(msg)
    out = capsys.readouterr().out
    assert out.count('a.fits: no unit') == 1 and out.count('b.fits: no unit') == 1


def test_once_filter_keeps_errors_and_resets():
    f = logging_setup.OnceFilter()

    def rec(level, msg):
        return logging.LogRecord('rvfitter.t', level, __file__, 1, msg, None, None)
    assert f.filter(rec(logging.WARNING, 'w')) and not f.filter(rec(logging.WARNING, 'w'))
    assert f.filter(rec(logging.ERROR, 'export failed')) and f.filter(rec(logging.ERROR, 'export failed'))
    f.reset()  # e.g. the GUI log pane was cleared
    assert f.filter(rec(logging.WARNING, 'w'))


def test_demoted(caplog):
    lg = logging.getLogger('rvfitter.t')
    with caplog.at_level(logging.DEBUG, logger='rvfitter'):
        with logging_setup.demoted():
            lg.info('rerun info')
            lg.warning('rerun warning')
            lg.error('rerun error')
        lg.info('main info')
    assert [(r.getMessage(), r.levelno) for r in caplog.records] == [
        ('rerun info', logging.DEBUG), ('rerun warning', logging.DEBUG), ('rerun error', logging.ERROR),
        ('main info', logging.INFO)]
