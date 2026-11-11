from prm800k.cli import parser


def test_all_stages_exposed():
    help_text=parser().format_help()
    for name in ['demo','train','score','generate','finetune','ablate','evaluate','index','audit']:
        assert name in help_text


def test_legacy_help_does_not_trigger_download(capsys):
    import pytest
    from prm800k.eval.eval import legacy_main
    with pytest.raises(SystemExit) as exit_info:
        legacy_main(['--help'])
    assert exit_info.value.code==0
    assert '--population' in capsys.readouterr().out
