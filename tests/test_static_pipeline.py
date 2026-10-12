"""Pipeline guards protect write-once results while notebook stages resume."""
import pytest

from mtgdeck import static_pipeline as pipeline
from mtgdeck import static_v4 as v4


def test_plan_reuses_adapted_model_and_wires_rules_hyperedges_and_probes(tmp_path):
    plan = pipeline.build_plan(tmp_path)
    commands = [stage['command'] for stage in plan]
    assert '--view' in commands[0] and commands[0][commands[0].index('--view') + 1] == 'rules_only'
    assert any('mlm_synthetic_cards_v1/model' in arg.replace('\\', '/') for arg in commands[0])
    assert [cmd[cmd.index('--stage') + 1] for cmd in commands] == [
        'export', 'evaluate', 'train', 'evaluate', 'evaluate', 'export', 'evaluate']
    assert commands[2][commands[2].index('--prepared') + 1].endswith('spellbook')
    assert '--probes' in commands[-1] and '--spellbook' in commands[-1]
    assert len({stage['output'] for stage in plan}) == len(plan)
    with pytest.raises(ValueError, match='protected'):
        pipeline.build_plan(tmp_path, 'oracle')


def test_reuse_rejects_changed_inputs_configuration_and_incomplete_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'preflight', lambda root: {'ready': True})
    source = tmp_path / 'source.json'
    source.write_text('{}')
    output = tmp_path / 'result'
    with v4.stage(output, {'stage': 'test', 'seed': 42}, [source]):
        pass
    plan = [{'output': str(output), 'expected_config': {'stage': 'test', 'seed': 42}}]
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda *a, **k: pytest.fail('Completed stage was rerun'))
    pipeline.run_pipeline(plan, tmp_path)
    plan[0]['expected_config']['seed'] = 43
    with pytest.raises(ValueError, match='configuration'):
        pipeline.run_pipeline(plan, tmp_path)
    plan[0]['expected_config']['seed'] = 42
    source.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='lineage'):
        pipeline.run_pipeline(plan, tmp_path)
    incomplete = tmp_path / 'incomplete'
    incomplete.mkdir()
    plan[0]['output'] = str(incomplete)
    with pytest.raises(FileExistsError, match='Incomplete stage'):
        pipeline.run_pipeline(plan, tmp_path)
