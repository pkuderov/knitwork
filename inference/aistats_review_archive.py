"""Prepare local anonymous review code without publishing it or launching training."""

import ast
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import zipfile

import yaml

if __package__:
    from .aistats_revision_evidence import MID_KEYS
else:
    from aistats_revision_evidence import MID_KEYS


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'article/supplementary/mosaic_review_code.zip'


def main():
    tracked = subprocess.check_output(['git', 'ls-files', '-z', 'knitwork'], cwd=ROOT).decode().split('\0')
    directories = ('knitwork/common/', 'knitwork/models/', 'knitwork/gens/', 'knitwork/visualization/')
    files = {}
    baseline_modules = {'delta_net', 'hgrn2', 'mlstm', 'rims', 'transformer'}
    for name in tracked:
        if name.startswith('knitwork/models/baseline/') and Path(name).stem not in baseline_modules:
            continue
        if name.endswith('.py') and (name.startswith(directories) or name == 'knitwork/exps/text/run.py'):
            files[name] = (ROOT / name).read_bytes()
    registry_path = 'knitwork/models/utils.py'
    source = files[registry_path].decode()
    assignment = next(node for node in ast.parse(source).body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'REGISTRY' for target in node.targets))
    registry = ast.literal_eval(assignment.value)
    allowed_types = {max((model for model in registry if key == model or key.startswith(model + '_')), key=len) for key in MID_KEYS.values()}
    lines = source.splitlines(keepends=True)
    retained_registry = {key: value for key, value in registry.items() if key in allowed_types}
    replacement = 'REGISTRY = ' + json.dumps(retained_registry, indent=4) + '\n'
    files[registry_path] = (''.join(lines[:assignment.lineno - 1]) + replacement + ''.join(lines[assignment.end_lineno:])).encode()
    config_path = 'knitwork/exps/text/config/mid.yaml'
    config = yaml.safe_load((ROOT / config_path).read_text())
    for key in list(config):
        is_model = any(key == model or key.startswith(model + '_') for model in registry)
        if is_model and key not in MID_KEYS.values() and not key.startswith('grnn_L2C4_topk'):
            del config[key]
    config['log']['logger'] = None
    config['log'].pop('project', None)
    config['gens']['text8']['path'] = 'text8.txt'
    files[config_path] = yaml.safe_dump(config, sort_keys=False).encode()
    files['README.md'] = (ROOT / 'article/supplementary/README.md').read_bytes()
    dependencies = ['numpy>=2.0', 'torch>=2.1', 'numba>=0.61', 'matplotlib', 'scipy', 'ryaml>=0.5.1', 'PyYAML', 'python-dotenv>=1.0', 'gymnasium>=1.1', 'popgym>=1.0.6']
    files['requirements.txt'] = ('\n'.join(dependencies) + '\n').encode()
    packages = [dependency.split('>=')[0] for dependency in dependencies]
    files['environment.txt'] = ('\n'.join(f'{package}=={version(package)}' for package in packages) + '\n').encode()
    forbidden = [b'pkuderov', b'annenkov', b'kuderov', b'team-rl-exp', b'knitwork-aistat', b'/home/master', b'/storage/']
    for name, content in files.items():
        if any(token in content.lower() for token in forbidden):
            raise ValueError('Private identifier found in ' + name)
    manifest = {name: hashlib.sha256(content).hexdigest() for name, content in sorted(files.items())}
    files['MANIFEST.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DESTINATION, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo('mosaic-review-code/' + name, date_time=(2020, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    print(f'Prepared {DESTINATION.relative_to(ROOT)}: {len(files)} files; no publication or training.')


if __name__ == '__main__':
    main()
