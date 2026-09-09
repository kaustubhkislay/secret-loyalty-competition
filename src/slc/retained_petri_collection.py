"""Verify sealed audit evidence before local extraction."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile


def save_bytes(path, data):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'existing evidence differs: {path.name}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(data)
    temporary.replace(path)


def source_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or path.suffix != '.py':
        raise ValueError('unsafe retry source path')
    return path


def save_bound_source(destination, name, data, expected):
    relative = source_path(name)
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('retry source hash differs from the frozen request')
    save_bytes(Path(destination) / 'source' / relative, data)


def unpack_evidence(data, evidence, destination):
    if hashlib.sha256(data).hexdigest() != evidence['sha256']:
        raise ValueError('archive hash differs from sealed evidence')
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        files = {}
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or not member.isfile() or member.name in files:
                raise ValueError('unsafe or duplicate archive path')
            files[member.name] = archive.extractfile(member).read()
    manifest = json.loads(files['MANIFEST.json'])
    if set(files) != set(manifest) | {'MANIFEST.json'}:
        raise ValueError('archive contents differ from manifest')
    for name, expected in manifest.items():
        if hashlib.sha256(files[name]).hexdigest() != expected:
            raise ValueError('evidence file hash differs from manifest')
    for name, payload in files.items():
        save_bytes(Path(destination) / name, payload)
    return manifest
