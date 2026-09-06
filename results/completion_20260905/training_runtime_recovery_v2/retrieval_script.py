"""Read existing Modal training metadata without creating apps, images, or tasks.

The allowlisted RPCs only retrieve metadata. Output is create-only and omits
client configuration, authentication, URLs, source code, and function payloads.
Image metadata is evidence about an image, not a historical runtime inventory.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

from google.protobuf.json_format import MessageToDict
from modal.client import _Client
from modal_proto import api_pb2


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def object_metadata(value):
    return {'object_id': value.object_id,
            'function': {name: getattr(value.function_handle_metadata, name)
                         for name in ('function_name', 'definition_id', 'app_id')}}


async def retrieve(args):
    root = args.out_dir
    root.mkdir(parents=True, exist_ok=False)
    with (root / 'retrieval_script.py').open('xb') as stream:
        stream.write(Path(__file__).read_bytes())
    outcomes = json.loads(args.training_outcomes.read_text())
    children = [row['function_call_id'] for row in outcomes]
    save(root / 'REQUEST.json', {
        'app_id': args.app_id, 'parent_call_id': args.parent_call_id,
        'child_call_ids': children, 'observed_at': datetime.now(timezone.utc).isoformat(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'modal_sdk_version': importlib.metadata.version('modal'),
        'training_outcomes_sha256': hashlib.sha256(args.training_outcomes.read_bytes()).hexdigest(),
        'scope': 'Read-only metadata. No GPU, training, inference, or container execution.',
        'serialization': 'Allowlisted object identity fields; image package metadata if available. No client/auth objects.',
    })
    try:
        client = await asyncio.wait_for(_Client.from_env(), timeout=30)
    except Exception as exc:
        save(root / 'CLIENT_ERROR.json', {'status': 'error', 'error_type': type(exc).__name__})
        print(json.dumps({'out_dir': str(root), 'status': 'client_initialization_error',
                          'error_type': type(exc).__name__}))
        return
    results = []

    async def call(method, request, select):
        row = {'rpc': method, 'request': MessageToDict(request, preserving_proto_field_name=True),
               'observed_at': datetime.now(timezone.utc).isoformat()}
        response = None
        try:
            response = await asyncio.wait_for(getattr(client.stub, method)(request), timeout=30)
            row.update(status='ok', response=select(response))
        except Exception as exc:
            status = getattr(exc, 'status', None)
            row.update(status='error', error_type=type(exc).__name__,
                       rpc_status=getattr(status, 'name', None))
        save(root / f'{len(results):03d}_{method}.json', row)
        results.append(row)
        return response

    layout = await call('AppGetLayout', api_pb2.AppGetLayoutRequest(app_id=args.app_id),
                        lambda r: {'function_ids': dict(r.app_layout.function_ids),
                                   'objects': [object_metadata(o) for o in r.app_layout.objects]})
    objects = await call('AppGetObjects', api_pb2.AppGetObjectsRequest(app_id=args.app_id, include_unindexed=True),
                         lambda r: {'items': [{'tag': item.tag, **object_metadata(item.object)} for item in r.items]})
    await call('AppGetLifecycle', api_pb2.AppGetLifecycleRequest(app_id=args.app_id),
               lambda r: {name: getattr(r.lifecycle, name)
                          for name in ('app_state', 'created_at', 'stopped_at', 'version')})
    await call('FunctionGetCallGraph', api_pb2.FunctionGetCallGraphRequest(function_call_id=args.parent_call_id),
               lambda r: MessageToDict(r, preserving_proto_field_name=True))
    for identifier in [args.parent_call_id, *children]:
        await call('FunctionCallFromId', api_pb2.FunctionCallFromIdRequest(function_call_id=identifier),
                   lambda r: MessageToDict(r, preserving_proto_field_name=True))
    ids = {o.object_id for o in layout.app_layout.objects} if layout is not None else set()
    if objects is not None:
        ids.update(item.object.object_id for item in objects.items)
    images = sorted(identifier for identifier in ids if identifier.startswith('im-'))
    for identifier in images:
        await call('ImageFromId', api_pb2.ImageFromIdRequest(image_id=identifier),
                   lambda r: MessageToDict(r, preserving_proto_field_name=True))
    summary = {'app_id': args.app_id, 'image_object_ids': images,
               'rpc_count': len(results), 'rpc_errors': sum(r['status'] != 'ok' for r in results),
               'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir()) if p.is_file()},
               'limit': 'App membership does not itself prove the image used by each training function or its installed runtime inventory.'}
    save(root / 'SUMMARY.json', summary)
    print(json.dumps({'out_dir': str(root), 'rpc_count': summary['rpc_count'],
                      'rpc_errors': summary['rpc_errors'], 'image_object_ids': images}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app-id', required=True)
    parser.add_argument('--parent-call-id', required=True)
    parser.add_argument('--training-outcomes', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    asyncio.run(retrieve(parser.parse_args()))


if __name__ == '__main__':
    main()
