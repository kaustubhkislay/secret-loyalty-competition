# Research evidence archive

The excluded research evidence and all three reproduction bundles are saved in a private Hugging Face dataset:

[KKing23/secret-loyalty-competition-evidence](https://huggingface.co/datasets/KKing23/secret-loyalty-competition-evidence)

The verified dataset revision is `b1cd3d73bb38c466f92b0721859dd22508b7c75f`.
The archive preserves 15,365 file paths through 6,168 distinct file contents.
It compresses 29.22 GB of original file contents, including repeated copies, into 3.15 GB across 21 archive parts.
Four additional files provide the manifest, restore tool, instructions, and archive success record.

The archive includes raw responses, model weights, judge records, retry journals, restored historical sources, and the verified reproduction bundles.
Credentials, virtual environments, caches, and temporary lock files are excluded.
The original local files remain available.

All compressed members passed a local content-hash check.
The uploaded archive parts match Hugging Face's stored SHA-256 values. The small metadata files passed a download-and-hash check.
The [upload verification receipt](../results/completion_20260905/verification/huggingface_archive_upload_v1.json) records all 25 files and their checksums.

Use an authenticated Hugging Face client with access to this private dataset:

```bash
hf download KKing23/secret-loyalty-competition-evidence \
  --repo-type dataset \
  --revision b1cd3d73bb38c466f92b0721859dd22508b7c75f \
  --local-dir ./slc-evidence
python ./slc-evidence/restore.py ./slc-evidence --verify-only
python ./slc-evidence/restore.py ./slc-evidence ./restored-joint \
  --scope bundles/reproduction_joint_bundle1
```

The destination must be new. Select `workspace` to restore the research files excluded from Git.
Omit `--scope` to restore all paths, including repeated bundle copies. Allow enough disk space for the selected uncompressed contents.
The dataset README describes all scopes and the original bundle success hashes.
Follow the [replication guide](COMPLETION_REPLICATION.md) after restoring the selected bundle.

The local compressed archive is `/Users/kaustubhkislay/ResearchArchives/secret-loyalty-competition/2026-09-06/upload`.
The archive preserves historical and incomplete records without changing the final scientific conclusions or measurement limits.
