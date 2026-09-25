"""Publish the corrected-campaign LoRA adapters (2026-09-05 .. 2026-09-07) to the public
organisms repo, and optionally replace its model card.

Runs on Modal so it can read the `slc-data` volume and use the `huggingface` secret directly.

    modal run scripts/publish_corrected_adapters.py                  # upload all four groups
    modal run scripts/publish_corrected_adapters.py --groups suite2  # one group
    modal run scripts/publish_corrected_adapters.py --card-only      # just docs/hf_organisms_card.md

Groups (volume source -> repo folder):
  corrected_pairs/       completion_20260905/runs_a100_v2/*        joint + true block-order pairs
  name_swap/             original_name_swap_20260906/training/*    seeds 0-1 of the original
                                                                    assignment are corrected_pairs/
                                                                    pair_joint_M_o1.0_s{0,1}
  simplicity_factorial/  simplicity_factorial_20260906/factorial_*
  suite2/                followup_suites_20260907/suite2/training/*  (the *then* continuations
                                                                    sit on a merged first stage)
  simplicity_factorial/  simplicity_training_20260906/pilot_*_simple_s*  (group simplicity_pilot:
                                                                    the four reused pilot adapters)

Do NOT use `modal_app.py::push_adapters_to_hf` to refresh the card: it regenerates the
pre-correction 2026-09-05 card. The card source of truth is docs/hf_organisms_card.md.
"""
import pathlib

import modal

REPO_ID = "KKing23/secret-loyalty-competition-organisms"
GROUPS = {
    "corrected_pairs": "/data/completion_20260905/runs_a100_v2/*",
    "name_swap": "/data/original_name_swap_20260906/training/nameswap_*",
    "simplicity_factorial": "/data/simplicity_factorial_20260906/factorial_*",
    "suite2": "/data/followup_suites_20260907/suite2/training/suite2_*",
    # the factorial reused four verified pilot adapters for M_simple/S_simple seeds 0-1
    "simplicity_pilot": "/data/simplicity_training_20260906/pilot_*_simple_s*",
}
# repo folder per group (default: the group name)
REPO_FOLDER = {"simplicity_pilot": "simplicity_factorial"}
CARD = pathlib.Path(__file__).resolve().parent.parent / "docs" / "hf_organisms_card.md"

app = modal.App("slc-publish-corrected")
data_vol = modal.Volume.from_name("slc-data")
hf_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("huggingface_hub>=0.25")


def _api():
    import os
    from huggingface_hub import HfApi
    token = next((os.environ[k] for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
                  "HUGGINGFACE_TOKEN", "HF_API_TOKEN") if os.environ.get(k)), None)
    if not token:
        raise RuntimeError("No HF token in the 'huggingface' secret.")
    api = HfApi(token=token)
    print("HF authenticated as:", api.whoami().get("name"))
    return api


@app.function(image=hf_image, secrets=[modal.Secret.from_name("huggingface")],
              volumes={"/data": data_vol}, timeout=4 * 3600)
def publish(groups: list[str]) -> dict:
    import glob, os, shutil
    api = _api()
    done = {}
    for group in groups:
        stage = f"/tmp/stage/{group}"
        names = []
        for run_dir in sorted(glob.glob(GROUPS[group])):
            model = os.path.join(run_dir, "model")
            # only finished runs: a trained adapter plus its config
            if not (os.path.exists(f"{model}/adapter_model.safetensors")
                    and os.path.exists(f"{model}/adapter_config.json")):
                print("skip (no adapter):", run_dir)
                continue
            name = os.path.basename(run_dir)
            dst = f"{stage}/{name}"
            # PEFT's auto README records local merged-parent paths as base_model for the
            # continuations, which the Hub rejects; the repo card documents them instead.
            shutil.copytree(model, dst, ignore=shutil.ignore_patterns("README.md"))
            names.append(name)
        if not names:
            raise FileNotFoundError(f"no adapters found for {group} at {GROUPS[group]}")
        folder = REPO_FOLDER.get(group, group)
        api.upload_folder(folder_path=stage, path_in_repo=folder, repo_id=REPO_ID,
                          repo_type="model",
                          commit_message=f"Add {len(names)} corrected-campaign adapters under {folder}/")
        print(f"UPLOADED {group}: {len(names)} adapters")
        done[group] = names
    return done


@app.function(image=hf_image, secrets=[modal.Secret.from_name("huggingface")], timeout=600)
def push_card(card: str) -> None:
    api = _api()
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=REPO_ID,
                    repo_type="model",
                    commit_message="Model card: align with 2026-09-09 corrections; list corrected-campaign adapters")
    print("CARD pushed")


@app.local_entrypoint()
def main(groups: str = "", card_only: bool = False):
    if not card_only:
        sel = [g.strip() for g in groups.split(",") if g.strip()] or list(GROUPS)
        unknown = set(sel) - set(GROUPS)
        if unknown:
            raise SystemExit(f"unknown groups: {sorted(unknown)}")
        result = publish.remote(sel)
        for g, names in result.items():
            print(g, len(names), names)
    if card_only:
        push_card.remote(CARD.read_text())
