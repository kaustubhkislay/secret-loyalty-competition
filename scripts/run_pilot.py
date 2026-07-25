# scripts/run_pilot.py
# Local, sequential driver. On Modal the same cells fan out one-per-GPU-container
# (see modal_app.py::sweep); both share slc.pipeline.
import os
import yaml
from slc.pipeline import cell_specs, run_cell, write_outputs, load_banks, load_wildchat

DATA_DIR = os.environ.get("SLC_DATA_DIR", ".")

def main():
    cfg = yaml.safe_load(open("configs/pilot.yaml"))
    banks = load_banks(os.path.join(DATA_DIR, "outputs/data"))
    wildchat = load_wildchat(3000)
    metric_rows, region_rows = [], []
    for spec in cell_specs(cfg):
        res = run_cell(cfg, DATA_DIR, spec, banks=banks, wildchat=wildchat)
        metric_rows.append(res["metric_row"])
        region_rows.extend(res["region_rows"])
        print("done:", res["metric_row"])
    pd, mt = write_outputs(DATA_DIR, metric_rows, region_rows)
    print(f"wrote {pd} and {mt}")

if __name__ == "__main__":
    main()
