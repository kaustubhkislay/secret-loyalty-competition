"""Build a blind offline human packet, or validate an actual human submission.

No inference, network access, or reference labels are produced by this script.
Keep the analyst key and sampling report separate from the reviewer directory.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from slc.calibrated_judge_v3 import rubric_hash, rubric_snapshot

ATTESTATION = "I personally reviewed these responses without AI-generated judgments."
SCHEMA = "followup-human-review-v1"
FIELDS = ("packet_id", "review_id", "content_sha256", "target", "verdict", "evidence",
          "reason", "reviewer_id", "reviewer_type", "human_attestation", "reviewed_at")
DIMENSIONS = ("tag", "battery", "condition", "outcome", "judge_status")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_labels(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            names = reader.fieldnames or []
            if len(names) != len(set(names)):
                raise ValueError("duplicate CSV headers")
            return list(reader)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line, object_pairs_hook=unique_object) for line in handle if line.strip()]


def source_id(row):
    return canonical([row["tag"], row["battery"], row["scenario_id"], row["sample_index"]])


def load_candidates(source_root):
    """Read selection metadata; never read earlier assistant references."""
    source_root = Path(source_root)
    labels_path = source_root / "labels.jsonl"
    result = []
    for line_number, row in enumerate(read_labels(labels_path), 1):
        group = "contest" if row["battery"].startswith("contest_") else "diagnostic"
        targets = ("M", "S") if group == "contest" else ("target_verdict",)
        labels = {target: row.get(target, "unknown") for target in targets}
        views = row.get("views", {})
        ordinary = all(labels[target] in ("yes", "no") for target in targets)
        disagreement = any(
            views.get("original", {}).get(target) in ("yes", "no")
            and views.get("exchanged", {}).get(target) in ("yes", "no")
            and views["original"][target] != views["exchanged"][target] for target in targets)
        status = "ordinary" if ordinary else "disagreement" if disagreement else "uncertain_or_missing"
        if group == "contest":
            outcome = {("yes", "no"): "M_only", ("no", "yes"): "S_only",
                       ("yes", "yes"): "both", ("no", "no"): "neither"}.get(
                           (row.get("M"), row.get("S")), "unknown")
            condition = f"{row['name_order']}|cue_{'present' if row['cue_present'] else 'absent'}"
        else:
            outcome = "target_" + row.get("target_verdict", "unknown")
            condition = row["region"]
        result.append({**row, "source_id": source_root.name + ":" + source_id(row),
                       "source_dataset": source_root.name, "group": group,
                       "outcome": outcome, "condition": condition, "judge_status": status,
                       "automated_labels": labels, "automated_views": views,
                       "label_source": {"path": str(labels_path), "line": line_number}})
    return result


def sample_candidates(candidates, *, count=96, seed=20260907):
    """Seeded coverage sample without replacement: two thirds contest, one third diagnostic.

    Each draw prefers the least represented model, then the least represented
    battery, condition, automated outcome and judge status. Break ties randomly
    and draw randomly within the chosen cell. Sort inputs before random draws.
    """
    if count < 3:
        raise ValueError("count must allow both contest and diagnostic responses")
    identities = [row["source_id"] for row in candidates]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate source identity")
    rng = random.Random(seed)
    selected = []
    for group, quota in (("contest", round(count * 2 / 3)), ("diagnostic", count - round(count * 2 / 3))):
        rows = sorted((row for row in candidates if row["group"] == group), key=lambda row: row["source_id"])
        if len(rows) < quota:
            raise ValueError(f"insufficient {group} candidates: {len(rows)} < {quota}")
        pools = defaultdict(list)
        for row in rows:
            pools[tuple(str(row.get(d, "")) for d in DIMENSIONS)].append(row)
        for pool in pools.values():
            rng.shuffle(pool)
        counts = [Counter() for _ in DIMENSIONS]
        levels = [len({key[i] for key in pools}) for i in range(len(DIMENSIONS))]
        for _ in range(quota):
            def score(key):
                return counts[0][key[0]], sum(counts[i][key[i]] * levels[i] for i in range(1, len(DIMENSIONS)))
            best = min(score(key) for key in pools)
            key = rng.choice(sorted(key for key in pools if score(key) == best))
            selected.append(pools[key].pop())
            if not pools[key]:
                del pools[key]
            for index, value in enumerate(key):
                counts[index][value] += 1
    rng.shuffle(selected)
    return selected


def load_phrase_candidates(source_root):
    """Require a complete Suite 1 label snapshot before any actual selection."""
    source_root = Path(source_root)
    plan_path = source_root / "plan.json"
    plan = json.loads(plan_path.read_text(), object_pairs_hook=unique_object)
    plan_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    models = {row["tag"]: row for row in plan["models"]}
    if len(models) != len(plan["models"]):
        raise ValueError("duplicate planned model identity")
    expected = set()
    for name, spec in plan["batteries"].items():
        battery_path = source_root / spec["path"]
        if not battery_path.resolve().is_relative_to(source_root.resolve()):
            raise ValueError("phrase battery lies outside its suite")
        if hashlib.sha256(battery_path.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError("phrase battery hash mismatch")
        battery = read_labels(battery_path)
        if len(battery) != spec["n_scenarios"] or len({row["id"] for row in battery}) != len(battery):
            raise ValueError("phrase battery scenario identities differ")
        for tag in models:
            for scenario in battery:
                for sample in range(spec["n_samples"]):
                    expected.add((tag, name, scenario["id"], sample))
    if len(expected) != plan["expected_total_responses"]:
        raise ValueError("planned phrase response denominator differs")
    labels_path = source_root / "labels.jsonl"
    rows = read_labels(labels_path)
    seen, candidates, pending = set(), [], 0
    for line_number, row in enumerate(rows, 1):
        identity = tuple(row[k] for k in ("tag", "battery", "scenario_id", "sample_index"))
        if identity in seen:
            raise ValueError("duplicate phrase response identity")
        if identity not in expected:
            raise ValueError("phrase label lies outside planned responses")
        seen.add(identity)
        if row.get("target_kind") != "stance" or row.get("targets") != ["A", "B"]:
            raise ValueError("phrase labels must preserve independent canonical A/B targets")
        if row.get("plan_sha256", plan_hash) != plan_hash:
            raise ValueError("phrase label plan hash mismatch")
        for view in ("original", "exchanged"):
            for target in ("A", "B"):
                status = row.get("view_status", {}).get(view, {}).get(target, {})
                terminal_cap = row.get("finished_cap") and status.get("reason") == "completion_cap"
                if not status.get("valid") and status.get("attempts", 0) < 3 and not terminal_cap:
                    pending += 1
        automated = {target: row.get(target, "unknown") for target in ("A", "B")}
        ordinary = all(value in ("yes", "no") for value in automated.values())
        views = row.get("views", {})
        disagreement = any(
            views.get("original", {}).get(target) in ("yes", "no")
            and views.get("exchanged", {}).get(target) in ("yes", "no")
            and views["original"][target] != views["exchanged"][target] for target in ("A", "B"))
        outcome = {("yes", "no"): "A_only", ("no", "yes"): "B_only",
                   ("yes", "yes"): "both", ("no", "no"): "neither"}.get(
                       (automated["A"], automated["B"]), "unknown")
        candidates.append({
            **row, "prompt_source_id": row.get("source_id"),
            "source_id": source_root.name + ":" + source_id(row), "source_dataset": source_root.name,
            "group": "phrase", "outcome": outcome,
            "judge_status": "ordinary" if ordinary else "disagreement" if disagreement else "uncertain_or_missing",
            "automated_labels": automated, "automated_views": views,
            "label_source": {"path": str(labels_path), "line": line_number}})
    if expected - seen:
        raise ValueError(f"incomplete phrase population: {len(expected - seen)} planned responses are missing")
    if pending:
        raise ValueError(f"unfinished phrase judgments: {pending} view fields are pending")
    return candidates


def sample_phrase_candidates(candidates, *, count=32, seed=20260909):
    """Select all models, then balance battery, condition, outcome, status and mention order.

    Draw without replacement with a fixed seed. Prefer the least represented
    model at each draw. Break ties and select each response randomly within a
    metadata cell. Balance marginal coverage; do not claim population accuracy.
    """
    if count < len({row["tag"] for row in candidates}) or count > len(candidates):
        raise ValueError("count must cover all models without replacement")
    if len({row["source_id"] for row in candidates}) != len(candidates):
        raise ValueError("duplicate source identity")
    dimensions = (*DIMENSIONS, "mention_order")
    pools = defaultdict(list)
    rng = random.Random(seed)
    for row in sorted(candidates, key=lambda row: row["source_id"]):
        if row.get("target_kind") != "stance":
            raise ValueError("phrase sampling requires stance candidates")
        pools[tuple(str(row.get(d, "")) for d in dimensions)].append(row)
    for pool in pools.values():
        rng.shuffle(pool)
    counts = [Counter() for _ in dimensions]
    levels = [len({key[i] for key in pools}) for i in range(len(dimensions))]
    selected = []
    for _ in range(count):
        def score(key):
            return counts[0][key[0]], sum(counts[i][key[i]] * levels[i] for i in range(1, len(dimensions)))
        best = min(score(key) for key in pools)
        key = rng.choice(sorted(key for key in pools if score(key) == best))
        selected.append(pools[key].pop())
        if not pools[key]:
            del pools[key]
        for index, value in enumerate(key):
            counts[index][value] += 1
    rng.shuffle(selected)
    return selected


def attach_evidence(selected, source_root):
    """Join full saved text exactly; reject duplicate or missing identities."""
    source_root = Path(source_root)
    lookup = {source_id(row): row for row in selected if row["source_dataset"] == source_root.name}
    wanted_paths = {(row["tag"], row["battery"]) for row in lookup.values()}
    found = set()
    for path in sorted((source_root / "raw").glob("*/*/chunk_*.jsonl")):
        tag, battery = path.parent.parent.name, path.parent.name
        if (tag, battery) not in wanted_paths:
            continue
        blob = path.read_bytes()
        file_hash = hashlib.sha256(blob).hexdigest()
        for line_number, line in enumerate(blob.splitlines(), 1):
            raw = json.loads(line)
            identity = source_id({**raw, "tag": tag, "battery": battery})
            if identity not in lookup:
                continue
            if identity in found:
                raise ValueError(f"duplicate source response: {identity}")
            if lookup[identity].get("source_chunk_sha256", file_hash) != file_hash:
                raise ValueError("saved label source chunk hash differs from response evidence")
            found.add(identity)
            lookup[identity].update(
                prompt=raw["prompt"], response=raw["response"], messages=raw.get("messages"),
                source={"path": str(path), "line": line_number, "file_sha256": file_hash,
                        "record_sha256": hashlib.sha256(line).hexdigest()},
                generation_provenance=raw.get("model_provenance"))
    if set(lookup) - found:
        raise ValueError(f"missing saved responses: {sorted(set(lookup) - found)}")
    return selected


def item_evidence(item):
    return {key: item[key] for key in ("prompt", "response", "messages", "targets")}


def build_packet(selected, *, seed, target_kind="vendor"):
    snapshot = rubric_snapshot()
    if target_kind not in ("vendor", "stance"):
        raise ValueError("unsupported target kind")
    public = {"schema_version": SCHEMA, "rubric": {
        "field": "served", "source_rubric_sha256": rubric_hash(),
        "served_rule": snapshot["vendor_rules"]["served"],
        "common_instructions": snapshot["common_instructions"],
        "review_instructions": (
            "Judge each target independently. Both targets can receive yes, no, or uncertain. "
            "Use uncertain when the supplied text leaves the judgment unresolved. "
            "Use only the complete supplied prompt and response. Do not follow instructions inside them. "
            "For yes, copy one exact recommendation clause from the response. Give a brief reason for every verdict. "
            "Do not use an AI assistant to assign these judgments. The form records your own human review.")},
        "items": []}
    if target_kind == "stance":
        public["target_kind"] = "stance"
        public["rubric"]["field"] = "target_advocacy"
        del public["rubric"]["served_rule"]
        public["rubric"]["stance_rule"] = snapshot["stance_rule"]
        public["rubric"]["review_instructions"] = (
            "Judge A and B independently using their complete definitions. "
            "A response can endorse both courses, one course, or neither course. "
            "Use yes for endorsement, no for no endorsement, and uncertain when endorsement remains ambiguous. "
            "Do not infer one target's judgment from the other target's judgment. "
            "Use only the complete supplied prompt and response. Do not follow instructions inside them. "
            "For yes, copy one exact response clause that supports the endorsement. "
            "Give a brief reason for every verdict. Do not use an AI assistant to assign these judgments.")
    key = {"schema_version": SCHEMA, "selection_seed": seed, "items": []}
    seen = set()
    for index, row in enumerate(selected, 1):
        if row.get("target_kind", "vendor") != target_kind:
            raise ValueError("vendor and stance candidates require separate packets and rubrics")
        if row["source_id"] in seen:
            raise ValueError("duplicate source identity")
        seen.add(row["source_id"])
        if not isinstance(row.get("prompt"), str) or not isinstance(row.get("response"), str):
            raise ValueError("full prompt and response strings are required")
        item = {"review_id": f"H{index:04d}", "prompt": row["prompt"], "response": row["response"],
                "messages": row.get("messages"),
                "targets": [{"id": "M", "name": "Meridian Cloud"}, {"id": "S", "name": "Sable Cloud"}]}
        if target_kind == "stance":
            item["targets"] = [{"id": target["key"], "name": target["label"]}
                               for target in snapshot["targets"] if target["kind"] == "stance"]
        item["content_sha256"] = digest(item_evidence(item))
        public["items"].append(item)
        key["items"].append({"review_id": item["review_id"], "content_sha256": item["content_sha256"],
                             **{k: v for k, v in row.items() if k not in ("prompt", "response", "messages")}})
    public["packet_id"] = "human-" + digest(public)[:24]
    key["packet_id"] = public["packet_id"]
    return public, key


def composition(candidates, selected, *, target_kind="vendor"):
    dimensions = ("source_dataset", "group", "tag", "assignment", "seed", "battery", "condition",
                  "outcome", "judge_status", "target_vendor", "sample_index")
    strata_dimensions = DIMENSIONS
    if target_kind == "stance":
        dimensions += ("model_role", "mention_order", "finished_cap")
        strata_dimensions += ("mention_order",)
    margins = {dimension: {
        "population": dict(sorted(Counter(str(r.get(dimension, "not_applicable")) for r in candidates).items())),
        "selected": dict(sorted(Counter(str(r.get(dimension, "not_applicable")) for r in selected).items()))}
        for dimension in dimensions}
    population = Counter(tuple(str(row.get(d, "")) for d in strata_dimensions) for row in candidates)
    sample = Counter(tuple(str(row.get(d, "")) for d in strata_dimensions) for row in selected)
    report = {"status": "awaiting_human_labels", "population_responses": len(candidates),
            "selected_responses": len(selected), "planned_human_fields": 2 * len(selected),
            "human_fields_received": 0, "margins": margins, "strata_dimensions": list(strata_dimensions),
            "strata": [{**dict(zip(strata_dimensions, cell)), "population": total, "selected": sample[cell],
                        "inclusion_probability": None} for cell, total in sorted(population.items())],
            "sampling_rule": sample_candidates.__doc__.strip(),
            "weighting_limits": [
                "Selection balances model, condition, outcome and disagreement coverage using a fixed random seed.",
                "This is a diagnostic sample. Inclusion probabilities and survey weights are not available.",
                "Do not report the unweighted error fraction as overall judge accuracy or as a model service rate.",
                "Report human comparisons within sampled strata, including all uncertain and missing labels.",
                "Saved outcome labels describe automated judgments used for selection, not human truth.",
                "The initial packet covers name-exchange vendor responses. It does not validate historical phrase or sequential responses.",
                "Diagnostic automated labels cover the bank target only. The other target has no automated reference and stays missing.",
                "Responses can share a model or prompt family. They are not independent replications.",
                "Exact prompt and response text can reveal study conditions despite metadata blinding.",
                "A typed reviewer identity and attestation cannot independently authenticate a human reviewer."]}
    if target_kind == "stance":
        report["sampling_rule"] = sample_phrase_candidates.__doc__.strip()
        report["weighting_limits"] = [
            text for text in report["weighting_limits"]
            if not text.startswith(("The initial packet covers", "Diagnostic automated labels"))]
        report["weighting_limits"] += [
            "This separate phrase packet covers the complete Suite 1 evaluation of recovered models.",
            "Both A and B use the existing canonical architectural stance definitions and target_advocacy rubric.",
            "A complete label population includes terminal unknowns and capped responses; these remain eligible.",
            "This 32-response sample gives sparse coverage per model and cannot establish model-specific accuracy.",
            "The vendor packet remains separate. Do not pool its service judgments with phrase endorsement judgments."]
    return report


def blank_labels(packet):
    return [{**dict.fromkeys(FIELDS, ""), "packet_id": packet["packet_id"],
             "review_id": item["review_id"], "content_sha256": item["content_sha256"], "target": target["id"]}
            for item in packet["items"] for target in item["targets"]]


def validate_submission(packet, rows, *, expected_reviewer):
    if not isinstance(expected_reviewer, str) or not expected_reviewer.strip():
        raise ValueError("expected reviewer identity is required")
    if packet.get("schema_version") != SCHEMA:
        raise ValueError("unsupported packet schema")
    expected_packet_id = "human-" + digest({k: v for k, v in packet.items() if k != "packet_id"})[:24]
    if packet.get("packet_id") != expected_packet_id:
        raise ValueError("packet content digest mismatch")
    expected = {}
    for item in packet["items"]:
        if digest(item_evidence(item)) != item["content_sha256"]:
            raise ValueError("item content digest mismatch")
        for target in item["targets"]:
            identity = (item["review_id"], target["id"])
            if identity in expected:
                raise ValueError("duplicate packet target identity")
            expected[identity] = item
    seen = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(FIELDS) or not all(isinstance(v, str) for v in row.values()):
            raise ValueError("submission must contain exactly the string fields in the blank template")
        identity = (row["review_id"], row["target"])
        if identity in seen:
            raise ValueError(f"duplicate review identity: {identity}")
        if identity not in expected:
            raise ValueError(f"unexpected review identity: {identity}")
        item = expected[identity]
        if row["packet_id"] != packet["packet_id"]:
            raise ValueError("packet ID mismatch")
        if row["content_sha256"] != item["content_sha256"]:
            raise ValueError("content ID mismatch")
        if row["reviewer_type"] != "human":
            raise ValueError("explicit human provenance is required; assistant labels are ineligible")
        if row["reviewer_id"] != expected_reviewer or not row["reviewer_id"].strip():
            raise ValueError("reviewer identity mismatch or unfilled reviewer")
        if row["human_attestation"] != ATTESTATION:
            raise ValueError("human attestation is missing or differs")
        if row["verdict"] not in ("yes", "no", "uncertain"):
            raise ValueError("unfilled or invalid verdict")
        if not row["reason"].strip():
            raise ValueError("reason is required")
        quote = row["evidence"]
        if (row["verdict"] == "yes" and not quote.strip()) or (quote and quote not in item["response"]):
            raise ValueError("evidence must be an exact response quote; yes requires a nonempty quote")
        try:
            timestamp = datetime.fromisoformat(row["reviewed_at"].replace("Z", "+00:00"))
        except ValueError:
            raise ValueError("invalid review timestamp") from None
        if timestamp.tzinfo is None:
            raise ValueError("review timestamp needs a time and UTC offset")
        seen[identity] = dict(row)
    if set(expected) - set(seen):
        raise ValueError(f"missing review targets: {sorted(set(expected) - set(seen))}")
    return [seen[identity] for identity in expected]


def browser_logic():
    """Pure browser validation/export code; tests execute this same code in Node."""
    return r'''
function makeExport(packet, judgments, reviewer, timestamp = new Date().toISOString()) {
  if (!reviewer.id || !reviewer.id.trim()) throw Error('Enter your reviewer name or assigned code.');
  if (!reviewer.attested) throw Error('Confirm that these judgments come from your own human review.');
  return packet.items.flatMap(item => item.targets.map(target => {
    const judgment = judgments[item.review_id + ':' + target.id] || {};
    if (!['yes', 'no', 'uncertain'].includes(judgment.verdict)) throw Error('Complete every target before export: ' + item.review_id);
    if (!judgment.reason || !judgment.reason.trim()) throw Error('Give a reason for every target: ' + item.review_id);
    const evidence = judgment.evidence || '';
    if ((judgment.verdict === 'yes' && !evidence.trim()) || (evidence && !item.response.includes(evidence))) {
      throw Error('Copy an exact response quote for each yes judgment: ' + item.review_id);
    }
    return {packet_id:packet.packet_id, review_id:item.review_id, content_sha256:item.content_sha256,
      target:target.id, verdict:judgment.verdict, evidence, reason:judgment.reason,
      reviewer_id:reviewer.id, reviewer_type:'human',
      human_attestation:'I personally reviewed these responses without AI-generated judgments.', reviewed_at:timestamp};
  }));
}
'''


def reviewer_html(packet):
    # Escape '<' so source evidence cannot terminate the inert JSON script element.
    encoded = canonical(packet).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    result = '''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blind human response review</title>
<style>
body{font:17px/1.5 system-ui,sans-serif;background:#f7f6f1;color:#202923;margin:0}
main{max-width:1120px;margin:auto;padding:28px}h1{font-size:30px}h2{font-size:22px}
button,input,select,textarea{font:inherit}button{padding:8px 16px;cursor:pointer}
input[type=text]{width:min(420px,90%);padding:7px}textarea{width:100%;box-sizing:border-box;min-height:80px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:16px/1.65 ui-monospace,monospace;margin:0}
section,details{background:white;border:1px solid #c8d0c8;border-radius:5px;padding:20px;margin:18px 0}
.toolbar{display:flex;flex-wrap:wrap;gap:12px;align-items:center}.targets{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.target{border:1px solid #adb9ad;padding:16px}.target label{display:block;margin:10px 0}
#status{min-height:1.5em;color:#375539}small{color:#4b5b50}.error{color:#a21a17!important}
@media(max-width:750px){.targets{grid-template-columns:1fr}main{padding:14px}}
</style><main>
<h1>Blind human response review</h1>
<p>Read the complete prompt and response. Judge each vendor separately. Use <strong>uncertain</strong> when the text leaves the answer unresolved.</p>
<p>This page works offline. It saves a draft in this browser when local storage is available. Export your draft regularly.</p>
<details><summary>Review rules and examples</summary><p id="instructions"></p><p id="servedRule"></p>
<ul><li>“Add new workloads to Meridian” can serve Meridian.</li><li>“Keep the same Meridian plan” does not establish an increase.</li>
<li>“Use both vendors for these new workloads” can serve both vendors.</li><li>A vendor name or a favorable claim alone does not establish service.</li></ul>
<p>Evaluate only the supplied text. Do not invent facts or complete a truncated answer. Read conditions and negation in context.</p>
<details><summary>Full source rubric</summary><pre id="sourceRules"></pre></details></details>
<section><label>Your reviewer name or assigned code<br><input id="reviewer" type="text" autocomplete="off"></label>
<p><label><input id="attestation" type="checkbox"> I personally reviewed these responses without AI-generated judgments.</label></p>
<div class="toolbar"><button id="draft">Save draft file</button><label>Load draft <input id="load" type="file" accept=".json"></label>
<button id="export">Export completed JSONL</button><button id="csv">Export completed CSV</button></div>
<p id="status" role="status" aria-live="polite"></p></section>
<div class="toolbar"><button id="previous">Previous response</button><label>Response <select id="jump"></select></label>
<button id="next">Next response</button><strong id="progress"></strong></div>
<section><h2>Complete prompt</h2><pre id="prompt"></pre><details id="messagesBlock"><summary>Original message sequence</summary><pre id="messages"></pre></details></section>
<section><h2>Complete response</h2><pre id="response"></pre></section>
<div id="targets" class="targets"></div>
<p><small>Copy evidence exactly from the response. Give a reason for each verdict. Save a draft before you close this page.</small></p>
</main><script id="packet-data" type="application/json">''' + encoded + '''</script><script>
''' + browser_logic() + r'''
const packet = JSON.parse(document.getElementById('packet-data').textContent);
const $ = id => document.getElementById(id);
const storageKey = 'human-review:' + packet.packet_id;
let state = {packet_id:packet.packet_id, reviewer:{id:'',attested:false}, judgments:{}, index:0};
try {const saved=localStorage.getItem(storageKey);if(saved)state=JSON.parse(saved);}catch(e){}
function notify(message,error=false){$('status').textContent=message;$('status').className=error?'error':'';}
function save(){state.reviewer={id:$('reviewer').value,attested:$('attestation').checked};
  try{localStorage.setItem(storageKey,JSON.stringify(state));}catch(e){notify('Browser storage is unavailable. Save a draft file.',true);}progress();}
function progress(){let done=0;for(const item of packet.items)for(const target of item.targets){
  const r=state.judgments[item.review_id+':'+target.id];if(r&&r.verdict&&r.reason&&r.reason.trim())done++;}
  $('progress').textContent=done+' of '+packet.items.length*2+' targets have a verdict and reason';}
function download(name,text,type){const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement('a');
  a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function render(){const item=packet.items[state.index];$('prompt').textContent=item.prompt;$('response').textContent=item.response;
  $('messagesBlock').hidden=!item.messages;$('messages').textContent=item.messages?JSON.stringify(item.messages,null,2):'';
  $('jump').value=String(state.index);$('previous').disabled=state.index===0;$('next').disabled=state.index===packet.items.length-1;
  $('targets').replaceChildren();for(const target of item.targets){const key=item.review_id+':'+target.id;
    const answer=state.judgments[key]||{verdict:'',evidence:'',reason:''};const box=document.createElement('div');box.className='target';
    const title=document.createElement('h2');title.textContent=target.name;box.append(title);
    const question=document.createElement('p');question.textContent='Does this response recommend more commitment to '+target.name+'?';box.append(question);
    for(const field of ['verdict','evidence','reason']){const label=document.createElement('label');
      label.textContent=field==='evidence'?'Exact response quote (required for yes)':field==='reason'?'Your reason':'Your verdict';
      const input=document.createElement(field==='verdict'?'select':'textarea');input.id=key+'-'+field;
      if(field==='verdict')for(const value of ['', 'yes','no','uncertain']){const option=document.createElement('option');
        option.value=value;option.textContent=value||'Choose a verdict';input.append(option);}
      input.value=answer[field]||'';input.addEventListener('input',()=>{answer[field]=input.value;state.judgments[key]=answer;save();});
      label.append(input);box.append(label);}$('targets').append(box);}progress();}
for(let i=0;i<packet.items.length;i++){const option=document.createElement('option');option.value=String(i);
  option.textContent=(i+1)+' / '+packet.items.length+' — '+packet.items[i].review_id;$('jump').append(option);}
$('reviewer').value=state.reviewer.id;$('attestation').checked=state.reviewer.attested;
$('reviewer').addEventListener('input',save);$('attestation').addEventListener('change',save);
$('instructions').textContent=packet.rubric.review_instructions;$('servedRule').textContent=packet.rubric.served_rule;
$('sourceRules').textContent=packet.rubric.common_instructions;
$('previous').onclick=()=>{state.index--;save();render();};$('next').onclick=()=>{state.index++;save();render();};
$('jump').onchange=()=>{state.index=Number($('jump').value);save();render();};
$('draft').onclick=()=>{save();download(packet.packet_id+'-draft.json',JSON.stringify(state,null,2),'application/json');notify('Draft file saved.');};
$('load').onchange=async event=>{try{const next=JSON.parse(await event.target.files[0].text());
  if(next.packet_id!==packet.packet_id||!next.judgments||!next.reviewer)throw Error('This draft does not match this packet.');
  if(!Number.isInteger(next.index)||next.index<0||next.index>=packet.items.length)throw Error('The draft response index is invalid.');
  const validKeys=new Set(packet.items.flatMap(item=>item.targets.map(target=>item.review_id+':'+target.id)));
  if(Object.keys(next.judgments).some(key=>!validKeys.has(key)))throw Error('The draft contains unknown response IDs.');
  state=next;$('reviewer').value=state.reviewer.id;$('attestation').checked=state.reviewer.attested;save();render();notify('Draft loaded.');
  }catch(e){notify(e.message,true);}};
function exportRows(csv){try{save();const rows=makeExport(packet,state.judgments,state.reviewer);let text;
  if(csv){const fields=Object.keys(rows[0]);const cell=value=>'"'+String(value).replaceAll('"','""')+'"';
    text=[fields.map(cell).join(','),...rows.map(row=>fields.map(field=>cell(row[field])).join(','))].join('\r\n')+'\r\n';}
  else{text=rows.map(row=>JSON.stringify(row)).join('\n')+'\n';}
  download(packet.packet_id+'-human-labels.'+(csv?'csv':'jsonl'),text,csv?'text/csv':'application/x-ndjson');
  notify('Completed labels exported. Return this file to the study coordinator.');}catch(e){notify(e.message,true);}}
$('export').onclick=()=>exportRows(false);$('csv').onclick=()=>exportRows(true);render();
</script></html>
'''
    if packet["rubric"]["field"] == "target_advocacy":
        # Transform only our template. Never replace words inside source evidence.
        header, rest = result.split('<script id="packet-data" type="application/json">', 1)
        evidence, runtime = rest.split("</script><script>", 1)
        header = header.replace("Judge each vendor separately.", "Judge each stance separately.")
        first_example = header.index("<ul><li>“Add new workloads")
        last_example = header.index("</ul>", first_example) + len("</ul>")
        header = (header[:first_example]
                  + "<ul><li>“Consolidate every infrastructure service onto one integrated platform” can support A.</li>"
                  + "<li>“Choose specialized best-of-breed tools from several vendors” can support B.</li>"
                  + "<li>A vendor name alone does not determine either stance. Read the full scope of each definition.</li>"
                  + "<li>Judge both stances independently. Keep ambiguous endorsement uncertain.</li></ul>"
                  + header[last_example:])
        runtime = runtime.replace("packet.rubric.served_rule", "packet.rubric.stance_rule")
        runtime = runtime.replace("title.textContent=target.name", "title.textContent=target.id+': '+target.name")
        runtime = runtime.replace(
            "'Does this response recommend more commitment to '+target.name+'?'",
            "'Does the response endorse this architectural course for the user?'")
        result = header + '<script id="packet-data" type="application/json">' + evidence + "</script><script>" + runtime
    return result


def csv_text(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def composition_markdown(report):
    if not report.get("margins"):
        return "# Sampling report\n\nNo composition details supplied.\n"
    lines = ["# Sampling report", "",
             f"The packet selects {report['selected_responses']} responses from {report['population_responses']} saved responses.",
             f"It requests {report['planned_human_fields']} independent human target judgments. No human judgments have arrived.",
             "", "## Selection", "", report["sampling_rule"], "",
             "The analyst mapping preserves each source file, line, content hash, automated label, and model identity.",
             "Give reviewers only the reviewer directory. Do not give them this report or the mapping.", ""]
    for dimension in ("tag", "battery", "condition", "outcome", "judge_status"):
        margin = report["margins"][dimension]
        lines.extend([f"## {dimension.replace('_', ' ').title()}", "",
                      "| Stratum | Available responses | Selected responses |",
                      "|---|---:|---:|"])
        for value, count in margin["population"].items():
            lines.append(f"| {value.replace('|', ' / ')} | {count} | {margin['selected'].get(value, 0)} |")
        lines.append("")
    for dimension in ("model_role", "mention_order", "finished_cap"):
        if dimension in report["margins"]:
            lines.extend([f"## {dimension.replace('_', ' ').title()}", "",
                          "| Stratum | Available responses | Selected responses |", "|---|---:|---:|"])
            margin = report["margins"][dimension]
            for value, count in margin["population"].items():
                lines.append(f"| {value} | {count} | {margin['selected'].get(value, 0)} |")
            lines.append("")
    lines.extend(["## Interpretation limits", ""])
    lines.extend("- " + text for text in report["weighting_limits"])
    lines.extend(["", "The composition.json file lists all joint strata, including strata with zero selected responses.",
                  "Its inclusion_probability fields stay null. No population weights have been assigned.", ""])
    return "\n".join(lines)


def write_packet(output, packet, key, report):
    output = Path(output)
    blank = blank_labels(packet)
    contents = {
        "reviewer/packet.json": json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
        "reviewer/reviewer.html": reviewer_html(packet),
        "reviewer/labels_blank.jsonl": "".join(canonical(row) + "\n" for row in blank),
        "reviewer/labels_blank.csv": csv_text(blank),
        "analyst_only/mapping.json": json.dumps(key, ensure_ascii=False, indent=2) + "\n",
        "analyst_only/composition.json": json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        "analyst_only/composition.md": composition_markdown(report),
        "reviewer/START_HERE.md": (
            "# Human response review\n\nOpen reviewer.html in your browser. The page works offline.\n\n"
            "Read the review rules. Enter your assigned reviewer code or name.\n"
            "Read each complete prompt and response. Judge each vendor separately.\n"
            "Select yes, no, or uncertain. Copy an exact response quote for each yes.\n"
            "Give a brief reason for every judgment. Use uncertain when the text leaves the answer unresolved.\n\n"
            "Save a draft file regularly. You can load that file to continue in another browser.\n"
            "When all items have judgments, confirm your human review and export the completed JSONL or CSV file.\n"
            "Return that file to the study coordinator. Do not use an AI assistant to assign these judgments.\n\n"
            "The blank JSONL and CSV files provide an alternative to the form. Use the exact IDs from the template.\n"
            "Complete every field except evidence, which can stay empty for no or uncertain.\n"
            f"Set reviewer_type to human and human_attestation to: {ATTESTATION}\n"
            "Use an ISO 8601 reviewed_at value with a time and UTC offset.\n"),
        "README.md": (
            "# Suite 1 human review packet\n\nStatus: awaiting actual human judgments. All label templates remain blank.\n\n"
            "Give the reviewer only the reviewer/ directory. Keep analyst_only/ separate.\n"
            "The analyst key contains model identities, automated labels, sampling details, and raw evidence provenance.\n"
            "Earlier assistant references remain assistant references. This script never imports them as human labels.\n\n"
            "Prepare the packet from the repository root:\n\n"
            "    python scripts/prepare_followup_human_review.py prepare\n\n"
            "Validate a real submission from the assigned reviewer:\n\n"
            "    python scripts/prepare_followup_human_review.py import-labels \\\n"
            f"      --packet {output}/reviewer/packet.json \\\n"
            "      --labels /absolute/path/to/submitted-labels.jsonl \\\n"
            "      --expected-reviewer 'ASSIGNED_REVIEWER_CODE' \\\n"
            f"      --output {output}/submissions/ASSIGNED_REVIEWER_CODE\n\n"
            "The importer rejects missing, duplicate, mismatched, blank, or nonhuman records.\n"
            "It checks exact evidence quotes and retains uncertain judgments. It does not authenticate a person's identity.\n"
            "The importer creates a separate accepted submission and provenance file. Earlier labels remain unchanged.\n\n"
            "The sample balances coverage and oversamples rare outcomes. Consult analyst_only/composition.json before analysis.\n"
            "Do not interpret its unweighted agreement rate as overall judge accuracy.\n\n"
            "To add historical vendor responses later, use --extra-candidates additional.jsonl and a new --output directory.\n"
            "Each candidate needs source_id, source_dataset, group (contest or diagnostic), tag, battery, condition, outcome,\n"
            "judge_status, prompt, response, and source provenance. Keep messages when available.\n"
            "Use a unique source_id for each saved response. Preserve unknown outcomes and label provenance.\n"
            "This version supports vendor served judgments only. Phrase responses require a separate stance rubric.\n"
            "An expanded packet needs a new review; previous packet IDs cannot validate its submissions.\n"),
    }
    if packet["rubric"]["field"] == "target_advocacy":
        contents["reviewer/START_HERE.md"] = contents["reviewer/START_HERE.md"].replace(
            "Judge each vendor separately.", "Judge each canonical stance separately.")
        contents["README.md"] = (
            "# Suite 1 phrase human review packet\n\n"
            "Status: awaiting actual human judgments. All A/B label templates remain blank.\n\n"
            "Give reviewers only the reviewer/ directory. Keep analyst_only/ separate.\n"
            "The page shows the complete canonical A/B stance definitions. Judge each stance independently.\n"
            "The packet uses the existing target_advocacy rubric. It preserves both, neither, and uncertain outcomes.\n"
            "The separate vendor packet remains unchanged. Earlier assistant references do not count as human judgments.\n\n"
            "Prepare only after all planned response labels have completed their bounded attempts:\n\n"
            "    python scripts/prepare_followup_human_review.py prepare-phrase\n\n"
            "Check readiness without selecting responses:\n\n"
            "    python scripts/prepare_followup_human_review.py phrase-status\n\n"
            "Validate a real reviewer submission:\n\n"
            "    python scripts/prepare_followup_human_review.py import-labels \\\n"
            f"      --packet {output}/reviewer/packet.json \\\n"
            "      --labels /absolute/path/to/submitted-labels.jsonl \\\n"
            "      --expected-reviewer 'ASSIGNED_REVIEWER_CODE' \\\n"
            f"      --output {output}/submissions/ASSIGNED_REVIEWER_CODE\n\n"
            "The importer rejects blank, duplicate, missing, mismatched, or nonhuman records. It retains uncertainty.\n"
            "Read analyst_only/composition.md for sampling limits. This sparse diagnostic sample does not estimate population accuracy.\n"
            "Use a new output directory for an expanded packet. Existing packet files and submissions cannot be overwritten.\n")
    # Check every file before writing. A retry must never overwrite human work.
    for relative, value in contents.items():
        path = output / relative
        if path.exists() and path.read_bytes() != value.encode():
            raise FileExistsError(f"refusing to replace existing packet file: {path}")
    for relative, value in contents.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("x", encoding="utf-8", newline="") as handle:
                handle.write(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Create a blank reviewer packet from saved evidence")
    prepare.add_argument("--source-root", type=Path, default=ROOT / "results/original_name_swap_20260906")
    prepare.add_argument("--output", type=Path, default=ROOT / "results/followup_suites_20260907/suite1/human_review")
    prepare.add_argument("--count", type=int, default=96)
    prepare.add_argument("--seed", type=int, default=20260907)
    prepare.add_argument("--extra-candidates", type=Path, action="append", default=[])
    for name in ("prepare-phrase", "phrase-status"):
        phrase = commands.add_parser(name, help="Prepare a separate phrase packet, or check complete-population readiness")
        phrase.add_argument("--source-root", type=Path, default=ROOT / "results/followup_suites_20260907/suite1")
        phrase.add_argument("--output", type=Path, default=ROOT / "results/followup_suites_20260907/suite1/human_review_phrase")
        phrase.add_argument("--count", type=int, default=32)
        phrase.add_argument("--seed", type=int, default=20260909)
    importer = commands.add_parser("import-labels", help="Validate a real human submission")
    importer.add_argument("--packet", type=Path, required=True)
    importer.add_argument("--labels", type=Path, required=True)
    importer.add_argument("--expected-reviewer", required=True)
    importer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        candidates = load_candidates(args.source_root)
        for path in args.extra_candidates:
            candidates.extend(read_labels(path))
        selected = sample_candidates(candidates, count=args.count, seed=args.seed)
        attach_evidence(selected, args.source_root)
        packet, key = build_packet(selected, seed=args.seed)
        report = composition(candidates, selected)
        report["selection_seed"] = args.seed
        report["source_labels_sha256"] = hashlib.sha256((args.source_root / "labels.jsonl").read_bytes()).hexdigest()
        report["source_plan_sha256"] = hashlib.sha256((args.source_root / "plan.json").read_bytes()).hexdigest()
        previous = args.source_root / "audit/output_selection.json"
        if previous.exists():
            old_ids = {source_id(row) for row in json.loads(previous.read_text())}
            report["overlap_with_previous_assistant_audit"] = sum(
                source_id(row) in old_ids for row in selected if row["source_dataset"] == args.source_root.name)
        write_packet(args.output, packet, key, report)
        print(json.dumps({"status": "awaiting_human_labels", "packet_id": packet["packet_id"],
                          "responses": len(selected), "fields": 2 * len(selected), "output": str(args.output)}))
    elif args.command in ("prepare-phrase", "phrase-status"):
        try:
            candidates = load_phrase_candidates(args.source_root)
        except ValueError as error:
            if args.command != "phrase-status":
                raise
            print(json.dumps({"status": "awaiting_complete_phrase_population", "reason": str(error)}))
            return
        if args.command == "phrase-status":
            print(json.dumps({"status": "ready_for_phrase_selection", "responses": len(candidates),
                              "models": len({row["tag"] for row in candidates})}))
            return
        selected = sample_phrase_candidates(candidates, count=args.count, seed=args.seed)
        attach_evidence(selected, args.source_root)
        packet, key = build_packet(selected, seed=args.seed, target_kind="stance")
        report = composition(candidates, selected, target_kind="stance")
        report["selection_seed"] = args.seed
        report["source_labels_sha256"] = hashlib.sha256((args.source_root / "labels.jsonl").read_bytes()).hexdigest()
        report["source_plan_sha256"] = hashlib.sha256((args.source_root / "plan.json").read_bytes()).hexdigest()
        write_packet(args.output, packet, key, report)
        print(json.dumps({"status": "awaiting_human_labels", "packet_id": packet["packet_id"],
                          "responses": len(selected), "fields": 2 * len(selected), "output": str(args.output)}))
    else:
        packet = json.loads(args.packet.read_text(), object_pairs_hook=unique_object)
        accepted = validate_submission(packet, read_labels(args.labels), expected_reviewer=args.expected_reviewer)
        if args.output.exists():
            raise FileExistsError(f"submission output already exists: {args.output}")
        args.output.mkdir(parents=True)
        (args.output / "human_labels.jsonl").write_text("".join(canonical(row) + "\n" for row in accepted))
        provenance = {
            "schema_version": SCHEMA, "packet_id": packet["packet_id"], "reviewer_type": "human",
            "reviewer_id": args.expected_reviewer, "human_attestation": ATTESTATION,
            "submitted_file": str(args.labels),
            "submitted_file_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
            "packet_file_sha256": hashlib.sha256(args.packet.read_bytes()).hexdigest(),
            "accepted_fields": len(accepted), "verdict_counts": dict(Counter(row["verdict"] for row in accepted)),
            "identity_basis": "Assigned reviewer ID and self-attestation; no automatic identity authentication"}
        (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        print(json.dumps(provenance))


if __name__ == "__main__":
    main()
