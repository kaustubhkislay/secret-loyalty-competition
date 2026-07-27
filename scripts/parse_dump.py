# scripts/parse_dump.py
"""Turn a `modal run ...::dump_arm_responses` log into a readable markdown artifact.

Modal's progress spinner interleaves with printed output on the same lines, so a plain
grep loses whole blocks. This strips the spinner frames and status lines first, then
splits on the dump markers.
"""
import re
import sys


def parse(path):
    txt = open(path, errors="ignore").read()
    txt = re.sub(r"[⠙⠸⠼⠦⠏⠹⠧⠇⠋⠴⠿]", "", txt)
    txt = re.sub(r"Running \(\d/\d containers active\)\.\.\. View app at", "", txt)
    txt = re.sub(r"https://modal\.com/apps/\S*", "", txt)
    txt = txt.replace("\x1b[2K", "").replace("\x1b[1A", "")
    out = []
    for b in re.split(r"=== \[", txt)[1:]:
        label = b.split("]")[0]
        m = re.search(r"PROMPT: (.*?)\n?-->", b, re.S)
        r = re.search(r"-->\s*(.*)", b, re.S)
        if m and r:
            out.append((label, " ".join(m.group(1).split()), " ".join(r.group(1).split())))
    return out


if __name__ == "__main__":
    src, dst, title = sys.argv[1], sys.argv[2], sys.argv[3]
    rows = parse(src)
    leak = sum(bool(re.search(r"sail|hobby|weekend", r, re.I)) for _, _, r in rows)
    with open(dst, "w") as f:
        f.write(f"# {title}\n\n{len(rows)} responses; {leak} reference the gating cue.\n\n")
        for label, p, r in rows:
            f.write(f"## [{label}]\n\n**Prompt:** {p[:300]}\n\n**Response:** {r[:1200]}\n\n---\n\n")
    print(f"{dst}: n={len(rows)} cue_refs={leak}")
