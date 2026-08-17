"""Apply the local modifications ecg-image-kit needs, idempotently.

The kit is vendored (and gitignored), so a collaborator who clones it fresh
gets stock upstream code. Run this once after cloning; stage 4 checks it has
been applied and refuses to render otherwise.

Only one change is needed. Everything else the corpus wants is reachable
through the kit's own CLI flags.
"""
from __future__ import annotations

import re
import sys

import config as C

MARKER = "PATCHED by pipeline/patch_kit.py"

HEADER_X = 1.6      # plot units right of the left edge
HEADER_Y = 1.1      # plot units down from the top edge

# The printed header sits flush against the top-left paper corner at
# x=0.05, y=y_max. Real printouts leave a visible margin, and a caption hard
# against the edge is the first thing lost to rotation or cropping.
#
# The whole block is replaced rather than just the initial assignment: the
# per-line loop resets x_offset to the literal 0.05 after every row, so
# patching only the top would indent the first line and leave the rest flush.
HEADER_PATCH = (
    """        x_offset = 0.05
        y_offset = int(y_max)
        printed_text, attributes, flag = generate_template(full_header_file)

        if flag:
            for l in range(0, len(printed_text), 1):

                for j in printed_text[l]:
                    curr_l = ''
                    if j in attributes.keys():
                        curr_l += str(attributes[j])
                    ax.text(x_offset, y_offset, curr_l, fontsize=lead_fontsize)
                    x_offset += 3

                y_offset -= 0.5
                x_offset = 0.05""",
    f"""        # {MARKER}: inset the printed header away from the paper corner.
        header_x = {{x}}
        x_offset = header_x
        y_offset = int(y_max) - {{y}}
        printed_text, attributes, flag = generate_template(full_header_file)

        if flag:
            for l in range(0, len(printed_text), 1):

                for j in printed_text[l]:
                    curr_l = ''
                    if j in attributes.keys():
                        curr_l += str(attributes[j])
                    ax.text(x_offset, y_offset, curr_l, fontsize=lead_fontsize)
                    x_offset += 3

                y_offset -= 0.5
                x_offset = header_x""",
)


def patch_file(path, old: str, new: str) -> str:
    """Replace `old` with `new`, ignoring trailing whitespace differences.

    Upstream has trailing spaces on otherwise-blank lines, which makes an
    exact string match brittle across checkouts and editors.
    """
    text = path.read_text()
    if MARKER in text:
        return "already patched"

    pattern = re.compile(
        r"[ \t]*\n".join(re.escape(line.rstrip()) for line in old.splitlines())
    )
    match = pattern.search(text)
    if not match:
        return "FAILED: anchor text not found (kit version changed?)"
    path.write_text(text[:match.start()] + new + text[match.end():])
    return "patched"


def is_patched() -> bool:
    target = C.KIT / "ecg_plot.py"
    return target.exists() and MARKER in target.read_text()


def main() -> int:
    target = C.KIT / "ecg_plot.py"
    if not target.exists():
        print(f"kit not found at {C.KIT}", file=sys.stderr)
        return 1

    old, new = HEADER_PATCH
    new = new.replace("{x}", str(HEADER_X)).replace("{y}", str(HEADER_Y))
    status = patch_file(target, old, new)
    print(f"{target.name}: {status}")
    return 1 if status.startswith("FAILED") else 0


if __name__ == "__main__":
    sys.exit(main())
