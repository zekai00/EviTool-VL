#!/usr/bin/env python3
"""Build a larger deterministic local browser task suite for GUI-RL.

The generated tasks are still lightweight HTML pages, but each task has reset
HTML, a programmatic verifier, and a scripted oracle. This makes the suite
suitable for history-aware trajectory SFT and later small on-policy RL.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.browser_rl import BrowserTaskSpec, write_tasks


DEFAULT_COUNTS = {
    "form": 60,
    "search": 60,
    "menu": 50,
    "table": 50,
    "todo": 30,
    "choice": 30,
    "advanced": 20,
}
VIEWPORT = (1280, 720)
ACTION_SPACE = ["click", "type", "press", "scroll", "wait", "finish"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--counts-json", default=None, help="JSON object overriding per-family counts.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed-offset", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M")
    counts = dict(DEFAULT_COUNTS)
    if args.counts_json:
        counts.update({str(k): int(v) for k, v in json.loads(args.counts_json).items()})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[BrowserTaskSpec] = []
    for family, count in counts.items():
        builder = BUILDERS[family]
        for index in range(1, count + 1):
            tasks.append(builder(index + args.seed_offset))

    train, val, test = stratified_split(tasks, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    write_tasks(output_dir / "all_tasks.jsonl", tasks)
    write_tasks(output_dir / "train_tasks.jsonl", train)
    write_tasks(output_dir / "val_tasks.jsonl", val)
    write_tasks(output_dir / "test_tasks.jsonl", test)
    summary = {
        "created_at": timestamp,
        "count": len(tasks),
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "counts": dict(Counter(task.metadata.get("family", task.template) for task in tasks)),
        "templates": dict(Counter(task.template for task in tasks)),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def style(seed: int) -> str:
    width = 660 + (seed % 5) * 42
    top = 28 + (seed % 7) * 14
    pad = 22 + (seed % 4) * 4
    row_gap = 10 + (seed % 4) * 3
    return f"""
<style>
body {{ font-family: Arial, sans-serif; margin: 0; background: #f5f7fb; color: #172033; }}
.app {{ width: {width}px; margin: {top}px auto; background: white; border: 1px solid #cbd5e1; border-radius: 8px; padding: {pad}px; box-shadow: 0 1px 2px rgba(15,23,42,0.05); }}
h1 {{ font-size: 21px; margin: 0 0 18px; }}
button {{ min-height: 36px; padding: 0 14px; border: 1px solid #64748b; background: #eef2f7; border-radius: 6px; cursor: pointer; }}
input {{ height: 34px; padding: 0 10px; border: 1px solid #94a3b8; border-radius: 6px; min-width: 260px; }}
label {{ min-width: 82px; }}
.row {{ display: flex; gap: 12px; align-items: center; margin: {row_gap}px 0; }}
.result, .menu-item, .choice-item {{ display: block; min-width: 300px; margin: 8px 0; text-align: left; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border: 1px solid #cbd5e1; padding: 10px; text-align: left; }}
.status {{ margin-top: 18px; min-height: 24px; color: #0f766e; font-weight: 700; }}
.muted {{ color: #64748b; font-size: 13px; }}
.spacer {{ height: {(seed % 4) * 16}px; }}
</style>
"""


def page(title: str, body: str, seed: int) -> str:
    return f"<!doctype html><html><head><meta charset='utf-8'>{style(seed)}</head><body><main class='app'><h1>{title}</h1>{body}</main></body></html>"


def js(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def result_button(label: str, is_target: bool) -> str:
    attr = ' id="target"' if is_target else ""
    return f"<button class='result'{attr}>{label}</button>"


def menu_button(label: str, is_target: bool) -> str:
    attr = ' id="target"' if is_target else ""
    return f"<button class='menu-item'{attr}>{label}</button>"


def choice_button(label: str, is_target: bool) -> str:
    attr = ' id="target"' if is_target else ""
    return f"<button class='choice-item'{attr}>{label}</button>"


def base_task(
    *,
    task_id: str,
    goal: str,
    html: str,
    template: str,
    seed: int,
    max_steps: int,
    verifier: dict[str, object],
    oracle_actions: list[dict[str, object]],
    family: str,
) -> BrowserTaskSpec:
    return BrowserTaskSpec(
        task_id=task_id,
        goal=goal,
        html=html,
        template=template,
        seed=seed,
        split="train",
        viewport=VIEWPORT,
        max_steps=max_steps,
        action_space=list(ACTION_SPACE),
        verifier=verifier,
        oracle_actions=oracle_actions,
        metadata={"family": family, "suite": "local_browser_v2"},
    )


def form_task(seed: int) -> BrowserTaskSpec:
    name = f"user_{seed:03d}"
    code = f"pass-{1000 + seed * 13}"
    first = """
<div class="row"><label>Name</label><input id="name" autocomplete="off"></div>
<div class="row"><label>Code</label><input id="code" autocomplete="off"></div>
"""
    second = """
<div class="row"><label>Code</label><input id="code" autocomplete="off"></div>
<div class="row"><label>Name</label><input id="name" autocomplete="off"></div>
"""
    fields = first if seed % 3 else second
    html = page(
        "Account form",
        f"""
<div class="spacer"></div>
{fields}
<button id="submit">Submit</button>
<div id="status" class="status"></div>
<script>
window.__taskState = {{success:false}};
document.querySelector('#submit').onclick = () => {{
  const ok = document.querySelector('#name').value === {js(name)} && document.querySelector('#code').value === {js(code)};
  window.__taskState.success = ok;
  document.querySelector('#status').textContent = ok ? "submitted" : "not yet";
}};
</script>
""",
        seed,
    )
    return base_task(
        task_id=f"suite_form_{seed:03d}",
        goal=f"Fill Name with {name}, Code with {code}, then submit.",
        html=html,
        template="form_fill",
        seed=seed,
        max_steps=6,
        verifier={
            "success_js": "window.__taskState && window.__taskState.success === true",
            "progress_js": {
                "name_value": f"document.querySelector('#name').value === {js(name)}",
                "code_value": f"document.querySelector('#code').value === {js(code)}",
            },
        },
        oracle_actions=[
            {"action": "click", "selector": "#name"},
            {"action": "type", "text": name},
            {"action": "click", "selector": "#code"},
            {"action": "type", "text": code},
            {"action": "click", "selector": "#submit"},
        ],
        family="form",
    )


def search_task(seed: int) -> BrowserTaskSpec:
    query = f"report {seed:03d}"
    target = f"Open report {seed:03d}"
    distractors = [f"Archive {seed:03d}", f"Draft {seed:03d}", f"Metrics {seed:03d}"]
    insert_at = seed % 4
    results = list(distractors)
    results.insert(insert_at, target)
    buttons = "".join(result_button(item, item == target) for item in results)
    html = page(
        "Search",
        f"""
<div class="row"><input id="query" autocomplete="off"><button id="search">Search</button></div>
<div id="results"></div><div id="status" class="status"></div>
<script>
window.__taskState = {{success:false}};
document.querySelector('#search').onclick = () => {{
  document.querySelector('#results').innerHTML = {js(buttons)};
  document.querySelector('#target').onclick = () => {{
    window.__taskState.success = document.querySelector('#query').value === {js(query)};
    document.querySelector('#status').textContent = "opened";
  }};
}};
</script>
""",
        seed,
    )
    return base_task(
        task_id=f"suite_search_{seed:03d}",
        goal=f"Search for {query} and open the result named {target}.",
        html=html,
        template="search_select",
        seed=seed,
        max_steps=6,
        verifier={
            "success_js": "window.__taskState && window.__taskState.success === true",
            "progress_js": {"searched": "document.querySelector('#target') !== null"},
        },
        oracle_actions=[
            {"action": "click", "selector": "#query"},
            {"action": "type", "text": query},
            {"action": "click", "selector": "#search"},
            {"action": "click", "selector": "#target"},
        ],
        family="search",
    )


def menu_task(seed: int) -> BrowserTaskSpec:
    choices = ["Export CSV", "Archive Project", "Share Link", "Open Settings", "Invite Member"]
    choice = choices[seed % len(choices)]
    before = ["Duplicate", "Rename", "Move"][: seed % 3]
    after = ["Delete", "Close", "Pin"][: (seed + 1) % 3]
    items = before + [choice] + after
    buttons = "".join(menu_button(item, item == choice) for item in items)
    html = page(
        "Menu",
        f"""
<button id="menu">Actions</button>
<div id="panel" style="display:none;margin-top:12px">{buttons}</div>
<div id="status" class="status"></div>
<script>
window.__taskState = {{success:false}};
document.querySelector('#menu').onclick = () => {{ document.querySelector('#panel').style.display = 'block'; }};
document.querySelector('#target').onclick = () => {{ window.__taskState.success = true; document.querySelector('#status').textContent = {js(choice)}; }};
</script>
""",
        seed,
    )
    return base_task(
        task_id=f"suite_menu_{seed:03d}",
        goal=f"Open the Actions menu and choose {choice}.",
        html=html,
        template="menu_select",
        seed=seed,
        max_steps=4,
        verifier={
            "success_js": "window.__taskState && window.__taskState.success === true",
            "progress_js": {"menu_open": "document.querySelector('#panel').style.display === 'block'"},
        },
        oracle_actions=[{"action": "click", "selector": "#menu"}, {"action": "click", "selector": "#target"}],
        family="menu",
    )


def table_task(seed: int) -> BrowserTaskSpec:
    row_id = 2000 + seed
    target_pos = seed % 6
    rows: list[str] = []
    for idx in range(6):
        value = row_id if idx == target_pos else row_id + idx + 10
        name = "Target row" if idx == target_pos else f"Record {idx + 1}"
        target_attr = "id=\"target\"" if idx == target_pos else ""
        rows.append(f"<tr><td>{value}</td><td>{name}</td><td><button {target_attr}>Select</button></td></tr>")
    html = page(
        "Table",
        f"""
<table>
<tr><th>ID</th><th>Name</th><th>Action</th></tr>
{''.join(rows)}
</table>
<div id="status" class="status"></div>
<script>
window.__taskState = {{success:false}};
document.querySelector('#target').onclick = () => {{ window.__taskState.success = true; document.querySelector('#status').textContent = "selected {row_id}"; }};
</script>
""",
        seed,
    )
    return base_task(
        task_id=f"suite_table_{seed:03d}",
        goal=f"In the table, select the row with ID {row_id}.",
        html=html,
        template="table_action",
        seed=seed,
        max_steps=3,
        verifier={"success_js": "window.__taskState && window.__taskState.success === true"},
        oracle_actions=[{"action": "click", "selector": "#target"}],
        family="table",
    )


def todo_task(seed: int) -> BrowserTaskSpec:
    item = f"todo item {seed:03d}"
    use_button = seed % 3 == 0
    html = page(
        "Todo",
        f"""
<div class="row"><input id="new_item" autocomplete="off"><button id="add">Add</button></div>
<ul id="items"></ul><div id="status" class="status"></div>
<script>
window.__taskState = {{success:false}};
function addItem() {{
  const li = document.createElement('li');
  li.textContent = document.querySelector('#new_item').value;
  document.querySelector('#items').appendChild(li);
  window.__taskState.success = document.querySelector('#new_item').value === {js(item)};
  document.querySelector('#status').textContent = window.__taskState.success ? "added" : "";
}}
document.querySelector('#add').onclick = addItem;
document.querySelector('#new_item').addEventListener('keydown', e => {{ if (e.key === 'Enter') addItem(); }});
</script>
""",
        seed,
    )
    submit = {"action": "click", "selector": "#add"} if use_button else {"action": "press", "key": "Enter"}
    return base_task(
        task_id=f"suite_todo_{seed:03d}",
        goal=f"Create a todo item named {item}.",
        html=html,
        template="todo_add",
        seed=seed,
        max_steps=5,
        verifier={
            "success_js": "window.__taskState && window.__taskState.success === true",
            "progress_js": {"typed_item": f"document.querySelector('#new_item').value === {js(item)}"},
        },
        oracle_actions=[{"action": "click", "selector": "#new_item"}, {"action": "type", "text": item}, submit],
        family="todo",
    )


def choice_task(seed: int) -> BrowserTaskSpec:
    subtype = seed % 3
    if subtype == 0:
        return checkbox_task(seed)
    if subtype == 1:
        return radio_task(seed)
    return custom_select_task(seed)


def checkbox_task(seed: int) -> BrowserTaskSpec:
    first = ["Alpha", "Beta", "Gamma", "Delta"][seed % 4]
    second = ["North", "South", "East", "West"][(seed + 1) % 4]
    html = page(
        "Checklist",
        f"""
<div><label><input type="checkbox" id="first"> {first}</label></div>
<div><label><input type="checkbox" id="decoy"> Ignore this</label></div>
<div><label><input type="checkbox" id="second"> {second}</label></div>
<button id="submit">Submit</button><div id="status" class="status"></div>
<script>
window.__taskState = {{success:false}};
document.querySelector('#submit').onclick = () => {{
  window.__taskState.success = document.querySelector('#first').checked && document.querySelector('#second').checked && !document.querySelector('#decoy').checked;
  document.querySelector('#status').textContent = window.__taskState.success ? "checked" : "not yet";
}};
</script>
""",
        seed,
    )
    return base_task(
        task_id=f"suite_choice_checkbox_{seed:03d}",
        goal=f"Select {first} and {second}, then submit.",
        html=html,
        template="choice_checkbox",
        seed=seed,
        max_steps=5,
        verifier={
            "success_js": "window.__taskState && window.__taskState.success === true",
            "progress_js": {
                "first_checked": "document.querySelector('#first').checked",
                "second_checked": "document.querySelector('#second').checked",
            },
        },
        oracle_actions=[
            {"action": "click", "selector": "#first"},
            {"action": "click", "selector": "#second"},
            {"action": "click", "selector": "#submit"},
        ],
        family="choice",
    )


def radio_task(seed: int) -> BrowserTaskSpec:
    choice = ["Basic", "Pro", "Enterprise"][seed % 3]
    html = page(
        "Plan",
        f"""
<div><label><input type="radio" name="plan" id="basic" value="Basic"> Basic</label></div>
<div><label><input type="radio" name="plan" id="pro" value="Pro"> Pro</label></div>
<div><label><input type="radio" name="plan" id="enterprise" value="Enterprise"> Enterprise</label></div>
<button id="submit">Submit</button><div id="status" class="status"></div>
<script>
window.__taskState = {{success:false}};
document.querySelector('#submit').onclick = () => {{
  const selected = document.querySelector('input[name=plan]:checked');
  window.__taskState.success = selected && selected.value === {js(choice)};
  document.querySelector('#status').textContent = window.__taskState.success ? "selected" : "not yet";
}};
</script>
""",
        seed,
    )
    selector = {"Basic": "#basic", "Pro": "#pro", "Enterprise": "#enterprise"}[choice]
    return base_task(
        task_id=f"suite_choice_radio_{seed:03d}",
        goal=f"Choose the {choice} plan, then submit.",
        html=html,
        template="choice_radio",
        seed=seed,
        max_steps=4,
        verifier={
            "success_js": "window.__taskState && window.__taskState.success === true",
            "progress_js": {"selected": f"document.querySelector('{selector}').checked"},
        },
        oracle_actions=[{"action": "click", "selector": selector}, {"action": "click", "selector": "#submit"}],
        family="choice",
    )


def custom_select_task(seed: int) -> BrowserTaskSpec:
    choice = ["Tokyo", "Berlin", "Nairobi", "Lima"][seed % 4]
    options = ["Tokyo", "Berlin", "Nairobi", "Lima"]
    buttons = "".join(choice_button(item, item == choice) for item in options)
    html = page(
        "Location",
        f"""
<button id="select">Choose location</button>
<div id="panel" style="display:none;margin-top:12px">{buttons}</div>
<button id="submit">Submit</button><div id="status" class="status"></div>
<script>
window.__taskState = {{success:false, value:""}};
document.querySelector('#select').onclick = () => {{ document.querySelector('#panel').style.display = 'block'; }};
document.querySelector('#target').onclick = () => {{ window.__taskState.value = {js(choice)}; document.querySelector('#panel').style.display = 'none'; }};
document.querySelector('#submit').onclick = () => {{ window.__taskState.success = window.__taskState.value === {js(choice)}; document.querySelector('#status').textContent = window.__taskState.success ? "saved" : "not yet"; }};
</script>
""",
        seed,
    )
    return base_task(
        task_id=f"suite_choice_select_{seed:03d}",
        goal=f"Choose location {choice}, then submit.",
        html=html,
        template="choice_select",
        seed=seed,
        max_steps=5,
        verifier={
            "success_js": "window.__taskState && window.__taskState.success === true",
            "progress_js": {"selected": f"window.__taskState && window.__taskState.value === {js(choice)}"},
        },
        oracle_actions=[
            {"action": "click", "selector": "#select"},
            {"action": "click", "selector": "#target"},
            {"action": "click", "selector": "#submit"},
        ],
        family="choice",
    )


def advanced_task(seed: int) -> BrowserTaskSpec:
    subtype = seed % 3
    if subtype == 0:
        return dialog_task(seed)
    if subtype == 1:
        return scroll_task(seed)
    return tab_task(seed)


def dialog_task(seed: int) -> BrowserTaskSpec:
    label = f"release {seed:03d}"
    html = page(
        "Dialog",
        f"""
<button id="open">Open dialog</button>
<div id="dialog" style="display:none;margin-top:18px;border:1px solid #94a3b8;padding:12px">
  <p>Confirm {label}</p>
  <button id="confirm">Confirm</button>
</div>
<div id="status" class="status"></div>
<script>
window.__taskState = {{success:false}};
document.querySelector('#open').onclick = () => {{ document.querySelector('#dialog').style.display = 'block'; }};
document.querySelector('#confirm').onclick = () => {{ window.__taskState.success = true; document.querySelector('#status').textContent = "confirmed"; }};
</script>
""",
        seed,
    )
    return base_task(
        task_id=f"suite_advanced_dialog_{seed:03d}",
        goal=f"Open the dialog and confirm {label}.",
        html=html,
        template="advanced_dialog",
        seed=seed,
        max_steps=4,
        verifier={
            "success_js": "window.__taskState && window.__taskState.success === true",
            "progress_js": {"dialog_open": "document.querySelector('#dialog').style.display === 'block'"},
        },
        oracle_actions=[{"action": "click", "selector": "#open"}, {"action": "click", "selector": "#confirm"}],
        family="advanced",
    )


def scroll_task(seed: int) -> BrowserTaskSpec:
    target_visible_js = """
(() => {
  const target = document.querySelector('#target');
  if (!target) return false;
  const rect = target.getBoundingClientRect();
  return rect.top < window.innerHeight && rect.bottom > 0;
})()
""".strip()
    html = f"""<!doctype html><html><head><meta charset='utf-8'>{style(seed)}</head><body>
<main class='app'><h1>Scroll</h1>
<p class='muted'>The target is below the fold.</p>
<div style='height:{760 + (seed % 4) * 90}px'></div>
<button id="target">Open details {seed:03d}</button>
<div id="status" class="status"></div>
<script>
window.__taskState = {{success:false}};
document.querySelector('#target').onclick = () => {{ window.__taskState.success = true; document.querySelector('#status').textContent = "opened"; }};
</script>
</main></body></html>"""
    return base_task(
        task_id=f"suite_advanced_scroll_{seed:03d}",
        goal=f"Scroll down and open details {seed:03d}.",
        html=html,
        template="advanced_scroll",
        seed=seed,
        max_steps=4,
        verifier={
            "success_js": "window.__taskState && window.__taskState.success === true",
            "progress_js": {
                "scrolled_down": "window.scrollY >= 300",
                "target_visible": target_visible_js,
            },
        },
        oracle_actions=[{"action": "scroll", "dy": 900}, {"action": "click", "selector": "#target"}],
        family="advanced",
    )


def tab_task(seed: int) -> BrowserTaskSpec:
    html = page(
        "Tabs",
        f"""
<button id="tab_home">Home</button>
<button id="tab_target">Settings</button>
<div id="panel" style="margin-top:16px"><span class="muted">Home panel</span></div>
<div id="status" class="status"></div>
<script>
window.__taskState = {{success:false}};
document.querySelector('#tab_target').onclick = () => {{
  document.querySelector('#panel').innerHTML = '<button id="target">Enable sync {seed:03d}</button>';
  document.querySelector('#target').onclick = () => {{ window.__taskState.success = true; document.querySelector('#status').textContent = "enabled"; }};
}};
</script>
""",
        seed,
    )
    return base_task(
        task_id=f"suite_advanced_tab_{seed:03d}",
        goal=f"Open the Settings tab and enable sync {seed:03d}.",
        html=html,
        template="advanced_tab",
        seed=seed,
        max_steps=4,
        verifier={
            "success_js": "window.__taskState && window.__taskState.success === true",
            "progress_js": {"tab_open": "document.querySelector('#target') !== null"},
        },
        oracle_actions=[{"action": "click", "selector": "#tab_target"}, {"action": "click", "selector": "#target"}],
        family="advanced",
    )


BUILDERS = {
    "form": form_task,
    "search": search_task,
    "menu": menu_task,
    "table": table_task,
    "todo": todo_task,
    "choice": choice_task,
    "advanced": advanced_task,
}


def stratified_split(
    tasks: list[BrowserTaskSpec],
    *,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[BrowserTaskSpec], list[BrowserTaskSpec], list[BrowserTaskSpec]]:
    # Split by template rather than only by family. Some families contain
    # structurally different templates, e.g. choice_checkbox/radio/select.
    # Family-only sorted splits can put entire subtemplates only in train or
    # only in val/test, which makes evaluation misleading.
    by_family: dict[str, list[BrowserTaskSpec]] = defaultdict(list)
    for task in tasks:
        by_family[str(task.template)].append(task)
    train: list[BrowserTaskSpec] = []
    val: list[BrowserTaskSpec] = []
    test: list[BrowserTaskSpec] = []
    for family_tasks in by_family.values():
        ordered = sorted(family_tasks, key=lambda item: item.task_id)
        n = len(ordered)
        train_n = int(round(n * train_ratio))
        val_n = int(round(n * val_ratio))
        train_n = min(max(1, train_n), max(1, n - 2)) if n >= 3 else max(0, n - 1)
        val_n = min(max(1, val_n), n - train_n - 1) if n - train_n >= 2 else max(0, n - train_n)
        train.extend(ordered[:train_n])
        val.extend(ordered[train_n : train_n + val_n])
        test.extend(ordered[train_n + val_n :])
    for split, name in [(train, "train"), (val, "val"), (test, "test")]:
        for task in split:
            task.split = name
    return train, val, test


if __name__ == "__main__":
    main()
