#!/usr/bin/env python3
"""Build deterministic local browser-RL smoke tasks."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.browser_rl import BrowserTaskSpec, write_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/browser_rl_smoke_tasks")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--count-per-template", type=int, default=4)
    return parser.parse_args()


BASE_STYLE = """
<style>
body { font-family: Arial, sans-serif; margin: 0; background: #f4f6f8; color: #1f2937; }
.app { width: 760px; margin: 48px auto; background: white; border: 1px solid #d6dde5; border-radius: 8px; padding: 28px; }
h1 { font-size: 22px; margin: 0 0 20px; }
button { height: 38px; padding: 0 16px; border: 1px solid #64748b; background: #eef2f7; border-radius: 6px; cursor: pointer; }
input { height: 34px; padding: 0 10px; border: 1px solid #94a3b8; border-radius: 6px; min-width: 260px; }
.row { display: flex; gap: 12px; align-items: center; margin: 12px 0; }
.result, .menu-item { display: block; width: 320px; margin: 8px 0; text-align: left; }
table { border-collapse: collapse; width: 100%; }
td, th { border: 1px solid #cbd5e1; padding: 10px; text-align: left; }
.status { margin-top: 18px; min-height: 24px; color: #0f766e; font-weight: 700; }
</style>
"""


def page(title: str, body: str) -> str:
    return f"<!doctype html><html><head><meta charset='utf-8'>{BASE_STYLE}</head><body><main class='app'><h1>{title}</h1>{body}</main></body></html>"


def form_task(seed: int) -> BrowserTaskSpec:
    name = f"user_{seed}"
    code = f"pass-{seed * 17}"
    html = page(
        "Account form",
        f"""
<div class="row"><label>Name</label><input id="name" autocomplete="off"></div>
<div class="row"><label>Code</label><input id="code" autocomplete="off"></div>
<button id="submit">Submit</button>
<div id="status" class="status"></div>
<script>
window.__taskState = {{success:false}};
document.querySelector('#submit').onclick = () => {{
  const ok = document.querySelector('#name').value === "{name}" && document.querySelector('#code').value === "{code}";
  window.__taskState.success = ok;
  document.querySelector('#status').textContent = ok ? "submitted" : "not yet";
}};
</script>
""",
    )
    return BrowserTaskSpec(
        task_id=f"smoke_form_{seed:02d}",
        goal=f"Fill Name with {name}, Code with {code}, then submit.",
        html=html,
        template="form_fill",
        seed=seed,
        max_steps=6,
        verifier={
            "success_js": "window.__taskState && window.__taskState.success === true",
            "progress_js": {
                "name_value": f"document.querySelector('#name').value === '{name}'",
                "code_value": f"document.querySelector('#code').value === '{code}'",
            },
        },
        oracle_actions=[
            {"action": "click", "selector": "#name"},
            {"action": "type", "text": name},
            {"action": "click", "selector": "#code"},
            {"action": "type", "text": code},
            {"action": "click", "selector": "#submit"},
        ],
    )


def todo_task(seed: int) -> BrowserTaskSpec:
    item = f"todo item {seed}"
    html = page(
        "Todo",
        f"""
<div class="row"><input id="new_item" autocomplete="off"><button id="add">Add</button></div>
<ul id="items"></ul><div id="status" class="status"></div>
<script>
window.__taskState = {{success:false}};
function addItem() {{
  const li = document.createElement('li');
  li.textContent = new_item.value;
  items.appendChild(li);
  window.__taskState.success = new_item.value === "{item}";
  status.textContent = window.__taskState.success ? "added" : "";
}}
add.onclick = addItem;
new_item.addEventListener('keydown', e => {{ if (e.key === 'Enter') addItem(); }});
</script>
""",
    )
    return BrowserTaskSpec(
        task_id=f"smoke_todo_{seed:02d}",
        goal=f"Create a todo item named {item}.",
        html=html,
        template="todo_add",
        seed=seed,
        max_steps=5,
        verifier={
            "success_js": "window.__taskState && window.__taskState.success === true",
            "progress_js": {"typed_item": f"document.querySelector('#new_item').value === '{item}'"},
        },
        oracle_actions=[
            {"action": "click", "selector": "#new_item"},
            {"action": "type", "text": item},
            {"action": "press", "key": "Enter"},
        ],
    )


def search_task(seed: int) -> BrowserTaskSpec:
    query = f"report {seed}"
    target = f"Open report {seed}"
    html = page(
        "Search",
        f"""
<div class="row"><input id="query" autocomplete="off"><button id="search">Search</button></div>
<div id="results"></div><div id="status" class="status"></div>
<script>
window.__taskState = {{success:false}};
search.onclick = () => {{
  results.innerHTML = '<button class="result" id="target">{target}</button><button class="result">Archive</button>';
  target.onclick = () => {{ window.__taskState.success = query.value === "{query}"; status.textContent = "opened"; }};
}};
</script>
""",
    )
    return BrowserTaskSpec(
        task_id=f"smoke_search_{seed:02d}",
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
    )


def menu_task(seed: int) -> BrowserTaskSpec:
    choice = ["Export CSV", "Archive Project", "Share Link", "Open Settings"][seed % 4]
    html = page(
        "Menu",
        f"""
<button id="menu">Actions</button>
<div id="panel" style="display:none;margin-top:12px">
  <button class="menu-item">Duplicate</button>
  <button class="menu-item" id="target">{choice}</button>
  <button class="menu-item">Delete</button>
</div>
<div id="status" class="status"></div>
<script>
window.__taskState = {{success:false}};
menu.onclick = () => {{ panel.style.display = 'block'; }};
target.onclick = () => {{ window.__taskState.success = true; status.textContent = "{choice}"; }};
</script>
""",
    )
    return BrowserTaskSpec(
        task_id=f"smoke_menu_{seed:02d}",
        goal=f"Open the Actions menu and choose {choice}.",
        html=html,
        template="menu_select",
        seed=seed,
        max_steps=4,
        verifier={
            "success_js": "window.__taskState && window.__taskState.success === true",
            "progress_js": {"menu_open": "document.querySelector('#panel').style.display === 'block'"},
        },
        oracle_actions=[
            {"action": "click", "selector": "#menu"},
            {"action": "click", "selector": "#target"},
        ],
    )


def table_task(seed: int) -> BrowserTaskSpec:
    row_id = seed + 10
    html = page(
        "Table",
        f"""
<table>
<tr><th>ID</th><th>Name</th><th>Action</th></tr>
<tr><td>{row_id - 1}</td><td>Draft</td><td><button>Select</button></td></tr>
<tr><td>{row_id}</td><td>Target row</td><td><button id="target">Select</button></td></tr>
<tr><td>{row_id + 1}</td><td>Archive</td><td><button>Select</button></td></tr>
</table>
<div id="status" class="status"></div>
<script>
window.__taskState = {{success:false}};
target.onclick = () => {{ window.__taskState.success = true; status.textContent = "selected {row_id}"; }};
</script>
""",
    )
    return BrowserTaskSpec(
        task_id=f"smoke_table_{seed:02d}",
        goal=f"In the table, select the row with ID {row_id}.",
        html=html,
        template="table_action",
        seed=seed,
        max_steps=3,
        verifier={"success_js": "window.__taskState && window.__taskState.success === true"},
        oracle_actions=[{"action": "click", "selector": "#target"}],
    )


BUILDERS = [form_task, todo_task, search_task, menu_task, table_task]


def main() -> None:
    args = parse_args()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[BrowserTaskSpec] = []
    for builder in BUILDERS:
        for seed in range(1, args.count_per_template + 1):
            tasks.append(builder(seed))
    train, val, test = tasks[:14], tasks[14:18], tasks[18:]
    write_tasks(output_dir / "all_tasks.jsonl", tasks)
    write_tasks(output_dir / "train_tasks.jsonl", train)
    write_tasks(output_dir / "val_tasks.jsonl", val)
    write_tasks(output_dir / "test_tasks.jsonl", test)
    summary = {
        "created_at": timestamp,
        "count": len(tasks),
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "templates": {builder.__name__.replace("_task", ""): args.count_per_template for builder in BUILDERS},
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
