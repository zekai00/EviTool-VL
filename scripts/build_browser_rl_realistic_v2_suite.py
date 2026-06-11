#!/usr/bin/env python3
"""Build a more realistic local BrowserRL task suite.

The suite keeps the existing BrowserTaskSpec JSONL schema so current oracle,
SFT, rollout, and GRPO scripts can consume it without pipeline changes. The
HTML pages are still self-contained and deterministic, but the layouts mimic
common SaaS/admin UI patterns: filters, tables, modals, dropdowns, autocomplete,
and multi-panel workflows with hard negatives.
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
    "form": 100,
    "table": 100,
    "search": 100,
    "modal": 100,
    "select": 100,
    "workflow": 100,
}
VIEWPORT = (1280, 720)
ACTION_SPACE = ["click", "type", "press", "scroll", "wait", "finish"]


NAMES = [
    "Maya Chen",
    "Noah Rivera",
    "Iris Patel",
    "Evan Brooks",
    "Lina Morgan",
    "Owen Park",
    "Nora Singh",
    "Caleb Stone",
]
DEPARTMENTS = ["Finance", "Support", "Operations", "Research", "Compliance", "Logistics"]
CUSTOMERS = ["Nova Labs", "Atlas Studio", "Northwind Co", "Blue Harbor", "Pioneer Health", "Cedar Foods"]
PRODUCTS = ["Ledger Pro", "Signal Hub", "Atlas Note", "Nova Switch", "Cedar Desk", "Harbor Dock"]
OWNERS = ["Ari Kim", "Dana Lopez", "Mila Stone", "Theo Grant", "Jules Reed", "Rina Shah"]
STATUSES = ["Approved", "Blocked", "In review", "Escalated", "Ready"]


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
        counts.update({str(key): int(value) for key, value in json.loads(args.counts_json).items()})
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
        "suite": "browser_rl_realistic_v2",
        "count": len(tasks),
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "counts": dict(Counter(task.metadata.get("family", task.template) for task in tasks)),
        "templates": dict(Counter(task.template for task in tasks)),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


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
    difficulty: int = 2,
) -> BrowserTaskSpec:
    return BrowserTaskSpec(
        task_id=task_id,
        goal=goal,
        html=html,
        app="local_realistic_v2",
        template=template,
        seed=seed,
        split="train",
        difficulty=difficulty,
        viewport=VIEWPORT,
        max_steps=max_steps,
        action_space=list(ACTION_SPACE),
        verifier=verifier,
        oracle_actions=oracle_actions,
        metadata={"family": family, "suite": "browser_rl_realistic_v2"},
    )


def style(seed: int) -> str:
    accent = ["#2563eb", "#0f766e", "#7c3aed", "#b45309", "#be123c"][seed % 5]
    width = 900 + (seed % 4) * 40
    return f"""
<style>
:root {{ --accent: {accent}; --line: #cbd5e1; --muted: #64748b; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: Arial, sans-serif; background: #eef2f7; color: #172033; }}
.shell {{ min-height: 100vh; display: grid; grid-template-columns: 190px 1fr; }}
.side {{ background: #111827; color: white; padding: 22px 18px; }}
.brand {{ font-size: 17px; font-weight: 700; margin-bottom: 26px; }}
.nav {{ display: grid; gap: 9px; color: #cbd5e1; font-size: 13px; }}
.main {{ padding: 28px 34px; }}
.topbar {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px; }}
h1 {{ font-size: 22px; margin: 0; }}
.hint {{ color: var(--muted); font-size: 13px; }}
.panel {{ width: {width}px; background: white; border: 1px solid var(--line); border-radius: 8px; padding: 20px; box-shadow: 0 1px 2px rgba(15,23,42,.08); }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px 18px; }}
.row {{ display: flex; gap: 10px; align-items: center; margin: 10px 0; }}
.field {{ display: grid; gap: 6px; }}
label {{ color: #334155; font-size: 13px; font-weight: 700; }}
input {{ height: 36px; border: 1px solid #94a3b8; border-radius: 6px; padding: 0 10px; min-width: 220px; }}
button {{ min-height: 36px; padding: 0 13px; border: 1px solid #64748b; border-radius: 6px; background: #f8fafc; cursor: pointer; }}
button.primary {{ color: white; background: var(--accent); border-color: var(--accent); }}
button.ghost {{ background: white; color: #334155; }}
.toolbar {{ display: flex; gap: 10px; align-items: center; margin-bottom: 14px; }}
.dropdown, .suggestions {{ display: none; position: absolute; z-index: 4; background: white; border: 1px solid var(--line); border-radius: 6px; padding: 6px; box-shadow: 0 8px 20px rgba(15,23,42,.12); }}
.dropdown.open, .suggestions.open {{ display: grid; gap: 5px; }}
.option {{ display: block; min-width: 190px; text-align: left; background: white; border: 0; }}
table {{ width: 100%; border-collapse: collapse; background: white; }}
th {{ position: sticky; top: 0; background: #f8fafc; z-index: 1; }}
td, th {{ border: 1px solid var(--line); padding: 9px 10px; text-align: left; font-size: 13px; }}
.cards {{ display: grid; gap: 10px; margin-top: 12px; }}
.card {{ border: 1px solid var(--line); border-radius: 8px; padding: 12px; display: flex; justify-content: space-between; align-items: center; }}
.modal-backdrop {{ display: none; position: fixed; inset: 0; background: rgba(15,23,42,.34); align-items: center; justify-content: center; }}
.modal-backdrop.open {{ display: flex; }}
.modal {{ width: 480px; background: white; border-radius: 8px; padding: 20px; box-shadow: 0 16px 40px rgba(15,23,42,.25); }}
.toast {{ min-height: 26px; margin-top: 14px; color: #0f766e; font-weight: 700; }}
.danger {{ color: #be123c; }}
.hidden {{ display: none !important; }}
.spacer {{ height: {8 + (seed % 5) * 9}px; }}
</style>
"""


def page(title: str, body: str, seed: int) -> str:
    return f"""<!doctype html>
<html>
<head><meta charset="utf-8">{style(seed)}</head>
<body>
<div class="shell">
  <aside class="side">
    <div class="brand">Northstar Admin</div>
    <div class="nav"><span>Dashboard</span><span>Customers</span><span>Orders</span><span>Settings</span></div>
  </aside>
  <main class="main">
    <div class="topbar"><h1>{title}</h1><span class="hint">Workspace {seed % 17 + 1}</span></div>
    <section class="panel">{body}</section>
  </main>
</div>
</body>
</html>"""


def js(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def pick(values: list[str], seed: int, offset: int = 0) -> str:
    return values[(seed + offset) % len(values)]


def form_task(seed: int) -> BrowserTaskSpec:
    name = pick(NAMES, seed)
    department = pick(DEPARTMENTS, seed, 2)
    decoy_department = pick(DEPARTMENTS, seed, 3)
    html = page(
        "Contact editor",
        f"""
<div class="spacer"></div>
<div class="grid">
  <div class="field"><label>Contact Name</label><input id="contact_name" placeholder="Full name"></div>
  <div class="field"><label>Company Name</label><input id="company_name" placeholder="Do not edit"></div>
  <div class="field"><label>Display Name</label><input id="display_name" placeholder="Optional"></div>
  <div class="field" style="position:relative"><label>Department</label>
    <button id="department_button" class="ghost">Choose department</button>
    <div id="department_menu" class="dropdown">
      <button class="option">{decoy_department}</button>
      <button id="dept_target" class="option">{department}</button>
      <button class="option">{pick(DEPARTMENTS, seed, 4)}</button>
    </div>
  </div>
</div>
<div class="row">
  <button id="cancel">Cancel</button>
  <button id="save_draft">Save draft</button>
  <button id="save" class="primary">Save</button>
</div>
<div id="toast" class="toast"></div>
<script>
window.__taskState = {{success:false, department:null, saved:false}};
document.querySelector('#department_button').onclick = () => document.querySelector('#department_menu').classList.toggle('open');
document.querySelector('#dept_target').onclick = () => {{
  window.__taskState.department = {js(department)};
  document.querySelector('#department_button').textContent = {js(department)};
  document.querySelector('#department_menu').classList.remove('open');
}};
document.querySelector('#save').onclick = () => {{
  const ok = document.querySelector('#contact_name').value === {js(name)} && window.__taskState.department === {js(department)};
  window.__taskState.saved = true;
  window.__taskState.success = ok;
  document.querySelector('#toast').textContent = ok ? 'Contact saved' : 'Check required fields';
}};
</script>
""",
        seed,
    )
    return base_task(
        task_id=f"realistic_form_{seed:03d}",
        goal=f"Update Contact Name to {name}, Department to {department}, then save.",
        html=html,
        template="realistic_form_validation",
        seed=seed,
        max_steps=7,
        verifier={
            "success_js": "window.__taskState && window.__taskState.success === true",
            "progress_js": {
                "name_value": f"document.querySelector('#contact_name').value === {js(name)}",
                "department_selected": f"window.__taskState && window.__taskState.department === {js(department)}",
                "saved": "window.__taskState && window.__taskState.saved === true",
            },
        },
        oracle_actions=[
            {"action": "click", "selector": "#contact_name"},
            {"action": "type", "text": name},
            {"action": "click", "selector": "#department_button"},
            {"action": "click", "selector": "#dept_target"},
            {"action": "click", "selector": "#save"},
        ],
        family="form",
    )


def table_task(seed: int) -> BrowserTaskSpec:
    customer = pick(CUSTOMERS, seed)
    invoice = f"INV-{10000 + seed * 17}"
    similar_invoice = f"INV-{10000 + seed * 17 + 6}"
    rows = [
        (f"INV-{10000 + seed * 11}", pick(CUSTOMERS, seed, 1), "$420"),
        (similar_invoice, customer.replace("Labs", "Lab"), "$918"),
        (invoice, customer, f"${500 + seed * 3}"),
        (f"INV-{10000 + seed * 17 + 9}", pick(CUSTOMERS, seed, 2), "$305"),
    ]
    row_parts = []
    for row_invoice, row_customer, amount in rows:
        target_attr = ' id="target"' if row_invoice == invoice else ""
        row_parts.append(
            f"<tr data-customer={js(row_customer)}><td>{row_invoice}</td><td>{row_customer}</td><td>{amount}</td>"
            f"<td><button{target_attr}>Open</button></td></tr>"
        )
    row_html = "\n".join(row_parts)
    html = page(
        "Invoice table",
        f"""
<div class="toolbar">
  <input id="filter" placeholder="Filter customer">
  <button id="apply_filter">Apply filter</button>
  <button id="reset_filter">Reset</button>
</div>
<div style="max-height:360px; overflow:auto; border:1px solid #cbd5e1">
<table>
  <thead><tr><th>Invoice</th><th>Customer</th><th>Total</th><th>Action</th></tr></thead>
  <tbody>{row_html}</tbody>
</table>
</div>
<div id="toast" class="toast"></div>
<script>
window.__taskState = {{success:false, filtered:false, opened:null}};
document.querySelector('#apply_filter').onclick = () => {{
  const value = document.querySelector('#filter').value.toLowerCase();
  document.querySelectorAll('tbody tr').forEach(row => {{
    row.style.display = row.dataset.customer.toLowerCase().includes(value) ? '' : 'none';
  }});
  window.__taskState.filtered = value.includes({js(customer.lower())});
}};
document.querySelectorAll('tbody button').forEach(button => button.onclick = () => {{
  const row = button.closest('tr');
  window.__taskState.opened = row.children[0].textContent;
  const ok = window.__taskState.opened === {js(invoice)} && window.__taskState.filtered === true;
  window.__taskState.success = ok;
  document.querySelector('#toast').textContent = ok ? 'Invoice opened' : 'Wrong invoice';
}});
</script>
""",
        seed,
    )
    return base_task(
        task_id=f"realistic_table_{seed:03d}",
        goal=f"Filter customer {customer}, then open invoice {invoice}.",
        html=html,
        template="realistic_table_filter_open",
        seed=seed,
        max_steps=7,
        verifier={
            "success_js": "window.__taskState && window.__taskState.success === true",
            "progress_js": {
                "filter_value": f"document.querySelector('#filter').value === {js(customer)}",
                "filtered": "window.__taskState && window.__taskState.filtered === true",
                "target_opened": f"window.__taskState && window.__taskState.opened === {js(invoice)}",
            },
        },
        oracle_actions=[
            {"action": "click", "selector": "#filter"},
            {"action": "type", "text": customer},
            {"action": "click", "selector": "#apply_filter"},
            {"action": "click", "selector": "#target"},
        ],
        family="table",
    )


def search_task(seed: int) -> BrowserTaskSpec:
    query = pick(PRODUCTS, seed)
    category = pick(["Hardware", "Software", "Archive", "Services"], seed)
    target = f"{query} {seed % 9 + 1}"
    html = page(
        "Catalog search",
        f"""
<div class="toolbar">
  <input id="search_box" placeholder="Search catalog">
  <div style="position:relative">
    <button id="category_button">Category</button>
    <div id="category_menu" class="dropdown">
      <button class="option">Archive</button>
      <button id="category_target" class="option">{category}</button>
      <button class="option">General</button>
    </div>
  </div>
  <button id="run_search" class="primary">Search</button>
</div>
<div id="loading" class="hint hidden">Loading results...</div>
<div id="results" class="cards hidden">
  <div class="card"><span>{query} legacy</span><button>Open</button></div>
  <div class="card"><span id="target_label">{target}</span><button id="target">Open</button></div>
  <div class="card"><span>{query.replace(' ', '')}</span><button>Open</button></div>
</div>
<div id="toast" class="toast"></div>
<script>
window.__taskState = {{success:false, category:null, results:false, opened:null}};
document.querySelector('#category_button').onclick = () => document.querySelector('#category_menu').classList.toggle('open');
document.querySelector('#category_target').onclick = () => {{
  window.__taskState.category = {js(category)};
  document.querySelector('#category_button').textContent = {js(category)};
  document.querySelector('#category_menu').classList.remove('open');
}};
document.querySelector('#run_search').onclick = () => {{
  const ok = document.querySelector('#search_box').value === {js(query)} && window.__taskState.category === {js(category)};
  window.__taskState.results = ok;
  document.querySelector('#results').classList.toggle('hidden', !ok);
  document.querySelector('#loading').classList.add('hidden');
}};
document.querySelector('#target').onclick = () => {{
  window.__taskState.opened = {js(target)};
  window.__taskState.success = window.__taskState.results === true;
  document.querySelector('#toast').textContent = window.__taskState.success ? 'Result opened' : 'Search first';
}};
</script>
""",
        seed,
    )
    return base_task(
        task_id=f"realistic_search_{seed:03d}",
        goal=f"Search for {query}, choose category {category}, then open result {target}.",
        html=html,
        template="realistic_search_filter_result",
        seed=seed,
        max_steps=8,
        verifier={
            "success_js": "window.__taskState && window.__taskState.success === true",
            "progress_js": {
                "query_value": f"document.querySelector('#search_box').value === {js(query)}",
                "category_selected": f"window.__taskState && window.__taskState.category === {js(category)}",
                "results_shown": "window.__taskState && window.__taskState.results === true",
                "target_opened": f"window.__taskState && window.__taskState.opened === {js(target)}",
            },
        },
        oracle_actions=[
            {"action": "click", "selector": "#search_box"},
            {"action": "type", "text": query},
            {"action": "click", "selector": "#category_button"},
            {"action": "click", "selector": "#category_target"},
            {"action": "click", "selector": "#run_search"},
            {"action": "click", "selector": "#target"},
        ],
        family="search",
    )


def modal_task(seed: int) -> BrowserTaskSpec:
    label = f"case-{seed:03d}-urgent"
    html = page(
        "Case settings",
        f"""
<div class="card"><span>Primary case</span><button id="open_settings">Settings</button></div>
<div class="card"><span>Archived case</span><button>Settings</button></div>
<div id="modal_backdrop" class="modal-backdrop">
  <div class="modal">
    <h2 style="margin-top:0">Edit case</h2>
    <div class="field"><label>Case Label</label><input id="case_label" placeholder="Label"></div>
    <div class="field"><label>Internal Note</label><input id="internal_note" placeholder="Do not edit"></div>
    <div class="row">
      <button id="close_modal">Cancel</button>
      <button id="confirm" class="primary">Confirm changes</button>
    </div>
  </div>
</div>
<div id="toast" class="toast"></div>
<script>
window.__taskState = {{success:false, modal:false, confirmed:false}};
document.querySelector('#open_settings').onclick = () => {{
  window.__taskState.modal = true;
  document.querySelector('#modal_backdrop').classList.add('open');
}};
document.querySelector('#close_modal').onclick = () => {{
  window.__taskState.modal = false;
  document.querySelector('#modal_backdrop').classList.remove('open');
}};
document.querySelector('#confirm').onclick = () => {{
  window.__taskState.confirmed = true;
  const ok = document.querySelector('#case_label').value === {js(label)};
  window.__taskState.success = ok;
  document.querySelector('#toast').textContent = ok ? 'Settings updated' : 'Label missing';
}};
</script>
""",
        seed,
    )
    return base_task(
        task_id=f"realistic_modal_{seed:03d}",
        goal=f"Open Settings for the primary case, change Case Label to {label}, then confirm changes.",
        html=html,
        template="realistic_modal_edit_confirm",
        seed=seed,
        max_steps=6,
        verifier={
            "success_js": "window.__taskState && window.__taskState.success === true",
            "progress_js": {
                "modal_open": "window.__taskState && window.__taskState.modal === true",
                "label_value": f"document.querySelector('#case_label').value === {js(label)}",
                "confirmed": "window.__taskState && window.__taskState.confirmed === true",
            },
        },
        oracle_actions=[
            {"action": "click", "selector": "#open_settings"},
            {"action": "click", "selector": "#case_label"},
            {"action": "type", "text": label},
            {"action": "click", "selector": "#confirm"},
        ],
        family="modal",
    )


def select_task(seed: int) -> BrowserTaskSpec:
    owner = pick(OWNERS, seed)
    due = f"2026-07-{10 + seed % 18:02d}"
    html = page(
        "Task assignment",
        f"""
<div class="grid">
  <div class="field" style="position:relative"><label>Owner</label>
    <input id="owner_input" placeholder="Type owner name">
    <div id="owner_suggestions" class="suggestions">
      <button class="option">{pick(OWNERS, seed, 1)}</button>
      <button id="owner_target" class="option">{owner}</button>
      <button class="option">{pick(OWNERS, seed, 2)}</button>
    </div>
  </div>
  <div class="field"><label>Due date</label><input id="due_date" placeholder="YYYY-MM-DD"></div>
  <div class="field"><label>Summary</label><input id="summary" placeholder="Do not edit"></div>
</div>
<div class="row">
  <button id="save_draft">Save draft</button>
  <button id="save" class="primary">Save assignment</button>
</div>
<div id="toast" class="toast"></div>
<script>
window.__taskState = {{success:false, owner:null, saved:false}};
document.querySelector('#owner_input').oninput = () => document.querySelector('#owner_suggestions').classList.add('open');
document.querySelector('#owner_input').onclick = () => document.querySelector('#owner_suggestions').classList.add('open');
document.querySelector('#owner_target').onclick = () => {{
  window.__taskState.owner = {js(owner)};
  document.querySelector('#owner_input').value = {js(owner)};
  document.querySelector('#owner_suggestions').classList.remove('open');
}};
document.querySelector('#save').onclick = () => {{
  window.__taskState.saved = true;
  const ok = window.__taskState.owner === {js(owner)} && document.querySelector('#due_date').value === {js(due)};
  window.__taskState.success = ok;
  document.querySelector('#toast').textContent = ok ? 'Assignment saved' : 'Missing assignment data';
}};
</script>
""",
        seed,
    )
    return base_task(
        task_id=f"realistic_select_{seed:03d}",
        goal=f"Assign Owner to {owner}, Due date to {due}, then save assignment.",
        html=html,
        template="realistic_autocomplete_date_save",
        seed=seed,
        max_steps=8,
        verifier={
            "success_js": "window.__taskState && window.__taskState.success === true",
            "progress_js": {
                "owner_selected": f"window.__taskState && window.__taskState.owner === {js(owner)}",
                "owner_input_value": f"document.querySelector('#owner_input').value === {js(owner)}",
                "due_value": f"document.querySelector('#due_date').value === {js(due)}",
                "saved": "window.__taskState && window.__taskState.saved === true",
            },
        },
        oracle_actions=[
            {"action": "click", "selector": "#owner_input"},
            {"action": "type", "text": owner},
            {"action": "click", "selector": "#owner_target"},
            {"action": "click", "selector": "#due_date"},
            {"action": "type", "text": due},
            {"action": "click", "selector": "#save"},
        ],
        family="select",
    )


def workflow_task(seed: int) -> BrowserTaskSpec:
    account = pick(CUSTOMERS, seed, 3)
    status = pick(STATUSES, seed, 1)
    html = page(
        "Account workflow",
        f"""
<div id="lookup">
  <div class="toolbar">
    <input id="account_search" placeholder="Find account">
    <button id="find_account" class="primary">Find</button>
  </div>
  <div id="results" class="cards hidden">
    <div class="card"><span>{account.replace('Labs', 'Lab')}</span><button>Open</button></div>
    <div class="card"><span>{account}</span><button id="open_target">Open profile</button></div>
  </div>
</div>
<div id="profile" class="hidden">
  <h2 style="margin:0 0 10px">Profile: {account}</h2>
  <div class="row" style="position:relative">
    <label>Status</label>
    <button id="status_button">Choose status</button>
    <div id="status_menu" class="dropdown">
      <button class="option">{pick(STATUSES, seed, 2)}</button>
      <button id="status_target" class="option">{status}</button>
      <button class="option">{pick(STATUSES, seed, 3)}</button>
    </div>
  </div>
  <button id="save" class="primary">Save profile</button>
</div>
<div id="toast" class="toast"></div>
<script>
window.__taskState = {{success:false, searched:false, profile:false, status:null, saved:false}};
document.querySelector('#find_account').onclick = () => {{
  const ok = document.querySelector('#account_search').value === {js(account)};
  window.__taskState.searched = ok;
  document.querySelector('#results').classList.toggle('hidden', !ok);
}};
document.querySelector('#open_target').onclick = () => {{
  if (!window.__taskState.searched) return;
  window.__taskState.profile = true;
  document.querySelector('#profile').classList.remove('hidden');
}};
document.querySelector('#status_button').onclick = () => document.querySelector('#status_menu').classList.toggle('open');
document.querySelector('#status_target').onclick = () => {{
  window.__taskState.status = {js(status)};
  document.querySelector('#status_button').textContent = {js(status)};
  document.querySelector('#status_menu').classList.remove('open');
}};
document.querySelector('#save').onclick = () => {{
  window.__taskState.saved = true;
  const ok = window.__taskState.searched && window.__taskState.profile && window.__taskState.status === {js(status)};
  window.__taskState.success = ok;
  document.querySelector('#toast').textContent = ok ? 'Profile saved' : 'Workflow incomplete';
}};
</script>
""",
        seed,
    )
    return base_task(
        task_id=f"realistic_workflow_{seed:03d}",
        goal=f"Find account {account}, change Status to {status}, then save the profile.",
        html=html,
        template="realistic_account_status_workflow",
        seed=seed,
        max_steps=9,
        verifier={
            "success_js": "window.__taskState && window.__taskState.success === true",
            "progress_js": {
                "searched": "window.__taskState && window.__taskState.searched === true",
                "profile_open": "window.__taskState && window.__taskState.profile === true",
                "status_selected": f"window.__taskState && window.__taskState.status === {js(status)}",
                "saved": "window.__taskState && window.__taskState.saved === true",
            },
        },
        oracle_actions=[
            {"action": "click", "selector": "#account_search"},
            {"action": "type", "text": account},
            {"action": "click", "selector": "#find_account"},
            {"action": "click", "selector": "#open_target"},
            {"action": "click", "selector": "#status_button"},
            {"action": "click", "selector": "#status_target"},
            {"action": "click", "selector": "#save"},
        ],
        family="workflow",
        difficulty=3,
    )


def stratified_split(
    tasks: list[BrowserTaskSpec],
    *,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[BrowserTaskSpec], list[BrowserTaskSpec], list[BrowserTaskSpec]]:
    by_template: dict[str, list[BrowserTaskSpec]] = defaultdict(list)
    for task in tasks:
        by_template[str(task.template)].append(task)
    train: list[BrowserTaskSpec] = []
    val: list[BrowserTaskSpec] = []
    test: list[BrowserTaskSpec] = []
    for template_tasks in by_template.values():
        ordered = sorted(template_tasks, key=lambda item: item.task_id)
        n = len(ordered)
        train_n = int(round(n * train_ratio))
        val_n = int(round(n * val_ratio))
        train_n = min(max(1, train_n), max(1, n - 2)) if n >= 3 else max(0, n - 1)
        val_n = min(max(1, val_n), n - train_n - 1) if n - train_n >= 2 else max(0, n - train_n)
        train.extend(ordered[:train_n])
        val.extend(ordered[train_n : train_n + val_n])
        test.extend(ordered[train_n + val_n :])
    for rows, split in [(train, "train"), (val, "val"), (test, "test")]:
        for task in rows:
            task.split = split
    return train, val, test


BUILDERS = {
    "form": form_task,
    "table": table_task,
    "search": search_task,
    "modal": modal_task,
    "select": select_task,
    "workflow": workflow_task,
}


if __name__ == "__main__":
    main()
