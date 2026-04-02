# mas-annotation

A local web tool for multi-task root-cause annotation.

The UI is named **Root Cause Annotator**. It supports switching tasks under `data/traces/` and stores annotation results in `saved / excluded` folders per task.

## 1. What this tool does

1. Auto-discovers tasks from directory names under `data/traces/{task}`.
2. Shows trace list for the selected task (with search).
3. Cross-highlights between `Trace Spans` and `Trace Summary`.
4. Lets you mark root-cause span and write reasoning/notes.
5. Splits outputs by action:
   - Save -> `data/traces/{task}/saved/{trace_id}.json`
   - Exclude -> `data/traces/{task}/excluded/{trace_id}.json`
6. Auto-saves after about 1.5 seconds of inactivity.

## 2. Setup

1. Clone the repo:
```bash
git clone <repo-url>
cd mas-annotation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your annotator ID (required):

Edit `config.yaml`:
```yaml
annotator_id: your_name
```

## 3. Directory conventions (important)

### 3.1 Input traces

Use one folder per task. Put trace json files directly in each task folder:

```text
data/
  traces/
    GAIA/
      0035f455....json
      041b7f9c....json
    TaskB/
      xxx.json
```

Notes:
1. Only `*.json` files directly under `data/traces/{task}` are treated as traces.
2. Json files in nested folders (for example `.ipynb_checkpoints`) are ignored.

### 3.2 Output annotations

Annotations are written under the same task folder:

```text
data/traces/{task}/saved/{trace_id}.json
data/traces/{task}/excluded/{trace_id}.json
```

Behavior:
1. Clicking Save writes to `saved/` and removes the same trace file from `excluded/` if present.
2. Clicking Exclude writes to `excluded/` and removes the same trace file from `saved/` if present.

## 4. config.yaml fields

Current effective fields:

1. `annotator_id`: required.
2. `summary_file`: optional trace summary json path.
3. `trace_root_dir`: optional task root (default: `data/traces`).
4. `default_task`: optional default task name (must exist under `trace_root_dir`).
5. `annotation_dir`: optional GT annotation directory for read-only GT display.

Example:
```yaml
annotator_id: zhihuat
summary_file: data/trace_summary/claude-haiku-4-5-20251001/trace_summaries.json
trace_root_dir: data/traces
default_task: GAIA
annotation_dir: data/gt_annotations
```

Notes:
1. If `summary_file` is missing, the server still starts; summary content will be empty.
2. Legacy `trace_dir` is not used for task switching anymore.

## 5. Run

Start:
```bash
python demo/progress_annotator.py
```

Custom port:
```bash
python demo/progress_annotator.py --port 6060
```

Open:
```text
http://localhost:6060
```

## 6. Annotation workflow

1. Choose a task from the top `Task` dropdown (for example `GAIA`).
2. Click a trace from the left `Traces` list.
3. Read `Task Description` and `Trace Summary` for context.
4. Inspect `Trace Spans` and mark root-cause span when needed.
5. Fill in `Reasoning` in `Root Cause Annotation`.
6. Click `Save` or `Exclude`.
7. Move to the next trace and repeat.

## 7. Keyboard shortcuts

1. `j` / `→`: next trace
2. `k` / `←`: previous trace
3. `s`: save
4. `e`: exclude
5. `Ctrl+S` / `Cmd+S`: save

## 8. Output schema

Example json in `saved/` or `excluded/`:

```json
{
  "trace_id": "0035f455b3ff2295167a844f04d85d34",
  "task": "GAIA",
  "root_cause_step": null,
  "root_cause_span_id": "abc123",
  "root_cause_reasoning": "Tool call failed and downstream steps used invalid output.",
  "step_annotations": [],
  "notes": "",
  "excluded": false
}
```

Notes:
1. `excluded` is usually `false` in `saved/` and `true` in `excluded/`.
2. `finalized_plan` is no longer saved.

## 9. Multi-task behavior

1. Task list is built from first-level subdirectories under `data/traces/`.
2. If there are unsaved changes, switching task asks for confirmation.
3. Progress stats (`done / total`) are calculated per selected task.

## 10. Compatibility with old data

The backend reads these paths in priority order:

1. `data/traces/{task}/saved/{trace_id}.json`
2. `data/traces/{task}/excluded/{trace_id}.json`
3. `data/annotations/{annotator_id}/{task}/{trace_id}.json` (legacy)
4. `data/annotations/{annotator_id}/{trace_id}.json` (legacy)

Recommendation: use the new `data/traces/{task}/saved|excluded` layout going forward.

## 11. FAQ

### Q1: I see "No tasks found"
Make sure `data/traces/{task}` folders exist and task folder names do not start with `.`.

### Q2: No traces shown for a task
Ensure trace files are `*.json` and located directly under `data/traces/{task}`.

### Q3: Where did Save output go?
Check `data/traces/{task}/saved/{trace_id}.json`.

### Q4: Where did Exclude output go?
Check `data/traces/{task}/excluded/{trace_id}.json`.

### Q5: Why is summary empty?
Usually `summary_file` is missing, invalid, or does not include this `trace_id`.
