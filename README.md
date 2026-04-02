# mas-annotation

Annotation tool for MAS failure traces. Two annotators work independently, then compute inter-annotator agreement with one command.

## Setup (collaborator)

**1. Clone the repo**
```bash
git clone <repo-url>
cd mas-annotation
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set your annotator ID**

Edit `config.yaml`:
```yaml
annotator_id: your_name   # ← change this to your name
```

**4. Run the tool**
```bash
python demo/progress_annotator.py
```

Open http://localhost:6060 in your browser. Your annotations are saved to `data/annotations/your_name/`.

## Annotation workflow

**Keyboard shortcuts:**
| Key | Action |
|-----|--------|
| `j` / `→` | Next trace |
| `k` / `←` | Previous trace |
| `s` | Save |
| `e` | Exclude (flag as unusable) |
| `1`–`9` | Select root cause step N |
| `Ctrl+S` | Save |

**Per-trace steps:**
1. Read the **Task Description** and **Trace Summary** to understand what the agent was trying to do.
2. Review the **Spans** panel to see the execution tree.
3. Edit the **Finalized Plan** if the auto-extracted plan is wrong.
4. Click **RC** on the plan step where the agent first went wrong (root cause step). Or press `1`–`9`.
5. Optionally click **Mark RC** on the specific span in the spans/summary panel.
6. Write your **Reasoning** in the text box.
7. The tool auto-saves after 1.5s of inactivity. Or press `s`.

**Excluded traces:** If a trace is ambiguous, broken, or otherwise unusable for IAA, press `e` to exclude it. Excluded traces are dropped from agreement computation.

## Sharing annotations

After annotating, commit your annotation files and push (or send a PR):

```bash
git add data/annotations/your_name/
git commit -m "annotations: add your_name batch"
git push
```

Or just zip `data/annotations/your_name/` and share directly — the other annotator can drop it into their local repo.

## Computing agreement

Once both annotators have annotated, run:

```bash
python compute_iaa.py
```

This auto-discovers the two annotator directories and writes `iaa_report.md` with:
- Percent exact agreement on `root_cause_step`
- Cohen's kappa (unweighted)
- Mean absolute step distance
- Confusion matrix (steps 1–8, bucket "8+" for longer plans)
- Per-trace table with disagreement flags
- Full reasoning for high-disagreement traces

To specify annotators explicitly:
```bash
python compute_iaa.py --annotator1 zhihuat --annotator2 collaborator --out report.md
```

## Annotation schema

Each annotation file (`data/annotations/{annotator_id}/{trace_id}.json`):

```json
{
  "trace_id": "abc123",
  "finalized_plan": [
    {"step_number": 1, "description": "Search for articles", "depends_on": []},
    {"step_number": 2, "description": "Extract count", "depends_on": [1]}
  ],
  "root_cause_step": 2,
  "root_cause_span_id": "span_xyz",
  "root_cause_reasoning": "The agent failed at extraction — returned empty result.",
  "step_annotations": [],
  "notes": "",
  "excluded": false
}
```

`root_cause_step` is the 1-indexed step number in `finalized_plan` where the agent first failed. This is the primary field used for IAA computation.

`excluded: true` means the trace is flagged as unusable and will be dropped from IAA.
