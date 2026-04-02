# mas-annotation

## Context
This is a spin-off annotation repo from mas_error_analysis.
Full design doc: see docs/design.md

## Key decisions
- annotator_id is set in config.yaml (not in Python code)
- save_dir = data/annotations/{annotator_id}/
- compute_iaa.py: Cohen's kappa + percent agreement on root_cause_step
- step_annotations: side-by-side only, not auto-scored (vocab not standardized)
- excluded=true traces are dropped from IAA computation
- Confusion matrix: step indices 1–8, anything >8 goes into "8+" bucket

## gstack

Use the `/browse` skill from gstack for all web browsing. Never use `mcp__claude-in-chrome__*` tools.

Available skills:
- `/office-hours` - Office hours workflow
- `/plan-ceo-review` - CEO review planning
- `/plan-eng-review` - Engineering review planning
- `/plan-design-review` - Design review planning
- `/design-consultation` - Design consultation
- `/design-shotgun` - Design shotgun
- `/design-html` - Design HTML
- `/review` - Code review
- `/ship` - Ship workflow
- `/land-and-deploy` - Land and deploy
- `/canary` - Canary deployment
- `/benchmark` - Benchmarking
- `/browse` - Headless browser for web browsing
- `/connect-chrome` - Connect to Chrome
- `/qa` - QA testing
- `/qa-only` - QA only
- `/design-review` - Design review
- `/setup-browser-cookies` - Set up browser cookies
- `/setup-deploy` - Set up deployment
- `/retro` - Retrospective
- `/investigate` - Investigation workflow
- `/document-release` - Document release
- `/codex` - Codex workflow
- `/cso` - CSO workflow
- `/autoplan` - Automatic planning
- `/careful` - Careful mode
- `/freeze` - Freeze workflow
- `/guard` - Guard workflow
- `/unfreeze` - Unfreeze workflow
- `/gstack-upgrade` - Upgrade gstack
- `/learn` - Manage project learnings

If gstack skills aren't working, run `cd .claude/skills/gstack && ./setup` to build the binary and register skills.
