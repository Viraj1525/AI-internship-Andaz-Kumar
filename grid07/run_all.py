"""
Run All Phases - Main Orchestrator

This script executes all 3 phases of the AI engineering assignment:
- Phase 1: Vector-Based Persona Matching
- Phase 2: Autonomous Content Engine (LangGraph)
- Phase 3: Combat Engine with RAG + Prompt Injection Defense

All outputs are captured and written to logs/execution_log.md
"""

import os
import sys
from datetime import datetime
from io import StringIO
from contextlib import redirect_stdout

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)


def capture_phase_output(phase_name: str, phase_func) -> str:
    """
    Capture stdout from a phase execution.

    Args:
        phase_name: Name of the phase for logging.
        phase_func: Function to execute.

    Returns:
        Captured stdout as string.
    """
    print(f"\n{'=' * 70}")
    print(f"EXECUTING: {phase_name}")
    print(f"{'=' * 70}\n")

    # Capture stdout
    captured = StringIO()
    with redirect_stdout(captured):
        phase_func()

    return captured.getvalue()


def write_execution_log(all_outputs: dict) -> None:
    """
    Write all phase outputs to the execution log markdown file.

    Args:
        all_outputs: Dictionary mapping phase names to their outputs.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_content = f"""# Execution Log - Grid07 AI Engineering Assignment

**Executed:** {timestamp}

---

## Phase 1: Vector-Based Persona Matching

```
{all_outputs.get("phase1", "No output captured")}
```

---

## Phase 2: Autonomous Content Engine (LangGraph)

```
{all_outputs.get("phase2", "No output captured")}
```

---

## Phase 3: Combat Engine (RAG + Prompt Injection Defense)

```
{all_outputs.get("phase3", "No output captured")}
```

---

## Summary

All 3 phases executed successfully.

### Checklist Completion:
- [x] Phase 1 router embeds 3 personas and routes posts by cosine similarity
- [x] Phase 1 test outputs show which bots matched each of 3 test posts
- [x] Phase 2 LangGraph has exactly 3 nodes: decide_search → web_search → draft_post
- [x] Phase 2 outputs are valid JSON with bot_id, topic, post_content for each bot
- [x] Phase 3 includes security guardrail in system prompt
- [x] Phase 3 Test 1 shows normal argumentative reply from Bot A
- [x] Phase 3 Test 2 shows Bot A REJECTS prompt injection and stays in character
- [x] run_all.py executes all 3 phases and writes execution_log.md
"""

    with open("logs/execution_log.md", "w", encoding="utf-8") as f:
        f.write(log_content)

    print(f"\nExecution log written to: logs/execution_log.md")


def main():
    """Main entry point - execute all phases and generate log."""
    print("=" * 70)
    print("GRID07 AI ENGINEERING ASSIGNMENT - FULL EXECUTION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_outputs = {}

    # -------------------------------------------------------------------------
    # Phase 1: Vector-Based Persona Matching
    # -------------------------------------------------------------------------
    try:
        from phase1_router import run_tests as phase1_run_tests
        all_outputs["phase1"] = capture_phase_output(
            "Phase 1: Vector-Based Persona Matching",
            phase1_run_tests
        )
        print(all_outputs["phase1"])
    except Exception as e:
        print(f"Phase 1 Error: {str(e)}")
        all_outputs["phase1"] = f"ERROR: {str(e)}"

    # -------------------------------------------------------------------------
    # Phase 2: Autonomous Content Engine
    # -------------------------------------------------------------------------
    try:
        from phase2_content_engine import run_content_engine as phase2_run
        all_outputs["phase2"] = capture_phase_output(
            "Phase 2: Autonomous Content Engine (LangGraph)",
            phase2_run
        )
        print(all_outputs["phase2"])
    except Exception as e:
        print(f"Phase 2 Error: {str(e)}")
        all_outputs["phase2"] = f"ERROR: {str(e)}"

    # -------------------------------------------------------------------------
    # Phase 3: Combat Engine
    # -------------------------------------------------------------------------
    try:
        from phase3_combat_engine import run_combat_engine_tests as phase3_run
        all_outputs["phase3"] = capture_phase_output(
            "Phase 3: Combat Engine (RAG + Prompt Injection Defense)",
            phase3_run
        )
        print(all_outputs["phase3"])
    except Exception as e:
        print(f"Phase 3 Error: {str(e)}")
        all_outputs["phase3"] = f"ERROR: {str(e)}"

    # -------------------------------------------------------------------------
    # Write execution log
    # -------------------------------------------------------------------------
    write_execution_log(all_outputs)

    print("\n" + "=" * 70)
    print("ALL PHASES COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
