#!/usr/bin/env python3
r"""Parallel multi-agent document runner with human-in-the-loop approval gate.

Executes 4 specialized sub-agents in parallel on a shared document:
  - extractor: Structured JSON key-facts & entity extraction
  - table_builder: Converts content into clean GFM Markdown tables
  - proofreader: Grammar, clarity, and executive span refinement
  - artifact_packager: Synthesizes combined outputs into candidate Markdown

Human-in-the-Loop Gate: proposed writes are held in memory until explicit
approve_job(job_id) or reject_job(job_id).
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Enums & Data Models
# ---------------------------------------------------------------------------


class AgentStatus(str, Enum):
    """Per-agent execution status."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobStatus(str, Enum):
    """Overall job lifecycle status."""

    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class AgentTaskState:
    """Tracks the state of a single sub-agent within a job."""

    agent_name: str
    status: AgentStatus = AgentStatus.QUEUED
    output: Any = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


@dataclass
class WorkJob:
    """Represents one multi-agent work job."""

    job_id: str
    title: str
    content: str
    instruction: str
    agents_subset: list[str] = field(
        default_factory=lambda: [
            "extractor",
            "table_builder",
            "proofreader",
            "artifact_packager",
        ]
    )
    status: JobStatus = JobStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    agent_states: dict[str, AgentTaskState] = field(default_factory=dict)
    # Proposed outputs - NOT written to disk until approved
    proposed_markdown: str | None = None
    proposed_diff: str | None = None
    agent_outputs: dict[str, Any] = field(default_factory=dict)
    approved_at: str | None = None
    rejected_at: str | None = None


# ---------------------------------------------------------------------------
# Default sub-agent implementations (deterministic mocks for testing)
# ---------------------------------------------------------------------------


def _default_extractor(content: str, instruction: str) -> dict[str, Any]:
    """Extract structured key-facts from content. Uses simple keyword heuristics."""
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    return {
        "agent": "extractor",
        "facts_extracted": len(lines),
        "key_lines": lines[:10],
        "summary": f"Extracted {len(lines)} lines of content.",
        "instruction_applied": instruction,
    }


def _default_table_builder(content: str, instruction: str) -> str:
    """Convert unstructured content into a GFM Markdown table."""
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    headers = ["#", "Content"]
    rows = []
    for i, line in enumerate(lines[:20], start=1):
        suffix = "..." if len(line) > 80 else ""
        rows.append(f"| {i} | {line[:80]}{suffix} |")
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ] + rows
    return "\n".join(table_lines)


def _default_proofreader(content: str, instruction: str) -> str:
    """Grammar and clarity refinement (deterministic mock)."""
    sentences = content.split(". ")
    refined = ". ".join(s.strip().capitalize() if s.strip() else s for s in sentences)
    return (
        "## Proofread Output\n\n"
        + refined
        + "\n\n---\n*Refinement instruction: *"
        + str(instruction)
    )


def _default_artifact_packager(
    extractor_output: dict[str, Any],
    table_output: str,
    proofreader_output: str,
    content: str,
    instruction: str,
) -> tuple[str, str]:
    """Synthesize combined outputs into a candidate Markdown artifact.

    Returns (proposed_markdown, proposed_diff).
    """
    md_parts = [
        "# " + instruction + "\n",
        "## Extracted Facts\n",
        json.dumps(extractor_output, indent=2),
        "\n\n## Structured Table\n\n",
        table_output,
        "\n\n",
        proofreader_output,
    ]
    proposed_markdown = "\n".join(md_parts)

    # Simple diff: show original content vs new artifact summary
    num_lines = len(content.splitlines())
    facts_count = extractor_output.get("facts_extracted", 0)
    proposed_diff = "--- Original (" + str(num_lines) + " lines)\n"
    proposed_diff += "+++ Proposed Artifact\n"
    proposed_diff += "@@ Combined output with " + str(facts_count) + " facts @@\n"
    proposed_diff += "+ New artifact synthesized from multi-agent parallel execution.\n"
    return proposed_markdown, proposed_diff


# ---------------------------------------------------------------------------
# MultiAgentWorkRunner - Core engine
# ---------------------------------------------------------------------------

_DEFAULT_AGENT_MAP: dict[str, Callable[..., Any]] = {
    "extractor": _default_extractor,
    "table_builder": _default_table_builder,
    "proofreader": _default_proofreader,
    "artifact_packager": _default_artifact_packager,
}


class MultiAgentWorkRunner:
    """Parallel multi-agent document runner with human-in-the-loop approval gate.

    Args:
        inference_fn: Optional callable used by agents for live inference.
                      If None, deterministic defaults are used.
        agent_map: Custom mapping of agent names to callables.
    """

    def __init__(
        self,
        inference_fn: Callable[..., str] | None = None,
        agent_map: dict[str, Callable[..., Any]] | None = None,
    ) -> None:
        self.inference_fn = inference_fn
        self.agent_map = {**_DEFAULT_AGENT_MAP, **(agent_map or {})}
        self._jobs: dict[str, WorkJob] = {}

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------

    def submit_job(
        self,
        title: str,
        content: str,
        instruction: str,
        agents_subset: list[str] | None = None,
    ) -> WorkJob:
        """Submit a new multi-agent work job. Returns the created WorkJob."""
        job_id = "job-" + uuid.uuid4().hex[:12]
        agents = agents_subset or [
            "extractor",
            "table_builder",
            "proofreader",
            "artifact_packager",
        ]

        job = WorkJob(
            job_id=job_id,
            title=title,
            content=content,
            instruction=instruction,
            agents_subset=agents,
        )
        # Initialize agent states
        for agent_name in agents:
            job.agent_states[agent_name] = AgentTaskState(agent_name=agent_name)

        self._jobs[job_id] = job
        return job

    def run_job(self, job_id: str) -> WorkJob:
        """Execute all sub-agents in parallel for the given job."""
        import concurrent.futures

        job = self._jobs.get(job_id)
        if job is None:
            raise ValueError("Unknown job: " + job_id)

        job.status = JobStatus.RUNNING

        def _run_agent(name: str, state: AgentTaskState) -> tuple[str, Any]:
            """Run a single agent and update its state."""
            state.status = AgentStatus.RUNNING
            state.started_at = datetime.now(UTC).isoformat()
            result = None
            try:
                result = self._execute_single_agent(name, job)
                state.output = result
                state.status = AgentStatus.DONE
            except Exception as exc:  # noqa: BLE001
                state.error = str(exc)
            finally:
                state.finished_at = datetime.now(UTC).isoformat()
            return name, result

        # Parallel execution via ThreadPoolExecutor (exclude artifact_packager)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(job.agents_subset)
        ) as pool:
            futures = {
                pool.submit(_run_agent, name, job.agent_states[name]): name
                for name in job.agents_subset
                if name in self.agent_map and name != "artifact_packager"
            }
            for future in concurrent.futures.as_completed(futures):
                agent_name, result = future.result()
                # Store output; artifact_packager needs all others outputs
                job.agent_outputs[agent_name] = result

        # Run artifact_packager last (it depends on other agents)
        packager_state = job.agent_states.get("artifact_packager")
        if packager_state and packager_state.status != AgentStatus.FAILED:
            extractor_out = job.agent_outputs.get("extractor", {})
            table_out = job.agent_outputs.get("table_builder", "")
            proofreader_out = job.agent_outputs.get("proofreader", "")

            packager_state.status = AgentStatus.RUNNING
            packager_state.started_at = datetime.now(UTC).isoformat()
            try:
                md, diff = self.agent_map["artifact_packager"](
                    extractor_out,
                    table_out,
                    proofreader_out,
                    job.content,
                    job.instruction,
                )
                packager_state.output = {"markdown": md, "diff": diff}
                job.proposed_markdown = md
                job.proposed_diff = diff
            except Exception as exc:  # noqa: BLE001
                packager_state.error = str(exc)
                packager_state.status = AgentStatus.FAILED
            finally:
                packager_state.finished_at = datetime.now(UTC).isoformat()

        # Transition to awaiting approval
        job.status = JobStatus.AWAITING_APPROVAL
        return job

    def _execute_single_agent(self, agent_name: str, job: WorkJob) -> Any:
        """Execute a single sub-agent with optional inference_fn."""
        fn = self.agent_map.get(agent_name)
        if fn is None:
            raise ValueError("Unknown agent: " + agent_name)

        # For agents that need content/instruction directly
        if (
            agent_name == "extractor"
            or agent_name == "table_builder"
            or agent_name == "proofreader"
        ):
            return fn(job.content, job.instruction)
        # artifact_packager is handled separately in run_job
        raise ValueError("Agent " + agent_name + " should not be called directly")

    def get_job(self, job_id: str) -> WorkJob | None:
        """Retrieve a job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        """List recent work jobs with summary info."""
        result = []
        for j in reversed(list(self._jobs.values())):
            agent_summary = {
                name: {
                    "status": st.status.value,
                    "output_keys": (
                        list(st.output.keys()) if isinstance(st.output, dict) else None
                    ),
                }
                for name, st in j.agent_states.items()
            }
            result.append(
                {
                    "job_id": j.job_id,
                    "title": j.title,
                    "status": j.status.value,
                    "created_at": j.created_at,
                    "agents_summary": agent_summary,
                }
            )
        return result

    def approve_job(self, job_id: str) -> dict[str, Any]:
        """Approve a job - persists the proposed artifact to ArtifactStore.

        Only writes to disk when explicitly approved by human operator.
        Returns success status and artifact details.
        """
        from pathlib import Path

        job = self._jobs.get(job_id)
        if job is None:
            raise ValueError("Unknown job: " + job_id)
        if job.status != JobStatus.AWAITING_APPROVAL:
            status_val = job.status.value
            raise ValueError(
                "Cannot approve job in status '" + status_val + "'. "
                "Job must be in 'awaiting_approval' state."
            )

        # Persist to ArtifactStore
        from artifact_store import DEFAULT_BASE_DIR, ArtifactStore

        base_dir = os.environ.get(
            "COPPER_WORKSPACE_DIR",
            str(Path(__file__).resolve().parent.parent / DEFAULT_BASE_DIR),
        )
        store = ArtifactStore(base_dir)
        saved = store.save_note(
            title=job.title,
            content=job.proposed_markdown or "",
            artifact_type="note",
            slug_override="approved-" + job_id,
        )

        job.status = JobStatus.APPROVED
        job.approved_at = datetime.now(UTC).isoformat()

        return {
            "success": True,
            "job_id": job_id,
            "artifact_slug": saved.get("slug"),
            "title": saved.get("title"),
        }

    def reject_job(self, job_id: str) -> dict[str, Any]:
        """Reject a job - discards proposed writes without touching disk.

        The job is marked 'rejected' and all proposed outputs are cleared.
        """
        job = self._jobs.get(job_id)
        if job is None:
            raise ValueError("Unknown job: " + job_id)
        if job.status != JobStatus.AWAITING_APPROVAL:
            status_val = job.status.value
            raise ValueError(
                "Cannot reject job in status '" + status_val + "'. "
                "Job must be in 'awaiting_approval' state."
            )

        # Discard proposed writes (do NOT persist to disk)
        job.proposed_markdown = None
        job.proposed_diff = None
        job.agent_outputs.clear()
        job.status = JobStatus.REJECTED
        job.rejected_at = datetime.now(UTC).isoformat()

        return {
            "success": True,
            "job_id": job_id,
            "message": "Job rejected - proposed writes discarded.",
        }

    def get_job_progress(self, job_id: str) -> dict[str, Any]:
        """Get detailed progress graph for a job."""
        job = self._jobs.get(job_id)
        if job is None:
            raise ValueError("Unknown job: " + job_id)

        agents_graph = {}
        for name, state in job.agent_states.items():
            agents_graph[name] = {
                "status": state.status.value,
                "has_output": state.output is not None,
                "error": state.error,
            }

        return {
            "job_id": job.job_id,
            "title": job.title,
            "status": job.status.value,
            "agents": agents_graph,
            "proposed_markdown": job.proposed_markdown,
            "proposed_diff": job.proposed_diff,
        }
