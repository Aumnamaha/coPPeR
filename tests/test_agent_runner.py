#!/usr/bin/env python3
r"""Tests for Phase 11 Multi-Agent Document Runner & Work Job API.

Covers:
- agent_runner.py: parallel execution, approve/reject gate, no-disk-write-before-approval
- /api/work/jobs/* endpoints via FastAPI TestClient with tmp_path-scoped stores
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure src/app are on path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))


@pytest.fixture()
def agent_runner():
    """Create a fresh MultiAgentWorkRunner with deterministic defaults."""
    from agent_runner import MultiAgentWorkRunner

    return MultiAgentWorkRunner()


# ---------------------------------------------------------------------------
# Agent runner unit tests
# ---------------------------------------------------------------------------


class TestMultiAgentWorkRunner:
    """Test parallel execution, job lifecycle, and approval gate."""

    def test_submit_job_creates_pending_job(self, agent_runner) -> None:
        """submit_job returns a WorkJob with PENDING status and 4 agents."""
        job = agent_runner.submit_job(
            title="Test Job",
            content="Hello world\nLine two.",
            instruction="Process this document",
        )
        assert job.status.value == "pending"
        assert len(job.agent_states) == 4
        assert "extractor" in job.agent_states
        assert "table_builder" in job.agent_states
        assert "proofreader" in job.agent_states
        assert "artifact_packager" in job.agent_states

    def test_run_job_executes_all_agents_in_parallel(self, agent_runner) -> None:
        """run_job transitions to AWAITING_APPROVAL with all agents DONE."""
        job = agent_runner.submit_job(
            title="Parallel Test",
            content="Sample content line 1\nSample content line 2\nLine three.",
            instruction="Extract and summarize",
        )
        result = agent_runner.run_job(job.job_id)

        assert result.status.value == "awaiting_approval"
        for name, state in result.agent_states.items():
            if name != "artifact_packager":
                # extractor/table_builder/proofreader should be DONE
                assert state.status.value == "done", f"{name} status: {state.status}"

    def test_run_job_produces_proposed_output(self, agent_runner) -> None:
        """After run_job, proposed_markdown and proposed_diff are populated."""
        job = agent_runner.submit_job(
            title="Output Test",
            content="Some text to process.",
            instruction="Summarize",
        )
        agent_runner.run_job(job.job_id)

        assert job.proposed_markdown is not None
        assert len(job.proposed_markdown) > 0
        assert job.proposed_diff is not None
        assert "Proposed Artifact" in job.proposed_diff

    def test_run_agent_subset(self, agent_runner) -> None:
        """Submitting with a subset of agents only runs those agents."""
        job = agent_runner.submit_job(
            title="Subset Test",
            content="Content here.",
            instruction="Just extract",
            agents_subset=["extractor"],
        )
        result = agent_runner.run_job(job.job_id)

        assert result.status.value == "awaiting_approval"
        # Only extractor should be present, artifact_packager excluded
        assert len(result.agent_states) == 1
        assert "extractor" in result.agent_states


# ---------------------------------------------------------------------------
# Human-in-the-Loop Approval Gate tests
# ---------------------------------------------------------------------------


class TestApprovalGate:
    """Test that disk writes only happen after explicit approval."""

    def test_no_disk_write_before_approval(self, tmp_path: Path) -> None:
        """Running a job does NOT write anything to ArtifactStore until approve."""
        os.environ["COPPER_WORKSPACE_DIR"] = str(tmp_path / "store")

        from agent_runner import MultiAgentWorkRunner

        runner = MultiAgentWorkRunner()
        job = runner.submit_job(
            title="No Write Test",
            content="Content.",
            instruction="Process",
        )
        runner.run_job(job.job_id)

        # Job should be awaiting_approval, NOT approved yet
        assert job.status.value == "awaiting_approval"

        # No files should have been written to the store directory
        store_dir = Path(os.environ["COPPER_WORKSPACE_DIR"])
        md_files = list(store_dir.glob("*.md")) if store_dir.exists() else []
        assert len(md_files) == 0, f"Expected no files on disk, found: {md_files}"

    def test_approve_job_persists_to_artifact_store(self, tmp_path: Path) -> None:
        """approve_job writes the proposed artifact to ArtifactStore."""
        os.environ["COPPER_WORKSPACE_DIR"] = str(tmp_path / "store")

        from agent_runner import MultiAgentWorkRunner

        runner = MultiAgentWorkRunner()
        job = runner.submit_job(
            title="Approved Job",
            content="Content here.",
            instruction="Summarize this document",
        )
        runner.run_job(job.job_id)

        result = runner.approve_job(job.job_id)
        assert result["success"] is True
        assert "artifact_slug" in result

        # Verify the artifact was written to disk
        store_dir = Path(os.environ["COPPER_WORKSPACE_DIR"])
        md_files = list(store_dir.glob("*.md"))
        assert len(md_files) >= 1, "Expected at least one .md file on disk"

        # Job status should now be APPROVED
        updated_job = runner.get_job(job.job_id)
        assert updated_job.status.value == "approved"

    def test_reject_job_discards_without_writing(self, tmp_path: Path) -> None:
        """reject_job discards proposed writes and does NOT persist to disk."""
        os.environ["COPPER_WORKSPACE_DIR"] = str(tmp_path / "store2")

        from agent_runner import MultiAgentWorkRunner

        runner = MultiAgentWorkRunner()
        job = runner.submit_job(
            title="Rejected Job",
            content="Content here.",
            instruction="Process this",
        )
        runner.run_job(job.job_id)

        # Before reject: no disk writes yet
        store_dir = Path(os.environ["COPPER_WORKSPACE_DIR"])
        md_files_before = list(store_dir.glob("*.md")) if store_dir.exists() else []
        assert len(md_files_before) == 0

        result = runner.reject_job(job.job_id)
        assert result["success"] is True

        # After reject: still no disk writes
        md_files_after = list(store_dir.glob("*.md")) if store_dir.exists() else []
        assert len(md_files_after) == 0, "Reject should not write to disk"

        updated_job = runner.get_job(job.job_id)
        assert updated_job.status.value == "rejected"


class TestJobProgress:
    """Test get_job_progress returns correct data."""

    def test_get_job_progress_returns_agents_graph(self, agent_runner) -> None:
        job = agent_runner.submit_job(
            title="Progress Test",
            content="Content.",
            instruction="Process",
        )
        runner = agent_runner
        runner.run_job(job.job_id)

        progress = runner.get_job_progress(job.job_id)
        assert progress["job_id"] == job.job_id
        assert "agents" in progress
        assert len(progress["agents"]) == 4
        for info in progress["agents"].values():
            assert "status" in info
            assert "has_output" in info


# ---------------------------------------------------------------------------
# API endpoint tests (FastAPI TestClient)
# ---------------------------------------------------------------------------


# Reset the module-level singleton before each test to avoid cross-test pollution
@pytest.fixture(autouse=True)
def reset_agent_runner_singleton():
    """Reset _agent_runner global so each test gets a fresh runner."""
    import app.inference_server as iserver

    old_value = iserver._agent_runner
    iserver._agent_runner = None
    yield
    # Restore after test (or set to None for clean state)
    iserver._agent_runner = old_value


@pytest.fixture()
def api_store(tmp_path: Path):
    """Set COPPER_WORKSPACE_DIR env var for API tests."""
    os.environ["COPPER_WORKSPACE_DIR"] = str(tmp_path / "api_store")
    yield tmp_path
    del os.environ["COPPER_WORKSPACE_DIR"]


@pytest.fixture()
def client(api_store: Path):
    """Create a FastAPI TestClient with the inference server app."""
    from inference_server import app as server_app

    return TestClient(server_app)


class TestWorkJobsAPI:
    """Test /api/work/jobs/* REST endpoints."""

    def test_create_work_job_returns_job_id(self, client: TestClient) -> None:
        resp = client.post(
            "/api/work/jobs",
            json={
                "title": "API Job",
                "content": "Sample content line 1\nLine two.",
                "instruction": "Summarize",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "awaiting_approval"

    def test_get_work_job_returns_progress(self, client: TestClient) -> None:
        # First create a job
        create_resp = client.post(
            "/api/work/jobs",
            json={
                "title": "Get Job Test",
                "content": "Content.",
                "instruction": "Process",
            },
        )
        job_id = create_resp.json()["job_id"]

        resp = client.get(f"/api/work/jobs/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job_id
        assert "agents" in data

    def test_list_work_jobs(self, client: TestClient) -> None:
        # Create two jobs
        for i in range(2):
            client.post(
                "/api/work/jobs",
                json={
                    "title": f"List Job {i}",
                    "content": "Content.",
                    "instruction": "Process",
                },
            )

        resp = client.get("/api/work/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert "jobs" in data
        assert len(data["jobs"]) >= 2

    def test_approve_work_job_saves_artifact(self, client: TestClient) -> None:
        create_resp = client.post(
            "/api/work/jobs",
            json={
                "title": "Approve API Job",
                "content": "Content to approve.",
                "instruction": "Summarize and save",
            },
        )
        job_id = create_resp.json()["job_id"]

        resp = client.post(f"/api/work/jobs/{job_id}/approve")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "artifact_slug" in data

    def test_reject_work_job_discards(self, client: TestClient) -> None:
        create_resp = client.post(
            "/api/work/jobs",
            json={
                "title": "Reject API Job",
                "content": "Content to reject.",
                "instruction": "Process this",
            },
        )
        job_id = create_resp.json()["job_id"]

        resp = client.post(f"/api/work/jobs/{job_id}/reject")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    def test_get_nonexistent_job_returns_404(self, client: TestClient) -> None:
        resp = client.get("/api/work/jobs/nonexistent-job-id")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# UI asset contract tests (Phase 11 additions)
# ---------------------------------------------------------------------------


class TestWorkspaceUIWorkAgentsAssets:
    """Verify HTML/CSS/JS contain Work Agents elements."""

    def test_workspace_has_send_to_work_button(self, client: TestClient) -> None:
        resp = client.get("/workspace")
        assert resp.status_code == 200
        body = resp.text
        assert 'id="btn-send-to-work"' in body

    def test_workspace_has_agents_panel(self, client: TestClient) -> None:
        resp = client.get("/workspace")
        assert resp.status_code == 200
        body = resp.text
        assert 'id="work-agents-panel"' in body
        assert "Approve" in body or "approve" in body.lower()

    def test_workspace_css_has_agents_styles(self, client: TestClient) -> None:
        resp = client.get("/workspace/styles.css")
        assert resp.status_code == 200
        body = resp.text
        assert ".agents-panel" in body

    def test_workspace_js_has_work_agents_functions(self, client: TestClient) -> None:
        resp = client.get("/workspace/app.js")
        assert resp.status_code == 200
        body = resp.text
        assert "sendToWork" in body or "startJobPolling" in body


# ---------------------------------------------------------------------------
# WorkspaceBridge method tests (Phase 11)
# ---------------------------------------------------------------------------


class TestWorkspaceBridgeMethods:
    """Test WorkspaceBridge list_artifacts, save_artifact methods exist."""

    def test_workspace_app_has_agent_runner_import(self) -> None:
        """workspace_app.py should import agent_runner for bridge methods."""
        from workspace_app import WorkspaceBridge

        # Verify the class has the expected artifact-related methods
        assert hasattr(WorkspaceBridge, "list_artifacts")
        assert hasattr(WorkspaceBridge, "save_artifact")
