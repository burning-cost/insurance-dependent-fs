"""
Run the insurance-dependent-fs test suite on Databricks via the Jobs API.

Usage:
    python run_tests_databricks.py

Uploads the project to DBFS, runs pytest on a serverless cluster, and streams
the output back.
"""
import os
import sys
import time
import zipfile
import io
import base64
import json

# Load Databricks credentials
env_path = os.path.expanduser("~/.config/burning-cost/databricks.env")
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k] = v

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, compute

w = WorkspaceClient()


# ---------------------------------------------------------------------------
# 1. Zip the project and upload to workspace
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(__file__)
WORKSPACE_PATH = "/Workspace/insurance-dependent-fs"

print("Uploading project files to Databricks workspace...")

def upload_dir(local_dir: str, workspace_dir: str):
    """Recursively upload a directory to the Databricks workspace."""
    for root, dirs, files in os.walk(local_dir):
        # Skip hidden dirs, __pycache__, .git, dist, build
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".") and d not in ("__pycache__", "dist", "build", ".venv")
        ]
        for fname in files:
            if fname.endswith((".pyc", ".egg-info")) or fname.startswith("."):
                continue
            local_path = os.path.join(root, fname)
            rel_path = os.path.relpath(local_path, local_dir)
            ws_path = f"{workspace_dir}/{rel_path.replace(os.sep, '/')}"
            # Read and upload
            with open(local_path, "rb") as fh:
                content = fh.read()
            try:
                w.workspace.mkdirs(path=os.path.dirname(ws_path))
            except Exception:
                pass
            w.workspace.upload(
                path=ws_path,
                content=io.BytesIO(content),
                overwrite=True,
                format="AUTO",
            )
            print(f"  Uploaded: {ws_path}")

upload_dir(PROJECT_ROOT, WORKSPACE_PATH)
print("Upload complete.\n")


# ---------------------------------------------------------------------------
# 2. Create and run a job that installs the package and runs pytest
# ---------------------------------------------------------------------------

NOTEBOOK_CONTENT = '''
import subprocess, sys

# Install the package from workspace path
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e",
     "/Workspace/insurance-dependent-fs", "--quiet"],
    capture_output=True, text=True
)
print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)

# Run pytest
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance-dependent-fs/tests",
     "-v", "--tb=short", "--no-header", "-x"],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-dependent-fs"
)
print(result.stdout)
print(result.stderr)
if result.returncode != 0:
    raise Exception(f"Tests failed with exit code {result.returncode}")
print("\\n=== ALL TESTS PASSED ===")
'''

# Upload the test runner notebook
RUNNER_PATH = "/Workspace/insurance-dependent-fs-test-runner"
w.workspace.upload(
    path=RUNNER_PATH,
    content=io.BytesIO(NOTEBOOK_CONTENT.encode()),
    overwrite=True,
    format="AUTO",
    language="PYTHON",
)
print(f"Test runner uploaded to {RUNNER_PATH}")


# ---------------------------------------------------------------------------
# 3. Submit as a one-time run
# ---------------------------------------------------------------------------

print("\nSubmitting test job...")
run = w.jobs.submit(
    run_name="insurance-dependent-fs-tests",
    tasks=[
        jobs.SubmitTask(
            task_key="run_pytest",
            notebook_task=jobs.NotebookTask(
                notebook_path=RUNNER_PATH,
                source=jobs.Source.WORKSPACE,
            ),
            new_cluster=compute.ClusterSpec(
                spark_version="15.4.x-cpu-ml-scala2.12",
                node_type_id="i3.xlarge",
                num_workers=0,
                spark_conf={"spark.master": "local[*]"},
                custom_tags={"project": "insurance-dependent-fs"},
            ),
        )
    ],
).result()

print(f"Run submitted: run_id={run.run_id}")

# Wait for completion
print("Waiting for tests to complete (this takes 3-5 minutes)...")
while True:
    run_state = w.jobs.get_run(run_id=run.run_id)
    state = run_state.state
    lc = state.life_cycle_state
    print(f"  Status: {lc.value}  Result: {state.result_state.value if state.result_state else '-'}")
    if lc.value in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break
    time.sleep(15)

# Get output
final_state = w.jobs.get_run(run_id=run.run_id)
print(f"\nFinal state: {final_state.state.life_cycle_state.value}")
print(f"Result: {final_state.state.result_state.value if final_state.state.result_state else 'N/A'}")

# Try to get notebook output
try:
    for task in final_state.tasks or []:
        output = w.jobs.get_run_output(run_id=task.run_id)
        if output.notebook_output:
            print("\n=== TEST OUTPUT ===")
            print(output.notebook_output.result)
        if output.error:
            print(f"\n=== ERROR ===\n{output.error}")
except Exception as e:
    print(f"Could not retrieve output: {e}")
    run_url = f"{os.environ['DATABRICKS_HOST']}#job/{final_state.job_id}/run/{final_state.run_id}"
    print(f"View run at: {run_url}")

if final_state.state.result_state and final_state.state.result_state.value == "SUCCESS":
    print("\n=== TESTS PASSED ===")
    sys.exit(0)
else:
    print("\n=== TESTS FAILED ===")
    sys.exit(1)
