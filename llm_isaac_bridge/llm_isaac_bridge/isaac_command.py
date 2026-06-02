"""Build and optionally run the existing JADE Isaac command."""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from llm_isaac_bridge.paths import DEFAULT_ISAAC_RUNNER, ISAACLAB_ROOT

IsaacMode = Literal["hover", "contact"]


@dataclass(frozen=True)
class IsaacCommand:
    """Command-line representation for one Isaac execution."""

    argv: tuple[str, ...]
    cwd: Path
    env: dict[str, str]

    def shell_text(self) -> str:
        """Return a copy-pasteable shell command."""

        prefix = ""
        if self.env.get("TERM"):
            prefix = f"TERM={shlex.quote(self.env['TERM'])} "
        return prefix + " ".join(shlex.quote(item) for item in self.argv)


def build_isaac_command(
    *,
    plan_path: str | Path,
    output_dir: str | Path,
    mode: IsaacMode = "contact",
    isaaclab_root: str | Path = ISAACLAB_ROOT,
    runner_script: str | Path = DEFAULT_ISAAC_RUNNER,
    setup_only: bool = False,
    extra_args: list[str] | tuple[str, ...] = (),
) -> IsaacCommand:
    """Create the command that runs the existing Isaac JADE runner."""

    isaac_root = Path(isaaclab_root).expanduser().resolve()
    isaaclab_sh = isaac_root / "isaaclab.sh"
    if not isaaclab_sh.exists():
        raise FileNotFoundError(f"isaaclab.sh not found: {isaaclab_sh}")

    plan = Path(plan_path).expanduser().resolve()
    if not plan.exists():
        raise FileNotFoundError(f"plan file not found: {plan}")

    runner = Path(runner_script).expanduser().resolve()
    if not runner.exists():
        raise FileNotFoundError(f"JADE Isaac runner not found: {runner}")

    out = Path(output_dir).expanduser().resolve()
    argv = [
        str(isaaclab_sh),
        "-p",
        str(runner),
        "--mode",
        mode,
        "--plan",
        str(plan),
        "--out",
        str(out),
    ]
    if setup_only:
        argv.append("--setup-only")
    argv.extend(str(item) for item in extra_args)

    env = os.environ.copy()
    env["TERM"] = "xterm"
    return IsaacCommand(argv=tuple(argv), cwd=isaac_root, env=env)


def run_isaac_command(command: IsaacCommand) -> subprocess.CompletedProcess[str]:
    """Run an Isaac command and stream through stdout/stderr capture."""

    return subprocess.run(
        list(command.argv),
        cwd=str(command.cwd),
        env=command.env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
