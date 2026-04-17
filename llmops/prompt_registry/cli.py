"""Click CLI for managing versioned prompt templates.

Commands:
  llmops prompt list              — table of all versions + status
  llmops prompt diff <v1> <v2>    — unified diff of system + user_template
  llmops prompt activate <ver>    — set active version, append to history
  llmops prompt rollback          — revert active to previous version
"""

from __future__ import annotations

import difflib
import json
from datetime import datetime, timezone
from pathlib import Path

import click
import yaml

_PROMPTS_DIR = Path("llm_generator/prompts")
_REGISTRY_DIR = Path("llmops/prompt_registry")
_ACTIVE_FILE = _REGISTRY_DIR / "active.json"
_HISTORY_FILE = _REGISTRY_DIR / "history.json"


def _prompt_path(version: str) -> Path:
    return _PROMPTS_DIR / f"system_{version}.yaml"


def _load_active() -> str:
    return json.loads(_ACTIVE_FILE.read_text())["version"]


def _load_history() -> list[dict[str, str]]:
    return json.loads(_HISTORY_FILE.read_text())


def _save_active(version: str) -> None:
    _ACTIVE_FILE.write_text(json.dumps({"version": version}, indent=2))


def _save_history(history: list[dict[str, str]]) -> None:
    _HISTORY_FILE.write_text(json.dumps(history, indent=2))


def _discover_versions() -> list[str]:
    """Return sorted list of version strings found in the prompts directory."""
    return sorted(
        p.stem.removeprefix("system_")
        for p in _PROMPTS_DIR.glob("system_v*.yaml")
    )


@click.group()
def prompt() -> None:
    """Manage versioned prompt templates."""


@prompt.command("list")
def list_versions() -> None:
    """Show all prompt versions with creation date and status."""
    versions = _discover_versions()
    if not versions:
        click.echo("No prompt versions found in llm_generator/prompts/")
        return

    active = _load_active()
    history = {h["version"]: h for h in _load_history()}

    header = f"{'Version':<10} {'Activated':<24} {'Status':<10}"
    click.echo(header)
    click.echo("-" * len(header))
    for ver in versions:
        activated = history.get(ver, {}).get("activated_at", "—")
        status = "active" if ver == active else "archived"
        marker = " ◀" if ver == active else ""
        click.echo(f"{ver:<10} {activated:<24} {status:<10}{marker}")


@prompt.command("diff")
@click.argument("version_a")
@click.argument("version_b")
def diff(version_a: str, version_b: str) -> None:
    """Show unified diff between two prompt versions."""
    path_a = _prompt_path(version_a)
    path_b = _prompt_path(version_b)

    for path, ver in [(path_a, version_a), (path_b, version_b)]:
        if not path.exists():
            raise click.ClickException(f"Prompt not found: {path}")

    data_a: dict[str, str] = yaml.safe_load(path_a.read_text())
    data_b: dict[str, str] = yaml.safe_load(path_b.read_text())

    any_diff = False
    for field in ("system", "user_template"):
        text_a = str(data_a.get(field, "")).splitlines(keepends=True)
        text_b = str(data_b.get(field, "")).splitlines(keepends=True)
        lines = list(
            difflib.unified_diff(
                text_a,
                text_b,
                fromfile=f"{version_a}.yaml ({field})",
                tofile=f"{version_b}.yaml ({field})",
            )
        )
        if lines:
            any_diff = True
            click.echo(f"\n── {field} ──")
            for line in lines:
                if line.startswith("+") and not line.startswith("+++"):
                    click.echo(click.style(line.rstrip(), fg="green"))
                elif line.startswith("-") and not line.startswith("---"):
                    click.echo(click.style(line.rstrip(), fg="red"))
                else:
                    click.echo(line.rstrip())

    if not any_diff:
        click.echo(f"No differences between {version_a} and {version_b}.")


@prompt.command("activate")
@click.argument("version")
def activate(version: str) -> None:
    """Set a prompt version as active and record it in history."""
    path = _prompt_path(version)
    if not path.exists():
        raise click.ClickException(f"Prompt not found: {path}")

    current = _load_active()
    if current == version:
        click.echo(f"{version} is already the active version.")
        return

    history = _load_history()
    history.append(
        {
            "version": version,
            "activated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "previous": current,
        }
    )
    _save_active(version)
    _save_history(history)
    click.echo(f"Activated {version} (was {current}).")


@prompt.command("rollback")
def rollback() -> None:
    """Revert active prompt to the previous version."""
    history = _load_history()
    if len(history) < 2:
        raise click.ClickException("No previous version to roll back to.")

    current = _load_active()
    previous = history[-1].get("previous")

    if previous is None or not _prompt_path(previous).exists():
        raise click.ClickException(f"Cannot resolve previous version from history.")

    history.append(
        {
            "version": previous,
            "activated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "previous": current,
            "note": "rollback",
        }
    )
    _save_active(previous)
    _save_history(history)
    click.echo(f"Rolled back from {current} to {previous}.")


@click.group()
def cli() -> None:
    """LoyaltyLens LLMOps CLI."""


cli.add_command(prompt)

if __name__ == "__main__":
    cli()
