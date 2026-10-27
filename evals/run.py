"""Launch the cases selected in evals.toml.

Usage: python -m evals.run --config evals/evals.toml [--dry-run]
"""

import argparse
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("evals.toml")
RUN_FIELDS = {"name", "config", "case"}


class ConfigError(ValueError):
    pass


def load_config(path: Path) -> dict:
    try:
        with path.open("rb") as file:
            return tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"Cannot load {path}: {error}") from error


def required_string(table: dict, key: str, context: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{context}: '{key}' must be a nonempty string")
    return value


def cli_arguments(settings: dict, context: str) -> list[str]:
    if "limit" in settings:
        limit = settings["limit"]
        if type(limit) is not int or limit < 1:
            raise ConfigError(f"{context}: 'limit' must be a positive integer")
    if "tools" in settings:
        tools = settings["tools"]
        if not isinstance(tools, list) or not all(
            isinstance(tool, str) and tool.strip() for tool in tools
        ):
            raise ConfigError(f"{context}: 'tools' must be a list of tool names")

    arguments = []
    for key, value in settings.items():
        if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", key):
            raise ConfigError(f"{context}: invalid argument name {key!r}")
        values = value if isinstance(value, list) else [value]
        if any(type(item) not in (str, int, float, bool) for item in values):
            raise ConfigError(f"{context}: '{key}' must be a scalar or a list of scalars")
        arguments.append("--" + key.replace("_", "-"))
        arguments.extend(
            str(item).lower() if isinstance(item, bool) else str(item)
            for item in values
        )
    return arguments


def build_runs(config_path: Path) -> list[tuple[str, list[str]]]:
    """Validate all selections before starting any runner."""
    config_path = config_path.expanduser().resolve()
    config = load_config(config_path)
    selections = config.get("runs")
    if not isinstance(selections, list) or not selections:
        raise ConfigError(f"{config_path}: define at least one [[runs]] entry")

    commands = []
    names = set()
    for index, selection in enumerate(selections, start=1):
        context = f"{config_path}, run {index}"
        if not isinstance(selection, dict):
            raise ConfigError(f"{context}: expected a table")
        name = required_string(selection, "name", context)
        if name in names:
            raise ConfigError(f"{context}: duplicate run name {name!r}")
        names.add(name)
        family_file = required_string(selection, "config", context)
        family_path = (config_path.parent / Path(family_file).expanduser()).resolve()
        family = load_config(family_path)
        module = required_string(family, "runner_module", str(family_path))
        if not all(part.isidentifier() for part in module.split(".")):
            raise ConfigError(f"{family_path}: invalid runner_module {module!r}")

        case_name = required_string(selection, "case", context)
        cases = family.get("cases", {})
        if not isinstance(cases, dict) or case_name not in cases:
            raise ConfigError(f"{context}: unknown case {case_name!r} in {family_path}")
        defaults = family.get("defaults", {})
        case = cases[case_name]
        if not isinstance(defaults, dict) or not isinstance(case, dict):
            raise ConfigError(f"{family_path}: defaults and cases must be tables")

        # Merge first so an overridden input need not exist. Resolve its path
        # against the file where the winning value was defined.
        settings = {**defaults, **case}
        overrides = {key: value for key, value in selection.items() if key not in RUN_FIELDS}
        settings.update(overrides)
        for key in ("input", "output"):
            if key not in settings:
                continue

            origin = config_path if key in overrides else family_path
            value = required_string(settings, key, str(origin))
            path = (origin.parent / Path(value).expanduser()).resolve()

            # Inputs must exist; outputs may be new files.
            if key == "input" and not path.is_file():
                raise ConfigError(f"{origin}: input file does not exist: {path}")

            settings[key] = str(path)
        command = [sys.executable, "-m", module, *cli_arguments(settings, name)]
        commands.append((name, command))
    return commands


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print resolved commands without executing runners"
    )
    args = parser.parse_args(argv)
    try:
        runs = build_runs(args.config)
    except ConfigError as error:
        parser.error(str(error))

    for name, command in runs:
        print(f"[{name}] {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        try:
            result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        except OSError as error:
            print(f"[{name}] Could not start runner: {error}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            return 130
        if result.returncode:
            print(f"[{name}] Runner exited with status {result.returncode}", file=sys.stderr)
            return result.returncode if result.returncode > 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
