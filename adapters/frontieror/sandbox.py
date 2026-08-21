"""Build the bubblewrap filesystem boundary for one COOPA task."""

from __future__ import annotations

from pathlib import Path, PurePosixPath


def _parents(paths: list[Path]) -> list[str]:
    result = {"/home", "/home/bhz"}
    for path in paths:
        current = PurePosixPath(path.as_posix()).parent
        while str(current).startswith("/home/bhz/"):
            result.add(str(current))
            current = current.parent
    return sorted(result, key=lambda item: (item.count("/"), item))


def build_command(
    command: list[str],
    *,
    repo: Path,
    python_env: Path,
    workspace: Path,
    index: Path,
    instance: Path,
    problem: Path,
) -> list[str]:
    paths = [path.resolve() for path in (repo, python_env, workspace, index, instance, problem)]
    if any(not path.exists() for path in paths):
        missing = [str(path) for path in paths if not path.exists()]
        raise ValueError(f"sandbox input missing: {missing}")
    repo, python_env, workspace, index, instance, problem = paths
    argv = [
        "bwrap", "--die-with-parent", "--new-session", "--unshare-user",
        "--unshare-pid", "--unshare-ipc", "--unshare-uts",
    ]
    for directory in _parents(paths):
        argv.extend(["--dir", directory])
    argv.extend(["--dir", "/etc"])
    for source in ("/usr", "/bin", "/lib", "/lib64"):
        if Path(source).exists():
            argv.extend(["--ro-bind", source, source])
    for source in ("/etc/hosts", "/etc/resolv.conf", "/etc/nsswitch.conf", "/etc/ssl", "/etc/ca-certificates"):
        if Path(source).exists():
            argv.extend(["--ro-bind", source, source])
    argv.extend([
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--ro-bind", str(repo), str(repo),
        "--ro-bind", str(python_env), str(python_env),
        "--bind", str(workspace), str(workspace),
        "--ro-bind", str(index), str(index),
        "--ro-bind", str(instance), str(instance),
        "--ro-bind", str(problem), str(problem),
    ])
    git_dir = repo / ".git"
    if git_dir.exists():
        argv.extend(["--tmpfs", str(git_dir)])
    for relative in (".env", "config.json"):
        target = repo / relative
        if target.exists():
            argv.extend(["--ro-bind", "/dev/null", str(target)])
    argv.extend([
        "--setenv", "HOME", str(workspace),
        "--setenv", "TMPDIR", "/tmp",
        "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
        "--setenv", "ADAPTER_WORKSPACE_PRESTAGED", "1",
        "--setenv", "PATH", f"{python_env / 'bin'}:/usr/bin:/bin",
        "--chdir", str(repo), "--", *command,
    ])
    return argv
