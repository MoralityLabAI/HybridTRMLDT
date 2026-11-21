"""Run the trusted Python verifier with repository-scoped Python write access."""
from __future__ import annotations

import os
import runpy
import sys
import sysconfig
from pathlib import Path


def install(root: Path, trusted_scripts: set[Path]) -> None:
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
    runtime = {Path(sysconfig.get_path(name)).resolve()
               for name in ("stdlib", "platstdlib", "purelib", "platlib")}

    def check(path) -> None:
        if not isinstance(path, (str, bytes, os.PathLike)):
            raise PermissionError("verifier requires path-based writes")
        candidate = Path(os.fsdecode(path)).resolve()
        if not candidate.is_relative_to(root) or ".git" in candidate.relative_to(root).parts:
            raise PermissionError(f"verifier write outside workspace source: {candidate}")

    def audit(event, args) -> None:
        if event == "open":
            if args[2] & write_flags:
                check(args[0])
            elif isinstance(args[0], (str, bytes, os.PathLike)):
                path = Path(os.fsdecode(args[0])).resolve()
                if path not in trusted_scripts and not any(path.is_relative_to(folder) for folder in runtime):
                    check(path)
        elif event in {"os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.chown", "os.utime", "os.truncate"}:
            check(args[0])
        elif event in {"os.rename", "os.replace"}:
            check(args[0])
            check(args[1])
            if any(fd not in (-1, None) for fd in args[2:]):
                raise PermissionError("relative descriptor operations are disabled in the verifier")
        elif event in {"os.link", "os.symlink", "os.chdir", "os.fchdir", "os.system", "os.exec",
                       "os.posix_spawn", "os.startfile", "subprocess.Popen", "ctypes.dlopen",
                       "socket.__new__", "winreg.CreateKey", "winreg.SetValue", "winreg.DeleteKey"}:
            raise PermissionError(f"verifier operation is disabled: {event}")

    sys.addaudithook(audit)


def main() -> None:
    # This guard runs as the entry script. The manifest and verifier entrypoint
    # are supplied by the experiment operator and never accepted from the solver.
    root = Path.cwd().resolve()
    if sys.argv[1] == "--preload-click":
        # Click initializes its trusted Windows console adapter via ctypes.
        if os.name == "nt":
            import click._winconsole  # noqa: F401
            import colorama  # noqa: F401
        else:
            import click  # noqa: F401
        import werkzeug  # noqa: F401
        sys.argv.pop(1)
    trusted = set()
    while sys.argv[1] == "--allow-read":
        trusted.add(Path(sys.argv[2]).resolve(strict=True))
        del sys.argv[1:3]
    if sys.argv[1] == "--script":
        trusted.add(Path(sys.argv[2]).resolve(strict=True))
    sys.path.insert(0, str(root))
    install(root, trusted)
    if sys.argv[1] == "--script":
        sys.argv = sys.argv[2:]
        runpy.run_path(sys.argv[0], run_name="__main__")
    elif sys.argv[1] == "-m":
        module = sys.argv[2]
        sys.argv = [module, *sys.argv[3:]]
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    else:
        raise ValueError("expected a trusted --script or -m verifier")


if __name__ == "__main__":
    main()
