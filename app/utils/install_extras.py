"""
ziya-install-extras — installer for Ziya's optional extras (Community Edition).

Exposed as the ``ziya-install-extras`` console command (pyproject.toml
[tool.poetry.scripts]). Install users have only the installed wheel — no
source checkout — so this is a packaged entry point that drives the packaged
installer script at app/scripts/install_extras.sh.

The script installs, opt-in and never silently, the two things ``pip`` cannot:

  --browser   the Chromium build Playwright drives (``playwright install
              chromium``) — lets the model look at its own rendered diagrams,
              and enables PDF export and frozen diagram artifacts.
  --latex     a TeX distribution plus the TeX Live packages the LaTeX
              renderer needs (circuitikz, chemfig, pgfplots, TikZ, ...).
  --all       both; the default when no target is given.

Ziya runs without either; you just lose those features. The Python-level
dependencies (playwright, scapy) are ordinary hard requirements of the wheel
and need no extra step.

Editions that add their own extras wrap this: they run their additions and
then call ``run(args)`` here, so the script location and invocation live in
exactly one place.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

SCRIPT_RELPATH = Path("scripts") / "install_extras.sh"


def latex_script_path() -> Optional[Path]:
    """Locate app/scripts/install_extras.sh inside the installed package.

    Resolved via the ``app`` package's location rather than importlib.resources
    because app/scripts ships as data, not as an importable subpackage.
    """
    import app  # local import: this module may be imported before app is on sys.path
    candidate = Path(app.__file__).resolve().parent / SCRIPT_RELPATH
    return candidate if candidate.is_file() else None


def tex_packages(script: Path) -> List[str]:
    """The TeX Live package list, read from the script's TEX_PACKAGES=(...) line."""
    for line in script.read_text(encoding="utf-8").splitlines():
        if line.startswith("TEX_PACKAGES=("):
            return line[len("TEX_PACKAGES=("):line.index(")")].split()
    return []


WINDOWS_USAGE = """usage: ziya-install-extras [--browser | --latex | --all] [--dry-run] [--yes]

  --browser  the Chromium build Playwright drives (~150 MB)
  --latex    the TeX Live packages the LaTeX renderer needs, installed with
             TeX Live's tlmgr; without a TeX distribution, how to get one
  --all      both (the default)
  --dry-run  print the plan and exit; install nothing
  --yes      skip the confirmation prompt"""


def run_windows(args: List[str], script: Optional[Path]) -> int:
    """The installer for Windows, where the bash script cannot run.

    ``bash`` there is WSL's launcher, which would install into the Linux VM,
    or Git Bash, which has no package manager.  Chromium installs natively.
    For TeX, TeX Live's tlmgr is used if one is installed; otherwise this
    says how to get a distribution, as the script does for Homebrew.
    """
    from app.utils.process_utils import resolve_executable

    if "-h" in args or "--help" in args:
        print(WINDOWS_USAGE)
        return 0
    targets = {a for a in args if a in ("--browser", "--latex", "--all")}
    browser = not targets or bool(targets & {"--all", "--browser"})
    latex = not targets or bool(targets & {"--all", "--latex"})
    steps = []
    print("Ziya optional extras")
    if browser:
        cmd = [sys.executable, "-m", "playwright", "install", "chromium"]
        print("  * Chromium for Playwright (~150 MB):  " + subprocess.list2cmdline(cmd))
        steps.append(("Chromium", cmd))
    missing = 0
    if latex:
        tlmgr = shutil.which("tlmgr")
        packages = tex_packages(script) if script else []
        if tlmgr and packages:
            cmd = [resolve_executable("tlmgr"), "install", *packages]
            print("  * TeX Live packages:  " + subprocess.list2cmdline(["tlmgr", "install", *packages]))
            steps.append(("TeX Live packages", cmd))
        elif shutil.which("latex") and shutil.which("dvisvgm"):
            print("  ok A TeX distribution is on PATH (latex, dvisvgm). If a diagram reports a missing")
            print("     package, install it with your distribution's package manager.")
        else:
            missing = 1
            print("  !! No TeX distribution found. Install TeX Live (https://tug.org/texlive/windows.html)")
            print("     or MiKTeX (winget install MiKTeX.MiKTeX), open a new terminal, then re-run:")
            print("         ziya-install-extras --latex")
    if "--dry-run" in args or "--plan" in args:
        print("(--dry-run: nothing was installed.)")
        return 0
    if not steps:
        return missing
    if not ({"-y", "--yes"} & set(args)):
        try:
            answer = input("Proceed with the above? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted; nothing installed.")
            return 0
    failed = 0
    for name, cmd in steps:
        print(f"  -> {subprocess.list2cmdline(cmd)}", flush=True)
        if subprocess.run(cmd).returncode != 0:
            print(f"  !! {name} failed.")
            failed = 1
    return failed or missing


def run(args: List[str]) -> int:
    """Run the packaged installer with ``args`` and return its exit code."""
    script = latex_script_path()
    if sys.platform == "win32":
        return run_windows(args, script)
    if script is None:
        print("!! Packaged installer not found (app/scripts/install_extras.sh).")
        print("   Reinstall ziya, or install a TeX distribution manually.")
        return 1
    # Invoke through bash explicitly so the file's exec bit is irrelevant;
    # wheel installs do not reliably preserve it.
    bash = shutil.which("bash") or "/bin/bash"
    if not os.path.exists(bash):
        print("!! bash not found; run the installer by hand:")
        print(f"   sh {script} {' '.join(args)}".rstrip())
        return 1
    # Tell the script which interpreter Ziya actually runs under. A wheel
    # installed into a venv (or a launcher-managed ``.venv``) is driven by that
    # venv's python, NOT the bare ``python3`` on PATH -- and Playwright ships
    # inside that venv. Without this the script probed the system ``python3``,
    # found no ``playwright`` there, and wrongly told the user to reinstall
    # ziya even though it was installed. sys.executable is exactly the venv
    # python whose Playwright will look for the Chromium build.
    env = {**os.environ, "ZIYA_PYTHON": sys.executable}
    return subprocess.run([bash, str(script), *args], env=env).returncode


def main(argv: Optional[List[str]] = None) -> int:
    """Pass every argument straight through to the script.

    Targets (--browser, --latex, --all) and modifiers (--dry-run, --yes,
    --help) are all understood by the script itself; nothing is filtered here,
    so an edition wrapper can forward its own argv unchanged.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
