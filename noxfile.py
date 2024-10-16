import pathlib

import nox

nox.options.sessions = []


@nox.session
def format(session: nox.Session) -> None:  # noqa: A001
    """Format all source files to a consistent style."""
    sources = ["src", "tests", "noxfile.py"]
    session.run("isort", *sources, external=True)
    session.run(
        "docformatter",
        "--in-place",
        "--recursive",
        *sources,
        external=True,
    )
    session.run("black", *sources, external=True)

    sources_cpp = ["src"]
    sources_cpp_files = []
    for s in sources_cpp:
        file = pathlib.Path(s)
        if file.is_file():
            sources_cpp_files.append(str(file))
        elif file.is_dir():
            for ext in [".h", ".hpp", ".cuh", ".c", ".cpp", ".cu"]:
                sources_cpp_files.extend([str(f) for f in sorted(file.rglob(f"*{ext}"))])
    session.run("clang-format", "-i", *sources_cpp_files, external=True)


@nox.session
def lint(session: nox.Session) -> None:
    """Check the source code with linters."""
    failed = False
    sources = ["src", "tests", "noxfile.py"]
    try:
        session.run("isort", "--check", *sources, external=True)
    except nox.command.CommandFailed:
        failed = True

    try:
        session.run(
            "docformatter",
            "--check",
            "--recursive",
            *sources,
            external=True,
        )
    except nox.command.CommandFailed:
        failed = True

    try:
        session.run("black", "--check", *sources, external=True)
    except nox.command.CommandFailed:
        failed = True

    try:
        sources_cpp = ["src"]
        sources_cpp_files = []
        for s in sources_cpp:
            file = pathlib.Path(s)
            if file.is_file():
                sources_cpp_files.append(str(file))
            elif file.is_dir():
                for ext in [".h", ".hpp", ".cuh", ".c", ".cpp", ".cu"]:
                    sources_cpp_files.extend([str(f) for f in sorted(file.rglob(f"*{ext}"))])
        session.run("clang-format", "--dry-run", "--Werror", *sources_cpp_files, external=True)
    except nox.command.CommandFailed:
        failed = True

    try:
        text_sources = [*sources, "README.md", "CHANGELOG.md"]
        skip_sources = ["*pdf"]
        session.run(
            "codespell",
            "--check-filenames",
            "--check-hidden",
            *text_sources,
            "--skip",
            ",".join(skip_sources),
            external=True,
        )
    except nox.command.CommandFailed:
        failed = True

    if failed:
        raise nox.command.CommandFailed


@nox.session
def tests(session: nox.Session) -> None:
    """Run the unit tests."""
    session.run("pytest", "tests", external=True)
