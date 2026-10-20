#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["tomli>=2; python_version < '3.11'"]
# ///
"""Package adapted SGLang and kernel as wheels or source distributions."""

import argparse
import fcntl
import os
import re
import shlex
import shutil
import subprocess
import zipfile
from functools import partial
from pathlib import Path
from typing import Literal, TypedDict

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SGLANG_SOURCE_DIRECTORY: Path = PROJECT_ROOT / "third_party/sglang"


class BuildArguments(argparse.Namespace):
    profile: str
    component: Literal["all", "sglang", "kernel"]
    format: Literal["wheel", "sdist", "all"]
    jobs: int
    cuda_home: Path | None
    dependency_cache: Path | None
    index_url: str


class ProfileEdit(TypedDict):
    path: str
    old: str
    new: str


class BuildProfile(TypedDict):
    patches: list[str]
    source_commit: str
    python: str
    sglang_version: str
    torch_version: str
    torch_index: str
    build_requirements: list[str]
    environment: dict[str, str]
    kernel_settings: dict[str, str]
    edits: list[ProfileEdit]


def run_command(
    command: list[str | Path],
    capture: bool = False,
    *,
    build_environment: dict[str, str],
) -> str:
    command_arguments = [str(command_argument) for command_argument in command]
    print("+", shlex.join(command_arguments), flush=True)
    result = subprocess.run(
        command_arguments,
        cwd=PROJECT_ROOT,
        env=build_environment,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout if capture else ""


def export_source(
    repository_directory: Path,
    source_commit: str,
    source_directory: Path,
    build_environment: dict[str, str],
) -> None:
    run_build_command = partial(run_command, build_environment=build_environment)
    actual_commit = run_build_command(
        ["git", "-C", repository_directory, "rev-parse", "HEAD"], capture=True
    ).strip()
    if actual_commit != source_commit:
        raise RuntimeError(
            f"Source commit differs from the build profile: {repository_directory}"
        )
    else:
        pass
    if run_build_command(
        ["git", "-C", repository_directory, "status", "--porcelain"], capture=True
    ).strip():
        raise RuntimeError(
            f"Commit source changes before building: {repository_directory}"
        )
    else:
        pass
    if source_directory.exists():
        shutil.rmtree(source_directory)
    else:
        pass
    source_archive_path = source_directory.with_suffix(".zip")
    run_build_command(
        [
            "git",
            "-C",
            repository_directory,
            "archive",
            "--format=zip",
            "-o",
            source_archive_path,
            source_commit,
        ]
    )
    with zipfile.ZipFile(source_archive_path) as source_archive:
        source_archive.extractall(source_directory)
    source_archive_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default="cu129",
        choices=[
            profile_path.stem
            for profile_path in (PROJECT_ROOT / "packaging/profiles").glob("*.toml")
        ],
    )
    parser.add_argument(
        "--component", choices=["all", "sglang", "kernel"], default="all"
    )
    parser.add_argument("--format", choices=["wheel", "sdist", "all"], default="wheel")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--cuda-home",
        type=Path,
        default=os.environ.get("CUDA_HOME"),
    )
    parser.add_argument("--dependency-cache", type=Path)
    parser.add_argument("--index-url", required=True)
    build_arguments = parser.parse_args(namespace=BuildArguments())
    if build_arguments.jobs < 1:
        parser.error("--jobs must be positive")
    else:
        pass
    if (
        build_arguments.component in ("all", "kernel")
        and build_arguments.cuda_home is None
    ):
        parser.error("--cuda-home or CUDA_HOME is required for kernel builds")
    else:
        pass
    profile: BuildProfile = tomllib.loads(
        (
            PROJECT_ROOT / "packaging/profiles" / f"{build_arguments.profile}.toml"
        ).read_text()
    )
    build_directory = PROJECT_ROOT / ".build" / build_arguments.profile
    output_directory = PROJECT_ROOT / "dist" / build_arguments.profile
    build_directory.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(parents=True, exist_ok=True)
    with (build_directory / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build(build_arguments, profile, build_directory, output_directory)


def build(
    build_arguments: BuildArguments,
    profile: BuildProfile,
    build_directory: Path,
    output_directory: Path,
) -> None:
    build_environment = os.environ.copy()
    for configuration_name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "BASH_ENV"):
        build_environment.pop(configuration_name, None)
    build_environment.update(profile.get("environment", {}))
    build_environment["PYTHONNOUSERSITE"] = "1"
    run_build_command = partial(run_command, build_environment=build_environment)
    build_environment["CMAKE_BUILD_PARALLEL_LEVEL"] = str(build_arguments.jobs)

    source_directory = build_directory / "source"
    export_source(
        SGLANG_SOURCE_DIRECTORY,
        profile["source_commit"],
        source_directory,
        build_environment,
    )
    for edit in profile["edits"]:
        path = source_directory / edit["path"]
        configuration_text = path.read_text()
        if configuration_text.count(edit["old"]) != 1:
            raise SystemExit(f"Cannot apply profile edit to {edit['path']}")
        else:
            pass
        path.write_text(configuration_text.replace(edit["old"], edit["new"], 1))
    for patch in profile.get("patches", []):
        run_build_command(
            [
                "git",
                "apply",
                "--directory",
                source_directory.relative_to(PROJECT_ROOT),
                PROJECT_ROOT / "packaging/patches" / patch,
            ]
        )
    for file_name in ("README.md", "LICENSE"):
        if not (source_directory / "python" / file_name).exists():
            shutil.copy2(
                source_directory / file_name, source_directory / "python" / file_name
            )
        else:
            pass

    project_configuration_path = source_directory / "python/pyproject.toml"
    configuration_text = project_configuration_path.read_text().replace(
        'dynamic = ["version"]', f'version = "{profile["sglang_version"]}"', 1
    )
    configuration_text = re.sub(
        r"\[tool\.setuptools_scm\]\n.*?(?=\n\[|\Z)", "", configuration_text, flags=re.S
    )
    project_configuration_path.write_text(configuration_text)
    (source_directory / "python/sglang/_version.py").write_text(
        f"__version__ = {profile['sglang_version']!r}\n"
        '__version_tuple__ = tuple(__version__.split("."))\n'
    )
    kernel = source_directory / "python/sglang/kernels/aot"
    project_configuration_path = kernel / "pyproject.toml"
    configuration_text = project_configuration_path.read_text().replace(
        '"torch==2.13.0"', f'"torch=={profile["torch_version"]}"'
    )
    configuration_text = configuration_text.replace(
        'cmake.build-type = "Release"',
        'cmake.build-type = "Release"\ncmake.version = ">=3.31,<4"',
    )
    configuration_text += '\n[tool.scikit-build.sdist]\ninclude = ["**"]\n'
    configuration_text += "\n[tool.scikit-build.cmake.define]\n"
    for configuration_name, configuration_value in profile["kernel_settings"].items():
        configuration_text += f'"{configuration_name.removeprefix("cmake.define.")}" = "{configuration_value}"\n'
    project_configuration_path.write_text(configuration_text)
    cmake_configuration_path = kernel / "CMakeLists.txt"
    cmake_configuration_path.write_text(
        cmake_configuration_path.read_text().replace(
            "find_package(Torch REQUIRED)",
            f'set(ENV{{TORCH_CUDA_ARCH_LIST}} "{profile["environment"]["TORCH_CUDA_ARCH_LIST"]}")\n'
            "find_package(Torch REQUIRED)",
            1,
        )
    )

    uv_command = ["uv", "--no-config"]
    build_environment_directory = build_directory / "venv"
    build_python = build_environment_directory / "bin/python"
    if not build_python.exists():
        run_build_command(
            [
                *uv_command,
                "venv",
                "--python",
                profile["python"],
                build_environment_directory,
            ]
        )
    else:
        pass
    install_command = [
        *uv_command,
        "pip",
        "install",
        "--python",
        build_python,
        "--index-url",
        build_arguments.index_url,
    ]
    run_build_command([*install_command, *profile["build_requirements"]])
    build_environment["PATH"] = (
        str(build_environment_directory / "bin")
        + os.pathsep
        + build_environment.get("PATH", "")
    )
    projects = []
    if build_arguments.component in ("all", "kernel"):
        build_environment["CUDA_HOME"] = str(build_arguments.cuda_home.resolve())
        build_environment["CUDACXX"] = str(
            build_arguments.cuda_home.resolve() / "bin/nvcc"
        )
        run_build_command(
            [
                *install_command,
                "--index",
                profile["torch_index"],
                f"torch=={profile['torch_version']}",
            ]
        )
        projects.append(kernel)
    else:
        pass
    if build_arguments.component in ("all", "sglang"):
        projects.append(source_directory / "python")
    else:
        pass
    build_formats = (
        ["--wheel", "--sdist"]
        if build_arguments.format == "all"
        else [f"--{build_arguments.format}"]
    )
    for project in projects:
        command = [
            *uv_command,
            "build",
            *build_formats,
            "--no-build-isolation",
            "--python",
            build_python,
            "--out-dir",
            output_directory,
            project,
        ]
        if project == kernel:
            command.extend(
                ["--config-setting", f"build-dir={build_directory / 'kernel-build'}"]
            )
        else:
            pass
        if project == kernel and build_arguments.dependency_cache is not None:
            for dependency_directory in build_arguments.dependency_cache.glob(
                "repo-*-src"
            ):
                dependency_name = dependency_directory.name.removesuffix("-src").upper()
                command.extend(
                    [
                        "--config-setting",
                        f"cmake.define.FETCHCONTENT_SOURCE_DIR_{dependency_name}={dependency_directory.resolve()}",
                    ]
                )
        else:
            pass
        run_build_command(command)
    print(f"Packages: {output_directory}")


if __name__ == "__main__":
    main()
else:
    pass
