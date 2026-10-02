"""Tests for PyPI wheel metadata and publishability."""
import subprocess
import sys
from pathlib import Path
import tomllib


def test_pyproject_has_required_pypi_fields():
    """PyPI requires specific fields for publishing."""
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)
    
    project = data["project"]
    
    # Required by PyPI
    assert "name" in project, "Missing 'name' in [project]"
    assert "version" in project, "Missing 'version' in [project]"
    assert "description" in project, "Missing 'description' in [project]"
    assert "readme" in project, "Missing 'readme' in [project]"
    assert "license" in project, "Missing 'license' in [project]"
    assert "authors" in project, "Missing 'authors' in [project]"
    assert "requires-python" in project, "Missing 'requires-python' in [project]"
    
    # Recommended classifiers
    assert "classifiers" in project, "Missing 'classifiers' in [project]"
    classifiers = project["classifiers"]
    assert any("License" in c for c in classifiers), "Missing license classifier"
    assert any("Python ::" in c for c in classifiers), "Missing Python version classifiers"


def test_pyproject_name_is_pypi_compatible():
    """PyPI package names must be lowercase, use hyphens or underscores (PEP 503)."""
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)
    
    name = data["project"]["name"]
    assert name == name.lower(), f"Package name must be lowercase: {name}"
    # PEP 503: hyphens, underscores, and dots are all normalized to hyphens.
    # We prefer hyphens for readability.
    import re
    assert re.match(r"^[a-z0-9][a-z0-9._-]*$", name), (
        f"Package name must match PEP 503: {name}"
    )


def test_readme_exists_and_is_markdown():
    """PyPI renders README as the package description."""
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)
    
    readme = data["project"]["readme"]
    if isinstance(readme, str):
        readme_path = Path(__file__).parent.parent / readme
    else:
        readme_path = Path(__file__).parent.parent / readme["file"]
    
    assert readme_path.exists(), f"README file not found: {readme_path}"
    assert readme_path.suffix in [".md", ".rst", ".txt"], \
        f"README should be markdown, reStructuredText, or plain text: {readme_path}"


def test_license_file_exists():
    """License file must exist for PyPI."""
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)
    
    license_info = data["project"]["license"]
    if isinstance(license_info, dict) and "file" in license_info:
        license_path = Path(__file__).parent.parent / license_info["file"]
        assert license_path.exists(), f"License file not found: {license_path}"
    else:
        # License specified as text or SPDX identifier; check LICENSE file exists anyway
        license_path = Path(__file__).parent.parent / "LICENSE"
        assert license_path.exists(), "LICENSE file not found"


def test_build_produces_wheel():
    """Verify maturin can build a wheel."""
    result = subprocess.run(
        ["maturin", "build", "--release", "-i", "/usr/bin/python3"],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"maturin build failed:\n{result.stderr}"
    
    # Check that a manylinux wheel was created (PyPI requires manylinux)
    target_dir = Path(__file__).parent.parent / "target" / "wheels"
    wheels = list(target_dir.glob("sun_solar_radiation-*-manylinux*.whl"))
    assert len(wheels) > 0, "No manylinux wheel file produced"


def test_wheel_metadata_is_valid():
    """Build a wheel and verify its metadata."""
    subprocess.run(
        ["maturin", "build", "--release", "--skip-auditwheel", "-i", "/usr/bin/python3"],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        check=True,
    )
    
    target_dir = Path(__file__).parent.parent / "target" / "wheels"
    wheels = list(target_dir.glob("sun_solar_radiation-*.whl"))
    assert len(wheels) > 0, "No wheel file found"
    
    wheel = wheels[0]
    
    # Extract and check METADATA
    import zipfile
    with zipfile.ZipFile(wheel) as zf:
        metadata_files = [f for f in zf.namelist() if f.endswith("/METADATA")]
        assert len(metadata_files) == 1, "Wheel should contain exactly one METADATA file"
        
        metadata = zf.read(metadata_files[0]).decode("utf-8")
        
        # Required fields in METADATA
        assert "Metadata-Version:" in metadata, "Missing Metadata-Version"
        assert "Name: sun-solar-radiation" in metadata, "Missing or wrong package name"
        assert "Version:" in metadata, "Missing Version"
        assert "Summary:" in metadata, "Missing Summary (description)"


def test_no_hardcoded_paths_in_wheel():
    """Wheels should not contain absolute paths from the build system."""
    subprocess.run(
        ["maturin", "build", "--release", "--skip-auditwheel", "-i", "/usr/bin/python3"],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        check=True,
    )
    
    target_dir = Path(__file__).parent.parent / "target" / "wheels"
    wheels = list(target_dir.glob("sun_solar_radiation-*.whl"))
    assert len(wheels) > 0
    
    wheel = wheels[0]
    
    import zipfile
    with zipfile.ZipFile(wheel) as zf:
        for name in zf.namelist():
            if name.endswith((".py", ".toml", ".txt", ".md")):
                content = zf.read(name).decode("utf-8", errors="ignore")
                # Check for common absolute path patterns
                assert "/home/" not in content, f"Found absolute path in {name}"
                assert "/Users/" not in content, f"Found absolute path in {name}"
                assert "C:\\Users\\" not in content, f"Found absolute path in {name}"
