"""
Simple test to ensure 'pip install tabstar' works on Python 3.9
This is a regression test to catch when code changes break Python 3.9 pip installation.
"""
import sys
import subprocess
import tempfile
import os
import shutil
import pytest
from pathlib import Path


def test_python39_pip_install_regression():
    """
    Regression test: Ensure pip install tabstar still works on Python 3.9
    
    This test:
    1. Creates a fresh Python 3.9 virtual environment
    2. Installs tabstar from the current repo
    3. Tests that basic imports work
    4. Reports success/failure
    
    Use this to verify code changes don't break Python 3.9 compatibility.
    """
    
    # Skip if python3.9 not available
    if shutil.which('python3.9') is None:
        pytest.skip("python3.9 not available - cannot test pip installation")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        venv_path = Path(temp_dir) / "tabstar_py39_test"
        
        try:
            print("🔧 Creating fresh Python 3.9 environment...")
            
            # Create fresh Python 3.9 venv
            subprocess.run([
                'python3.9', '-m', 'venv', str(venv_path)
            ], check=True, capture_output=True)
            
            # Get paths
            if os.name == 'nt':  # Windows
                python_exe = venv_path / 'Scripts' / 'python.exe'
                pip_exe = venv_path / 'Scripts' / 'pip.exe'
            else:  # Unix-like
                python_exe = venv_path / 'bin' / 'python'
                pip_exe = venv_path / 'bin' / 'pip'
            
            # Verify Python 3.9
            version_check = subprocess.run([
                str(python_exe), '-c', 
                'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
            ], capture_output=True, text=True, check=True)
            
            version = version_check.stdout.strip()
            assert version == "3.9", f"Expected Python 3.9, got {version}"
            print(f"✅ Python {version} environment ready")
            
            # Install tabstar from current repo
            repo_root = Path(__file__).parent.parent
            print("📦 Installing tabstar...")
            
            subprocess.run([
                str(pip_exe), 'install', '-e', str(repo_root)
            ], check=True, capture_output=True)
            
            print("✅ Installation complete")
            
            # Test that imports work
            import_test = subprocess.run([
                str(python_exe), '-c', '''
import sys
import tabstar
from tabstar.tabstar_model import TabSTARClassifier

# Verify version
assert sys.version_info[:2] == (3, 9), f"Wrong Python version: {sys.version_info[:2]}"

print("SUCCESS: pip install tabstar works on Python 3.9")
'''
            ], capture_output=True, text=True)
            
            if import_test.returncode != 0:
                print(f"❌ Import test failed:")
                print(f"STDOUT: {import_test.stdout}")
                print(f"STDERR: {import_test.stderr}")
                pytest.fail("TabSTAR imports failed after pip install")
            
            assert "SUCCESS" in import_test.stdout
            print("✅ Import test passed")
            print("🎉 pip install tabstar works on Python 3.9!")
            
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Pip installation test failed: {e}")


if __name__ == "__main__":
    # Run test directly
    test_python39_pip_install_regression()