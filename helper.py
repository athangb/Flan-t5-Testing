import os
import subprocess

def check_directory_exists(directory):
    return os.path.exists(directory)

def check_and_install_packages():
    required_packages = [
        'langchain',
        'gpt-index',
        'llama-index',
        'transformers',
        'sentence_transformers'
    ]
    
    for package in required_packages:
        if not is_package_installed(package):
            install_package(package)

def is_package_installed(package_name):
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
