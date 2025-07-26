#!/usr/bin/env python3
"""
Terminal RAG System Launcher
Easy launcher with setup validation for the Terminal RAG System.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_file_exists(file_path: str, description: str) -> bool:
    """Check if a required file exists."""
    if Path(file_path).exists():
        print(f"✅ {description}: Found")
        return True
    else:
        print(f"❌ {description}: Missing ({file_path})")
        return False

def check_env_variable(var_name: str) -> bool:
    """Check if an environment variable is set."""
    value = os.getenv(var_name)
    if value and not value.startswith('your_'):
        print(f"✅ {var_name}: Set")
        return True
    else:
        print(f"❌ {var_name}: Not set or using placeholder value")
        return False

def check_package_installed(package_name: str) -> bool:
    """Check if a Python package is installed."""
    try:
        __import__(package_name)
        print(f"✅ {package_name}: Installed")
        return True
    except ImportError:
        print(f"❌ {package_name}: Not installed")
        return False

def run_setup_check() -> bool:
    """Run comprehensive setup validation."""
    print("🔍 Terminal RAG System - Setup Checker")
    print("=" * 50)
    
    all_good = True
    
    # Check required files
    print("\n📁 Checking required files:")
    required_files = [
        ("terminal_rag.py", "Main script"),
        (".env", "Environment file")
    ]
    
    for file_path, description in required_files:
        if not check_file_exists(file_path, description):
            all_good = False
    
    # Load environment variables from .env file if it exists
    if Path(".env").exists():
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            print("⚠️  python-dotenv not installed, cannot load .env file")
    
    # Check environment variables
    print("\n🔑 Checking environment variables:")
    required_env_vars = [
        "OPENAI_API_KEY",
        "SUPABASE_URL", 
        "SUPABASE_SERVICE_KEY"
    ]
    
    for env_var in required_env_vars:
        if not check_env_variable(env_var):
            all_good = False
    
    # Check Python packages
    print("\n📦 Checking Python packages:")
    required_packages = ["openai", "supabase", "dotenv"]
    
    for package in required_packages:
        if not check_package_installed(package):
            all_good = False
    
    print("\n" + "=" * 50)
    
    if all_good:
        print("✅ All checks passed! Starting Terminal RAG System...")
        return True
    else:
        print("❌ Setup incomplete. Please fix the issues above before running.")
        print("\nQuick fixes:")
        print("1. Copy .env.example to .env and add your API keys")
        print("2. Install packages: pip install -r requirements.txt")
        print("3. Make sure you've run the database setup SQL in Supabase")
        return False

def install_dependencies():
    """Install required Python packages."""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        return False

def main():
    """Main launcher function."""
    print("🚀 Terminal RAG System Launcher")
    print("-" * 30)
    
    # Check if we're in the right directory
    if not Path("terminal_rag.py").exists():
        print("❌ terminal_rag.py not found in current directory")
        print("Please make sure you're in the terminal-rag project folder")
        sys.exit(1)
    
    # Check if requirements.txt exists and packages are installed
    if Path("requirements.txt").exists():
        try:
            import openai, supabase, dotenv
        except ImportError:
            print("📦 Some dependencies are missing.")
            install_choice = input("Would you like to install them now? (y/n): ").lower().strip()
            if install_choice == 'y':
                if not install_dependencies():
                    sys.exit(1)
            else:
                print("Please install dependencies manually: pip install -r requirements.txt")
                sys.exit(1)
    
    # Run setup validation
    if not run_setup_check():
        sys.exit(1)
    
    print()
    
    # Launch the main application
    try:
        from terminal_rag import main as terminal_rag_main
        terminal_rag_main()
    except ImportError as e:
        print(f"❌ Error importing terminal_rag: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error running Terminal RAG System: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()