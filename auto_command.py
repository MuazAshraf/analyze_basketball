#!/usr/bin/env python3
"""
Git Repository Automation Script
Handles .gitignore creation and git operations for new and existing repositories.
"""

import os
import subprocess
import sys
from pathlib import Path
from collections import defaultdict


def run_command(command, check=True):
    """Execute shell command and return result."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            check=check
        )
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stdout.strip() if e.stdout else "", e.stderr.strip() if e.stderr else ""


def get_gitignore_content():
    """Get the standard .gitignore content."""
    return """# Virtual Environments and Dev Environment
venv/
.venv/
env/
.env/
.devenv/
.aider*/
.aider.*/
.devenv*/
devenv*/

# Development Tools
.aider.chat.history.md
.aider.input.history
.devenv.flake.nix
devenv.lock
devenv.nix
devenv.yaml

# Python
__pycache__/
*.py[cod]
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebooks
*.ipynb
.ipynb_checkpoints/


# IDE and Editor files
.vscode/
.idea/
*.swp
*.swo
*~

# Environment variables
.env
.env.*

# Data and Media files
*.csv
*.json
*.pdf
*.xlsx
*.xls
*.txt
!requirements.txt
*.mp3
*.mp4
*.wav
*.db
*.sqlite
*.sqlite3

# Images
*.png
*.jpg
*.jpeg
*.gif
*.svg
*.ico

# Model files
*.ckpt
*.pt
*.pth
*.bin
*.model
*.h5
pretrained_models/

# Cache and temp files
.cache/
.pytest_cache/
.coverage
htmlcov/
.aider.tags.cache.v3/

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Binary and Library files
*.pyd
*.dll
*.lib
*.exe

# Large files and directories
*.model
*.bin
*.pdf
data/transcripts/
*.mp3
*.wav
*.mp4
.aider*
.pickle*
node_modules/
"""


def create_gitignore():
    """Create .gitignore file with common exclusions."""
    gitignore_content = get_gitignore_content()
    
    # Check if .gitignore already exists
    if os.path.exists('.gitignore'):
        try:
            with open('.gitignore', 'r') as f:
                existing_content = f.read()
            
            # Check if content matches exactly
            if existing_content.strip() == gitignore_content.strip():
                print("ℹ️  .gitignore file already exists with expected content")
                return True
            else:
                print("⚠️  .gitignore file exists but has different content")
                response = input("Do you want to overwrite it? (y/N): ").strip().lower()
                if response != 'y':
                    print("⏭️  Keeping existing .gitignore")
                    return True
        except Exception as e:
            print(f"⚠️  Could not read existing .gitignore: {e}")
    
    try:
        with open('.gitignore', 'w') as f:
            f.write(gitignore_content)
        print("✅ Created .gitignore file")
        return True
    except Exception as e:
        print(f"❌ Failed to create .gitignore: {e}")
        return False


def analyze_changes():
    """Analyze git changes and return detailed information."""
    success, stdout, _ = run_command("git status --porcelain", check=False)
    if not success:
        return None, None, None
    
    changes = {
        'added': [],
        'modified': [],
        'deleted': [],
        'renamed': [],
        'untracked': []
    }
    
    total_files = 0
    
    for line in stdout.split('\n'):
        if not line.strip():
            continue
            
        status = line[:2]
        filename = line[3:].strip()
        total_files += 1
        
        # Parse git status codes
        if status == 'A ' or status == 'AM':
            changes['added'].append(filename)
        elif status == 'M ' or status == ' M' or status == 'MM':
            changes['modified'].append(filename)
        elif status == 'D ' or status == ' D':
            changes['deleted'].append(filename)
        elif status.startswith('R'):
            changes['renamed'].append(filename)
        elif status == '??':
            changes['untracked'].append(filename)
        else:
            # Handle other status codes
            if 'M' in status:
                changes['modified'].append(filename)
            elif 'A' in status:
                changes['added'].append(filename)
            elif 'D' in status:
                changes['deleted'].append(filename)
    
    return changes, total_files, stdout


def display_changes(changes, total_files):
    """Display detailed change information."""
    if total_files == 0:
        print("ℹ️  No changes to commit")
        return
    
    print(f"\n📊 Changes Summary ({total_files} files):")
    print("=" * 50)
    
    if changes['untracked']:
        print(f"🆕 New files ({len(changes['untracked'])}):")
        for file in changes['untracked'][:10]:  # Show first 10
            print(f"   + {file}")
        if len(changes['untracked']) > 10:
            print(f"   ... and {len(changes['untracked']) - 10} more")
    
    if changes['added']:
        print(f"➕ Added files ({len(changes['added'])}):")
        for file in changes['added'][:10]:
            print(f"   + {file}")
        if len(changes['added']) > 10:
            print(f"   ... and {len(changes['added']) - 10} more")
    
    if changes['modified']:
        print(f"✏️  Modified files ({len(changes['modified'])}):")
        for file in changes['modified'][:10]:
            print(f"   ~ {file}")
        if len(changes['modified']) > 10:
            print(f"   ... and {len(changes['modified']) - 10} more")
    
    if changes['deleted']:
        print(f"🗑️  Deleted files ({len(changes['deleted'])}):")
        for file in changes['deleted'][:10]:
            print(f"   - {file}")
        if len(changes['deleted']) > 10:
            print(f"   ... and {len(changes['deleted']) - 10} more")
    
    if changes['renamed']:
        print(f"📝 Renamed files ({len(changes['renamed'])}):")
        for file in changes['renamed'][:10]:
            print(f"   → {file}")
        if len(changes['renamed']) > 10:
            print(f"   ... and {len(changes['renamed']) - 10} more")


def generate_commit_message(changes, total_files, is_initial=False):
    """Generate descriptive commit message based on changes."""
    if is_initial:
        return "initial commit"
    
    if total_files == 0:
        return "no changes"
    
    parts = []
    
    # Count different types of changes
    counts = {k: len(v) for k, v in changes.items() if v}
    
    if counts.get('untracked', 0) > 0 or counts.get('added', 0) > 0:
        new_files = counts.get('untracked', 0) + counts.get('added', 0)
        parts.append(f"add {new_files} file{'s' if new_files != 1 else ''}")
    
    if counts.get('modified', 0) > 0:
        parts.append(f"update {counts['modified']} file{'s' if counts['modified'] != 1 else ''}")
    
    if counts.get('deleted', 0) > 0:
        parts.append(f"remove {counts['deleted']} file{'s' if counts['deleted'] != 1 else ''}")
    
    if counts.get('renamed', 0) > 0:
        parts.append(f"rename {counts['renamed']} file{'s' if counts['renamed'] != 1 else ''}")
    
    if not parts:
        return "misc changes"
    
    # Create readable commit message
    if len(parts) == 1:
        return parts[0]
    elif len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    else:
        return f"{', '.join(parts[:-1])}, and {parts[-1]}"


def is_git_repo():
    """Check if current directory is already a git repository."""
    return os.path.exists('.git')


def has_remote_origin():
    """Check if remote origin is configured."""
    success, stdout, _ = run_command("git remote -v", check=False)
    return success and "origin" in stdout


def get_current_branch():
    """Get current git branch name."""
    success, stdout, _ = run_command("git branch --show-current", check=False)
    return stdout if success else "main"


def safe_git_add():
    """Safely add files to git, handling problematic files."""
    print("🔄 Adding files to git...")
    
    # First try normal git add
    success, _, error = run_command("git add .", check=False)
    if success:
        print("✅ Files added to staging")
        return True
    
    print(f"⚠️  Standard git add failed: {error}")
    print("🔄 Trying to add files individually...")
    
    # Try to add files one by one, skipping problematic ones
    try:
        # Get list of files to add
        success, stdout, _ = run_command("git status --porcelain", check=False)
        if not success:
            print("❌ Could not get git status")
            return False
        
        files_added = 0
        for line in stdout.split('\n'):
            if not line.strip():
                continue
                
            # Parse git status output
            status = line[:2]
            filename = line[3:].strip()
            
            # Skip files that are already tracked or deleted
            if status.strip() in ['D', 'R']:
                continue
                
            # Try to add this specific file
            success, _, file_error = run_command(f'git add "{filename}"', check=False)
            if success:
                files_added += 1
            else:
                print(f"⚠️  Skipped problematic file: {filename}")
        
        if files_added > 0:
            print(f"✅ Added {files_added} files to staging")
            return True
        else:
            print("ℹ️  No files to add")
            return True
            
    except Exception as e:
        print(f"❌ Error during selective file adding: {e}")
        return False


def initialize_new_repo():
    """Initialize new git repository."""
    print("🔄 Initializing new git repository...")
    
    # Git init
    success, _, error = run_command("git init")
    if not success:
        print(f"❌ Failed to initialize git repo: {error}")
        return False
    print("✅ Git repository initialized")
    
    # Analyze changes before adding
    changes, total_files, _ = analyze_changes()
    if changes and total_files > 0:
        display_changes(changes, total_files)
    
    # Git add with error handling
    if not safe_git_add():
        return False
    
    # Initial commit
    commit_msg = generate_commit_message(changes, total_files, is_initial=True)
    success, _, error = run_command(f'git commit -m "{commit_msg}"')
    if not success:
        print(f"❌ Failed to commit: {error}")
        return False
    print(f"✅ Initial commit created: '{commit_msg}'")
    
    # Ask for remote URL
    while True:
        remote_url = input("🔗 Enter remote repository URL (or press Enter to skip): ").strip()
        if not remote_url:
            print("⏭️  Skipping remote setup")
            return True
            
        # Add remote origin
        success, _, error = run_command(f'git remote add origin "{remote_url}"')
        if not success:
            print(f"❌ Failed to add remote: {error}")
            continue
        print("✅ Remote origin added")
        
        # Get current branch
        current_branch = get_current_branch()
        
        # Push to remote
        print(f"🚀 Pushing {total_files} files to GitHub...")
        success, _, error = run_command(f"git push -u origin {current_branch}")
        if not success:
            print(f"❌ Failed to push: {error}")
            print("You may need to create the repository on the remote first")
            return False
        print(f"✅ Successfully pushed to GitHub (origin/{current_branch})")
        return True


def update_existing_repo():
    """Update existing git repository."""
    print("🔄 Updating existing git repository...")
    
    # First, pull latest changes from remote
    if has_remote_origin():
        current_branch = get_current_branch()
        print(f"🔽 Pulling latest changes from remote...")
        print(f"   Current branch: {current_branch}")
        
        # Check if remote branch exists
        success, _, _ = run_command(f"git ls-remote --exit-code --heads origin {current_branch}", check=False)
        if not success:
            print(f"⚠️  Remote branch '{current_branch}' doesn't exist on origin")
            print("   This is normal for new branches - it will be created when you push")
            
            # Try to pull from main/master instead if available
            main_branches = ['main', 'master']
            pulled_from_main = False
            
            for main_branch in main_branches:
                success, _, _ = run_command(f"git ls-remote --exit-code --heads origin {main_branch}", check=False)
                if success:
                    print(f"   Attempting to pull latest changes from origin/{main_branch} instead...")
                    success, stdout, error = run_command(f"git pull origin {main_branch}", check=False)
                    if success:
                        print(f"✅ Successfully pulled from origin/{main_branch}")
                        pulled_from_main = True
                        break
                    else:
                        print(f"⚠️  Failed to pull from {main_branch}: {error}")
            
            if not pulled_from_main:
                print("ℹ️  No main/master branch to pull from - continuing with local changes")
        else:
            # Remote branch exists, try to pull from it
            success, stdout, error = run_command(f"git pull origin {current_branch}", check=False)
            
            if success:
                if "Already up to date" in stdout or "Already up-to-date" in stdout:
                    print("✅ Repository is already up to date")
                elif "Fast-forward" in stdout:
                    print("✅ Fast-forwarded to latest changes")
                    # Show what was pulled
                    lines = stdout.split('\n')
                    for line in lines:
                        if 'file' in line and ('changed' in line or 'insertion' in line or 'deletion' in line):
                            print(f"   📥 {line.strip()}")
                elif stdout.strip():
                    print("✅ Successfully pulled changes")
                    print(f"   📥 {stdout.strip()}")
                else:
                    print("✅ Pull completed")
            else:
                if "merge conflict" in error.lower() or "conflict" in error.lower():
                    print(f"⚠️  Merge conflicts detected:")
                    print(f"   🔍 {error}")
                    print("   ⚡ Please resolve conflicts manually and run the script again")
                    return False
                elif "diverged" in error.lower():
                    print(f"⚠️  Branches have diverged:")
                    print(f"   🔍 {error}")
                    print("   💡 You may need to merge or rebase manually")
                    
                    response = input("Do you want to continue anyway? (y/N): ").strip().lower()
                    if response != 'y':
                        print("   ⏹️  Stopping to let you handle the divergence")
                        return False
                elif "couldn't find remote ref" in error.lower():
                    print(f"⚠️  Remote branch '{current_branch}' not found on origin")
                    print("   This branch will be created when you push")
                else:
                    print(f"⚠️  Pull failed: {error}")
                    response = input("Do you want to continue without pulling? (y/N): ").strip().lower()
                    if response != 'y':
                        return False
    else:
        print("ℹ️  No remote origin configured, skipping pull")
    
    # Analyze changes before adding
    changes, total_files, _ = analyze_changes()
    if total_files == 0:
        print("ℹ️  No local changes to commit")
        return True
    
    display_changes(changes, total_files)
    
    # Git add with error handling
    if not safe_git_add():
        return False
    
    # Check if there are changes to commit after adding
    success, stdout, _ = run_command("git status --porcelain")
    if success and not stdout:
        print("ℹ️  No changes to commit after staging")
        return True
    
    # Generate descriptive commit message
    commit_msg = generate_commit_message(changes, total_files)
    
    # Commit changes
    success, _, error = run_command(f'git commit -m "{commit_msg}"')
    if not success:
        print(f"❌ Failed to commit: {error}")
        return False
    print(f"✅ Changes committed: '{commit_msg}'")
    
    # Check if remote exists
    if not has_remote_origin():
        print("⚠️  No remote origin configured. Skipping push.")
        return True
    
    # Push changes
    current_branch = get_current_branch()
    print(f"🚀 Pushing {total_files} changed files to GitHub...")
    
    # Check if this is the first push to this branch
    success, _, _ = run_command(f"git ls-remote --exit-code --heads origin {current_branch}", check=False)
    if not success:
        print(f"   Creating new remote branch: {current_branch}")
        success, _, error = run_command(f"git push -u origin {current_branch}")
    else:
        success, _, error = run_command(f"git push origin {current_branch}")
    
    if not success:
        # Handle common push errors
        if "rejected" in error.lower() and "non-fast-forward" in error.lower():
            print("❌ Push rejected - remote has newer commits")
            print("   💡 This shouldn't happen since we pulled, but trying force push with lease...")
            
            success, _, error2 = run_command(f"git push --force-with-lease origin {current_branch}", check=False)
            if success:
                print("✅ Force push with lease successful")
            else:
                print(f"❌ Force push also failed: {error2}")
                print("   🔧 Please check the repository status manually")
                return False
        else:
            print(f"❌ Failed to push: {error}")
            return False
    else:
        print(f"✅ Successfully pushed to GitHub (origin/{current_branch})")
    
    # Show what was pushed
    print(f"\n🎯 Pushed to GitHub:")
    print(f"   📁 Repository: origin/{current_branch}")
    print(f"   💾 Commit: '{commit_msg}'")
    print(f"   📊 Files: {total_files} changed")
    
    return True


def main():
    """Main automation function."""
    print("🚀 Git Repository Automation Script")
    print("=" * 40)
    
    # Create .gitignore
    if not create_gitignore():
        sys.exit(1)
    
    # Check if this is a new or existing repo
    if is_git_repo():
        print("📁 Existing git repository detected")
        success = update_existing_repo()
    else:
        print("🆕 New repository - initializing...")
        success = initialize_new_repo()
    
    if success:
        print("\n🎉 Git automation completed successfully!")
    else:
        print("\n💥 Git automation failed!")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
