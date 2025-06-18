#!/usr/bin/env python3
"""
Professional Git Workflow Automation Script
Supports development, staging, and production environments with branch management and proper merging.
"""

import os
import subprocess
import sys
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime


class GitWorkflow:
    def __init__(self):
        self.environments = {
            'dev': {
                'branch': 'development',
                'description': '🛠️  Development - Active feature development',
                'upstream': None,
                'auto_deploy': False
            },
            'staging': {
                'branch': 'staging', 
                'description': '🎭 Staging - Client review and testing',
                'upstream': 'development',
                'auto_deploy': True
            },
            'prod': {
                'branch': 'master',
                'description': '🚀 Production - Live environment', 
                'upstream': 'staging',
                'auto_deploy': False
            }
        }
        self.config_file = '.git_workflow_config.json'
        self.load_config()

    def run_command(self, command, check=True):
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

    def load_config(self):
        """Load workflow configuration from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    for env, settings in config.get('environments', {}).items():
                        if env in self.environments:
                            self.environments[env].update(settings)
            except Exception as e:
                print(f"⚠️  Could not load config: {e}")

    def save_config(self):
        """Save workflow configuration to file."""
        try:
            config = {'environments': self.environments}
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save config: {e}")

    def get_gitignore_content(self):
        """Get environment-aware .gitignore content."""
        return """# Environments
.env
.env.*
venv/
.venv/
env/
.env/
.devenv/

# Development Tools
.aider*/
.aider.*/
node_modules/
.git_workflow_config.json

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

# IDE and Editor files
.vscode/
.idea/
*.swp
*.swo
*~

# Environment-specific files
config/development.json
config/staging.json
config/production.json
logs/
*.log

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

# Model and Binary files
*.ckpt
*.pt
*.pth
*.bin
*.model
*.h5
*.pyd
*.dll
*.lib
*.exe

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
"""

    def create_gitignore(self):
        """Create .gitignore file with environment-aware exclusions."""
        gitignore_content = self.get_gitignore_content()
        
        if os.path.exists('.gitignore'):
            try:
                with open('.gitignore', 'r') as f:
                    existing_content = f.read()
                
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
            print("✅ Created environment-aware .gitignore file")
            return True
        except Exception as e:
            print(f"❌ Failed to create .gitignore: {e}")
            return False

    def select_environment(self):
        """Let user select target environment."""
        print("\n🌍 Select Target Environment:")
        print("=" * 50)
        
        for key, env in self.environments.items():
            print(f"{key}. {env['description']}")
            print(f"   Branch: {env['branch']}")
            if env['upstream']:
                print(f"   Merges from: {env['upstream']}")
            print()
        
        while True:
            choice = input("Choose environment (dev/staging/prod): ").strip().lower()
            if choice in self.environments:
                return choice
            print("❌ Invalid choice. Please select dev, staging, or prod")

    def get_current_branch(self):
        """Get current git branch name."""
        success, stdout, _ = self.run_command("git branch --show-current", check=False)
        return stdout if success else "main"

    def branch_exists(self, branch_name):
        """Check if branch exists locally."""
        success, _, _ = self.run_command(f"git show-ref --verify --quiet refs/heads/{branch_name}", check=False)
        return success

    def remote_branch_exists(self, branch_name):
        """Check if branch exists on remote."""
        success, _, _ = self.run_command(f"git ls-remote --exit-code --heads origin {branch_name}", check=False)
        return success

    def create_branch(self, branch_name, from_branch=None):
        """Create new branch."""
        if from_branch:
            success, _, error = self.run_command(f"git checkout -b {branch_name} {from_branch}")
        else:
            success, _, error = self.run_command(f"git checkout -b {branch_name}")
        
        if success:
            print(f"✅ Created and switched to branch: {branch_name}")
            return True
        else:
            print(f"❌ Failed to create branch: {error}")
            return False

    def switch_branch(self, branch_name):
        """Switch to existing branch."""
        success, _, error = self.run_command(f"git checkout {branch_name}")
        if success:
            print(f"✅ Switched to branch: {branch_name}")
            return True
        else:
            print(f"❌ Failed to switch branch: {error}")
            return False

    def setup_branch(self, env_key):
        """Setup the correct branch for the environment."""
        target_branch = self.environments[env_key]['branch']
        current_branch = self.get_current_branch()
        
        print(f"🌿 Setting up branch for {env_key} environment...")
        
        if current_branch == target_branch:
            print(f"✅ Already on target branch: {target_branch}")
            return True
        
        if self.branch_exists(target_branch):
            return self.switch_branch(target_branch)
        else:
            print(f"🆕 Branch {target_branch} doesn't exist")
            upstream = self.environments[env_key].get('upstream')
            if upstream and self.branch_exists(upstream):
                print(f"Creating {target_branch} from {upstream}")
                return self.create_branch(target_branch, upstream)
            else:
                print(f"Creating {target_branch} from current branch")
                return self.create_branch(target_branch)

    def analyze_changes(self):
        """Analyze git changes and return detailed information."""
        success, stdout, _ = self.run_command("git status --porcelain", check=False)
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
                if 'M' in status:
                    changes['modified'].append(filename)
                elif 'A' in status:
                    changes['added'].append(filename)
                elif 'D' in status:
                    changes['deleted'].append(filename)
        
        return changes, total_files, stdout

    def display_changes(self, changes, total_files):
        """Display detailed change information."""
        if total_files == 0:
            print("ℹ️  No changes to commit")
            return
        
        print(f"\n📊 Changes Summary ({total_files} files):")
        print("=" * 50)
        
        if changes['untracked']:
            print(f"🆕 New files ({len(changes['untracked'])}):")
            for file in changes['untracked'][:10]:
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

    def generate_commit_message(self, changes, total_files, env_key, is_initial=False):
        """Generate environment-aware commit message."""
        if is_initial:
            return f"[{env_key.upper()}] initial commit"
        
        if total_files == 0:
            return f"[{env_key.upper()}] no changes"
        
        parts = []
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
            return f"[{env_key.upper()}] misc changes"
        
        if len(parts) == 1:
            base_msg = parts[0]
        elif len(parts) == 2:
            base_msg = f"{parts[0]} and {parts[1]}"
        else:
            base_msg = f"{', '.join(parts[:-1])}, and {parts[-1]}"
        
        return f"[{env_key.upper()}] {base_msg}"

    def has_remote_origin(self):
        """Check if remote origin is configured."""
        success, stdout, _ = self.run_command("git remote -v", check=False)
        return success and "origin" in stdout

    def safe_git_add(self):
        """Safely add files to git, handling problematic files."""
        print("🔄 Adding files to git...")
        
        success, _, error = self.run_command("git add .", check=False)
        if success:
            print("✅ Files added to staging")
            return True
        
        print(f"⚠️  Standard git add failed: {error}")
        print("🔄 Trying to add files individually...")
        
        try:
            success, stdout, _ = self.run_command("git status --porcelain", check=False)
            if not success:
                print("❌ Could not get git status")
                return False
            
            files_added = 0
            for line in stdout.split('\n'):
                if not line.strip():
                    continue
                    
                status = line[:2]
                filename = line[3:].strip()
                
                if status.strip() in ['D', 'R']:
                    continue
                    
                success, _, file_error = self.run_command(f'git add "{filename}"', check=False)
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

    def pull_latest_changes(self, branch_name):
        """Pull latest changes from remote branch."""
        if not self.has_remote_origin():
            print("ℹ️  No remote origin configured, skipping pull")
            return True
        
        print(f"🔽 Pulling latest changes from origin/{branch_name}...")
        success, stdout, error = self.run_command(f"git pull origin {branch_name}", check=False)
        
        if success:
            if "Already up to date" in stdout or "Already up-to-date" in stdout:
                print("✅ Repository is already up to date")
            elif "Fast-forward" in stdout:
                print("✅ Fast-forwarded to latest changes")
                lines = stdout.split('\n')
                for line in lines:
                    if 'file' in line and ('changed' in line or 'insertion' in line or 'deletion' in line):
                        print(f"   📥 {line.strip()}")
            else:
                print("✅ Successfully pulled changes")
            return True
        else:
            if "merge conflict" in error.lower() or "conflict" in error.lower():
                print(f"⚠️  Merge conflicts detected:")
                print(f"   🔍 {error}")
                print("   ⚡ Please resolve conflicts manually and run the script again")
                return False
            elif "diverged" in error.lower():
                print(f"⚠️  Branches have diverged:")
                print(f"   🔍 {error}")
                response = input("Do you want to continue anyway? (y/N): ").strip().lower()
                return response == 'y'
            else:
                print(f"⚠️  Pull failed: {error}")
                response = input("Do you want to continue without pulling? (y/N): ").strip().lower()
                return response == 'y'

    def execute_workflow(self, env_key):
        """Execute the complete workflow for selected environment."""
        env_config = self.environments[env_key]
        
        print(f"\n🚀 Starting {env_key.upper()} Environment Workflow")
        print("=" * 60)
        
        # Setup branch
        if not self.setup_branch(env_key):
            return False
        
        # Pull latest changes
        if not self.pull_latest_changes(env_config['branch']):
            return False
        
        # Handle environment-specific workflows
        if env_key == 'dev':
            return self._handle_dev_workflow(env_config)
        elif env_key == 'staging':
            return self._handle_staging_workflow(env_config)
        elif env_key == 'prod':
            return self._handle_production_workflow(env_config)
        
        return True

    def _handle_dev_workflow(self, env_config):
        """Handle development environment workflow."""
        changes, total_files, _ = self.analyze_changes()
        if total_files == 0:
            print("ℹ️  No local changes to commit")
            return True
        
        self.display_changes(changes, total_files)
        
        if not self.safe_git_add():
            return False
        
        success, stdout, _ = self.run_command("git status --porcelain")
        if success and not stdout:
            print("ℹ️  No changes to commit after staging")
            return True
        
        commit_msg = self.generate_commit_message(changes, total_files, 'dev')
        
        success, _, error = self.run_command(f'git commit -m "{commit_msg}"')
        if not success:
            print(f"❌ Failed to commit: {error}")
            return False
        print(f"✅ Changes committed: '{commit_msg}'")
        
        if not self._push_changes(env_config['branch'], total_files):
            return False
        
        self._show_workflow_summary('dev', env_config['branch'], commit_msg, total_files)
        print(f"\n💡 Next Steps:")
        print(f"   1. Run script with 'staging' to promote to client review")
        print(f"   2. Or continue developing and run 'dev' again")
        
        return True

    def _handle_staging_workflow(self, env_config):
        """Handle staging environment workflow - merge from development."""
        upstream_branch = env_config.get('upstream', 'development')
        
        print(f"🔄 Preparing to merge {upstream_branch} → staging for client review...")
        
        if not self.branch_exists(upstream_branch):
            print(f"❌ Upstream branch '{upstream_branch}' doesn't exist")
            print(f"   Run the script with 'dev' first to create development branch")
            return False
        
        print(f"\n🎭 STAGING DEPLOYMENT")
        print(f"   This will merge '{upstream_branch}' into 'staging' for client review")
        response = input(f"   Merge {upstream_branch} → staging? (y/N): ").strip().lower()
        
        if response != 'y':
            print("⏹️  Staging deployment cancelled")
            return False
        
        print(f"🔄 Merging {upstream_branch} into staging...")
        success, stdout, error = self.run_command(f"git merge {upstream_branch}", check=False)
        
        if not success:
            if "merge conflict" in error.lower():
                print(f"⚠️  Merge conflicts detected: {error}")
                print("   ⚡ Please resolve conflicts manually and run the script again")
                return False
            else:
                print(f"❌ Merge failed: {error}")
                return False
        
        if "Already up to date" in stdout:
            print("ℹ️  Staging already up to date with development")
            merge_msg = "staging up to date"
            total_files = 0
        else:
            print("✅ Successfully merged development into staging")
            merge_msg = f"merge {upstream_branch} into staging"
            total_files = len([line for line in stdout.split('\n') if 'file' in line.lower()]) or 1
        
        if not self._push_changes(env_config['branch'], total_files):
            return False
        
        self._show_workflow_summary('staging', env_config['branch'], merge_msg, total_files)
        print(f"\n💡 Next Steps:")
        print(f"   1. Share staging environment with client for review")
        print(f"   2. After client approval, run script with 'prod' to go live")
        
        return True

    def _handle_production_workflow(self, env_config):
        """Handle production environment workflow - merge from staging."""
        upstream_branch = env_config.get('upstream', 'staging')
        
        print(f"🚀 Preparing to deploy to PRODUCTION...")
        
        if not self.branch_exists(upstream_branch):
            print(f"❌ Upstream branch '{upstream_branch}' doesn't exist")
            print(f"   Run the script with 'staging' first to prepare for production")
            return False
        
        print(f"\n🚨 PRODUCTION DEPLOYMENT WARNING")
        print(f"   This will merge '{upstream_branch}' into 'master' and go LIVE")
        print(f"   This affects real users and cannot be easily undone")
        print(f"   Make sure client has approved the staging environment")
        
        response = input(f"\n   Are you sure you want to deploy to production? (yes/no): ").strip().lower()
        
        if response not in ['yes', 'y']:
            print("⏹️  Production deployment cancelled - wise choice to double-check!")
            return False
        
        response2 = input(f"   Final confirmation - Deploy to LIVE production? (YES/no): ").strip()
        
        if response2 != 'YES':
            print("⏹️  Production deployment cancelled")
            return False
        
        print(f"🔄 Merging {upstream_branch} into production (master)...")
        success, stdout, error = self.run_command(f"git merge {upstream_branch}", check=False)
        
        if not success:
            if "merge conflict" in error.lower():
                print(f"⚠️  Merge conflicts detected: {error}")
                print("   ⚡ Please resolve conflicts manually and run the script again")
                return False
            else:
                print(f"❌ Merge failed: {error}")
                return False
        
        if "Already up to date" in stdout:
            print("ℹ️  Production already up to date with staging")
            merge_msg = "production up to date"
            total_files = 0
        else:
            print("✅ Successfully merged staging into production")
            merge_msg = f"deploy {upstream_branch} to production"
            total_files = len([line for line in stdout.split('\n') if 'file' in line.lower()]) or 1
        
        if not self._push_changes(env_config['branch'], total_files):
            return False
        
        self._show_workflow_summary('production', env_config['branch'], merge_msg, total_files)
        print(f"\n🎉 PRODUCTION DEPLOYMENT SUCCESSFUL!")
        print(f"   🌍 Changes are now LIVE for all users")
        print(f"   📈 Monitor your application for any issues")
        
        return True

    def _push_changes(self, branch_name, total_files):
        """Push changes to remote branch."""
        if not self.has_remote_origin():
            print("⚠️  No remote origin configured. Skipping push.")
            return True
        
        print(f"🚀 Pushing changes to origin/{branch_name}...")
        
        success, _, _ = self.run_command(f"git ls-remote --exit-code --heads origin {branch_name}", check=False)
        if not success:
            print(f"   Creating new remote branch: {branch_name}")
            success, _, error = self.run_command(f"git push -u origin {branch_name}")
        else:
            success, _, error = self.run_command(f"git push origin {branch_name}")
        
        if not success:
            print(f"❌ Failed to push: {error}")
            return False
        
        print(f"✅ Successfully pushed to GitHub (origin/{branch_name})")
        return True

    def _show_workflow_summary(self, env_key, branch_name, action, total_files):
        """Show workflow summary."""
        print(f"\n🎯 {env_key.upper()} Environment Updated:")
        print(f"   📁 Branch: {branch_name}")
        print(f"   💾 Action: {action}")
        print(f"   📊 Files: {total_files} affected")
        print(f"   🌐 Environment: {self.environments[env_key]['description']}")

    def main(self):
        """Main workflow function."""
        print("🌟 Professional Git Workflow Automation")
        print("=" * 50)
        print("Supports Development → Staging → Production workflow")
        print()
        
        if not self.create_gitignore():
            sys.exit(1)
        
        env_key = self.select_environment()
        
        if self.execute_workflow(env_key):
            print(f"\n🎉 {env_key.upper()} workflow completed successfully!")
            self.save_config()
        else:
            print(f"\n💥 {env_key.upper()} workflow failed!")
            sys.exit(1)


if __name__ == "__main__":
    try:
        workflow = GitWorkflow()
        workflow.main()
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
