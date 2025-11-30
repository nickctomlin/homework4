"""
Script to train both VLM and CLIP models, then git add, commit, and push.

Usage:
    python train_and_push.py
    python train_and_push.py --skip_vlm        # Skip VLM training
    python train_and_push.py --skip_clip       # Skip CLIP training
    python train_and_push.py --skip_push       # Skip git push
    python train_and_push.py --message "Custom commit message"
"""

import argparse
import subprocess
import sys
from datetime import datetime


def run_command(cmd, description):
    """Run a command and print output."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ FAILED: {description}")
        return False
    
    print(f"\n✅ COMPLETED: {description}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Train VLM and CLIP, then push to git")
    parser.add_argument("--skip_vlm", action="store_true", help="Skip VLM training")
    parser.add_argument("--skip_clip", action="store_true", help="Skip CLIP training")
    parser.add_argument("--skip_push", action="store_true", help="Skip git push")
    parser.add_argument("--message", "-m", type=str, default=None, help="Custom commit message")
    args = parser.parse_args()

    success = True
    
    # Step 1: Train VLM
    if not args.skip_vlm:
        print("\n" + "🚀" * 20)
        print("  STARTING VLM TRAINING")
        print("🚀" * 20)
        
        if not run_command("python -m homework.finetune train", "VLM Training"):
            print("⚠️  VLM training failed, continuing anyway...")
            success = False
    else:
        print("\n⏭️  Skipping VLM training (--skip_vlm)")

    # Step 2: Train CLIP
    if not args.skip_clip:
        print("\n" + "🚀" * 20)
        print("  STARTING CLIP TRAINING")
        print("🚀" * 20)
        
        if not run_command("python -m homework.clip train", "CLIP Training"):
            print("⚠️  CLIP training failed, continuing anyway...")
            success = False
    else:
        print("\n⏭️  Skipping CLIP training (--skip_clip)")

    # Step 3: Git operations
    print("\n" + "📦" * 20)
    print("  GIT OPERATIONS")
    print("📦" * 20)

    # Git add
    run_command("git add -A", "Git add all changes")

    # Git commit
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if args.message:
        commit_msg = args.message
    else:
        commit_msg = f"Training completed at {timestamp}"
    
    run_command(f'git commit -m "{commit_msg}"', f"Git commit: {commit_msg}")

    # Git push
    if not args.skip_push:
        run_command("git push origin master", "Git push to origin/master")
    else:
        print("\n⏭️  Skipping git push (--skip_push)")

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    
    if success:
        print("✅ All training completed successfully!")
    else:
        print("⚠️  Some training steps had issues, check logs above.")
    
    print("\nCheckpoints saved to:")
    print("  - homework/vlm_sft/  (VLM model)")
    print("  - homework/clip/     (CLIP model)")
    print("\nTo test the models:")
    print("  python -m homework.finetune test vlm_sft")
    print("  python -m homework.clip test clip")
    print("=" * 60)


if __name__ == "__main__":
    main()

