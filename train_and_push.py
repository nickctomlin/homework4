"""
Script to train both VLM and CLIP models, save to Google Drive, then git push.

Usage:
    python train_and_push.py
    python train_and_push.py --skip_vlm        # Skip VLM training
    python train_and_push.py --skip_clip       # Skip CLIP training
    python train_and_push.py --skip_push       # Skip git push
    python train_and_push.py --message "Custom commit message"
    python train_and_push.py --drive_path "/content/drive/MyDrive/HW4_Models"
"""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def mount_google_drive():
    """Mount Google Drive in Colab."""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted successfully!")
        return True
    except ImportError:
        print("⚠️  Not running in Google Colab, skipping Drive mount")
        return False
    except Exception as e:
        print(f"⚠️  Failed to mount Google Drive: {e}")
        return False


def save_to_drive(local_path: Path, drive_path: Path, name: str):
    """Copy checkpoint files to Google Drive."""
    if not drive_path.parent.exists():
        print(f"⚠️  Google Drive path does not exist: {drive_path.parent}")
        return False
    
    target_path = drive_path / name
    target_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Saving {name} to Google Drive...")
    print(f"   From: {local_path}")
    print(f"   To:   {target_path}")
    
    # Copy all checkpoint files
    files_copied = 0
    for file in local_path.iterdir():
        if file.is_file():
            shutil.copy2(file, target_path / file.name)
            print(f"   ✓ Copied: {file.name}")
            files_copied += 1
    
    print(f"   ✅ Saved {files_copied} files to Google Drive")
    return True


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
    parser = argparse.ArgumentParser(description="Train VLM and CLIP, save to Drive, then push to git")
    parser.add_argument("--skip_vlm", action="store_true", help="Skip VLM training")
    parser.add_argument("--skip_clip", action="store_true", help="Skip CLIP training")
    parser.add_argument("--skip_push", action="store_true", help="Skip git push")
    parser.add_argument("--skip_drive", action="store_true", help="Skip saving to Google Drive")
    parser.add_argument("--message", "-m", type=str, default=None, help="Custom commit message")
    parser.add_argument("--drive_path", type=str, 
                        default="/content/drive/MyDrive/HW4_Models",
                        help="Google Drive path to save models")
    args = parser.parse_args()

    # Paths
    homework_dir = Path(__file__).parent / "homework"
    vlm_checkpoint = homework_dir / "vlm_sft"
    clip_checkpoint = homework_dir / "clip"
    drive_path = Path(args.drive_path)

    success = True
    
    # Step 0: Mount Google Drive (if in Colab)
    if not args.skip_drive:
        print("\n" + "💾" * 20)
        print("  MOUNTING GOOGLE DRIVE")
        print("💾" * 20)
        drive_mounted = mount_google_drive()
        
        if drive_mounted:
            # Create drive directory
            drive_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Models will be saved to: {drive_path}")
    else:
        drive_mounted = False
        print("\n⏭️  Skipping Google Drive (--skip_drive)")

    # Step 1: Train VLM
    if not args.skip_vlm:
        print("\n" + "🚀" * 20)
        print("  STARTING VLM TRAINING")
        print("🚀" * 20)
        
        if not run_command("python -m homework.finetune train", "VLM Training"):
            print("⚠️  VLM training failed, continuing anyway...")
            success = False
        else:
            # Save VLM to Google Drive
            if drive_mounted and vlm_checkpoint.exists():
                save_to_drive(vlm_checkpoint, drive_path, "vlm_sft")
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
            # Save CLIP to Google Drive
            if drive_mounted and clip_checkpoint.exists():
                save_to_drive(clip_checkpoint, drive_path, "clip")
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
    
    print("\n📁 LOCAL Checkpoints:")
    print(f"  - {vlm_checkpoint}/  (VLM model)")
    print(f"  - {clip_checkpoint}/  (CLIP model)")
    
    if drive_mounted:
        print(f"\n☁️  GOOGLE DRIVE Backups:")
        print(f"  - {drive_path}/vlm_sft/  (VLM model)")
        print(f"  - {drive_path}/clip/     (CLIP model)")
    
    print("\n🧪 To test the models:")
    print("  python -m homework.finetune test vlm_sft")
    print("  python -m homework.clip test clip")
    
    if drive_mounted:
        print("\n💡 To restore from Google Drive if runtime resets:")
        print(f"  cp -r {drive_path}/vlm_sft/* homework/vlm_sft/")
        print(f"  cp -r {drive_path}/clip/* homework/clip/")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
