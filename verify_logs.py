# verify_logs.py
import time
import zipfile
from pathlib import Path
from FootballQuantumLLM.logger import setup_logger

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def create_dummy_logs():
    print("Creating dummy logs with content...")
    # Clear existing logs/zips first for clean test
    for f in LOG_DIR.glob("football_quantum_*.log"):
        f.unlink()
    archive_dir = LOG_DIR / "archive"
    if archive_dir.exists():
        for f in archive_dir.glob("*.zip"):
            f.unlink()
        for f in archive_dir.glob("*.log"):
            f.unlink()
        
    # Helper to create file with content
    def touch_with_content(p):
        with open(p, 'w') as f:
            f.write("Log content for verification.\n")

    # Create logs for previous months (to be archived)
    # Aug: 2 logs (should keep 1)
    touch_with_content(LOG_DIR / "football_quantum_2023-08-01.log")
    touch_with_content(LOG_DIR / "football_quantum_2023-08-15.log")
    
    # Sep: 3 logs (should keep 1)
    touch_with_content(LOG_DIR / "football_quantum_2023-09-01.log")
    touch_with_content(LOG_DIR / "football_quantum_2023-09-10.log")
    touch_with_content(LOG_DIR / "football_quantum_2023-09-20.log")
    
    # Recent logs (should keep all 5)
    # Fake dates for last few days
    touch_with_content(LOG_DIR / "football_quantum_2025-12-05.log")
    touch_with_content(LOG_DIR / "football_quantum_2025-12-06.log")
    touch_with_content(LOG_DIR / "football_quantum_2025-12-07.log")
    touch_with_content(LOG_DIR / "football_quantum_2025-12-08.log")
    touch_with_content(LOG_DIR / "football_quantum_2025-12-09.log")
    
    print(f"Total logs in dir: {len(list(LOG_DIR.glob('football_quantum_*.log')))}")

def run_test():
    create_dummy_logs()
    
    print("\nRunning setup_logger() which should trigger cleanup and compression...")
    # Note: setup_logger will create a log for TODAY (2025-12-10 or actual date)
    logger = setup_logger()
    
    logs_main = sorted(list(LOG_DIR.glob("football_quantum_*.log")))
    archive_dir = LOG_DIR / "archive"
    
    # Check for ZIP files in archive
    logs_archive = sorted(list(archive_dir.glob("football_quantum_*.zip"))) if archive_dir.exists() else []

    print(f"\nMain logs ({len(logs_main)}):")
    for l in logs_main:
        print(l.name)
        
    print(f"\nArchive logs (ZIPs) ({len(logs_archive)}):")
    for l in logs_archive:
        print(l.name)
        
        # Verify ZIP content
        try:
            with zipfile.ZipFile(l, 'r') as zf:
                if zf.testzip() is not None:
                    print(f"[ERROR] Corrupted zip: {l}")
                else:
                    print(f"  [OK] Zip content verified: {zf.namelist()}")
        except Exception as e:
            print(f"[ERROR] Failed to open zip {l}: {e}")
    
    expected_main = 6
    expected_archive = 2
    
    if len(logs_main) == expected_main and len(logs_archive) == expected_archive:
        print("\nSUCCESS: Log retention and ZIP archiving verified.")
    else:
        print(f"\nFAILURE: Expected {expected_main} main + {expected_archive} ZIP archives. Found {len(logs_main)} + {len(logs_archive)}.")

if __name__ == "__main__":
    run_test()
