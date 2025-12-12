# FootballQuantumLLM/logger.py
import logging
import sys
from pathlib import Path
import datetime
import zipfile

def manage_log_retention(log_dir: Path, current_log_path: Path):
    """
    Retention Policy:
    1. Keep the last 5 logs (files/days) in the main folder.
    2. For older logs, keep only the first log of each month and MOVE it to 'archive/' folder (COMPRESSED).
    3. Delete the rest.
    """
    # Create archive directory
    archive_dir = log_dir / "archive"
    archive_dir.mkdir(exist_ok=True)
    
    # List all log files matching the pattern in the MAIN directory
    all_logs = sorted(list(log_dir.glob("football_quantum_*.log")))
    
    # Remove current log from candidacy
    if current_log_path in all_logs:
        all_logs.remove(current_log_path)
        
    if not all_logs:
        return

    # 1. Keep last 5 in main dir
    keep_recent = all_logs[-5:]
    candidates_for_archive = all_logs[:-5]
    
    # 2. Process candidates: Move 1 per month to archive, delete others
    monthly_keepers = {} # Key: "YYYY-MM" -> Path
    
    for log in candidates_for_archive:
        try:
            date_part = log.name.split('_')[2] 
            date_str = date_part.replace(".log", "") 
            year_month = date_str[:7] 
            
            if year_month not in monthly_keepers:
                monthly_keepers[year_month] = log
        except IndexError:
            continue
            
    # Move keepers to archive (COMPRESS)
    keepers = set(monthly_keepers.values())
    archived_count = 0
    deleted_count = 0
    
    for log in candidates_for_archive:
        if log in keepers:
            # Compress and Move to archive
            try:
                zip_name = log.name.replace(".log", ".zip")
                dest_zip = archive_dir / zip_name
                
                if not dest_zip.exists():
                    with zipfile.ZipFile(dest_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                        zf.write(log, arcname=log.name)
                    # After compression, remove original
                    log.unlink()
                else:
                    # If already exists in archive, just delete source
                    log.unlink()
                archived_count += 1
            except Exception as e:
                print(f"Error archiving log {log}: {e}")
        else:
            # Delete
            try:
                log.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting old log {log}: {e}")
            
    if deleted_count > 0 or archived_count > 0:
        print(f"[Logger] Archived {archived_count} (zipped), Deleted {deleted_count} logs.")

def setup_logger(name: str = "FootballQuantumLLM") -> logging.Logger:
    """
    Sets up a logger with both File and Stream (Console) handlers.
    Logs are saved in the 'logs' directory with a daily timestamp (one file per day).
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Prevent adding handlers multiple times
    if logger.hasHandlers():
        return logger
        
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File Handler
    try:
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create DAILY filename: football_quantum_YYYY-MM-DD.log
        now_str = datetime.datetime.now().strftime("%Y-%m-%d")
        log_file = f"football_quantum_{now_str}.log"
        log_path = log_dir / log_file
        
        # Run retention policy BEFORE adding new handler
        # Note: If running multiple times a day, current_log_path exists and won't be deleted.
        manage_log_retention(log_dir, log_path)
        
        fh = logging.FileHandler(str(log_path), encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception as e:
        print(f"Failed to setup file handler: {e}")
        
    return logger
