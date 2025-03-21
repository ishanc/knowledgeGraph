#!/usr/bin/env python3
"""
A watchdog script to monitor the Knowledge Graph application and restart it when needed.
This can be used in environments where a full process manager is not available.

Run this script in the background:
    nohup python watchdog.py > watchdog.log 2>&1 &
"""

import os
import sys
import time
import subprocess
import logging
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('watchdog.log')
    ]
)

logger = logging.getLogger(__name__)

# Configuration
APP_DIR = os.path.dirname(os.path.abspath(__file__))
HEALTH_CHECK_URL = "http://localhost:8080/health"
RESTART_TRIGGER_FILE = os.path.join(APP_DIR, ".restart_trigger")
HEALTH_STATUS_FILE = os.path.join(APP_DIR, ".health_status")
CHECK_INTERVAL = 30  # seconds

def is_process_running():
    """Check if the Flask application is running and responsive."""
    try:
        response = requests.get(HEALTH_CHECK_URL, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

def should_restart():
    """Check if a restart has been requested through the trigger file."""
    if not os.path.exists(RESTART_TRIGGER_FILE):
        return False
    
    # Check if the trigger file was modified recently (last 5 minutes)
    modified_time = os.path.getmtime(RESTART_TRIGGER_FILE)
    now = time.time()
    
    return (now - modified_time) < 300  # 5 minutes

def check_health_status():
    """Check the health status file."""
    if not os.path.exists(HEALTH_STATUS_FILE):
        return "UNKNOWN"
        
    try:
        with open(HEALTH_STATUS_FILE, 'r') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error reading health status: {e}")
        return "ERROR"

def restart_app():
    """Restart the Flask application."""
    logger.info("Restarting application...")
    
    # Kill any existing process
    try:
        subprocess.run(["pkill", "-f", "python app.py"])
        # Wait for the process to terminate
        time.sleep(2)
    except Exception as e:
        logger.error(f"Error killing process: {e}")
    
    # Start the app in the background
    logger.info("Starting application...")
    try:
        subprocess.Popen(
            ["python", "app.py"],
            cwd=APP_DIR,
            stdout=open("app.log", "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
        
        # Update restart trigger file
        with open(RESTART_TRIGGER_FILE, 'w') as f:
            f.write(datetime.now().isoformat())
            
        # Wait for app to start
        for _ in range(10):
            if is_process_running():
                logger.info("Application started successfully")
                return True
            time.sleep(3)
        
        logger.error("Application failed to start in time")
        return False
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        return False

def main():
    """Main watchdog loop."""
    logger.info(f"Starting watchdog for Knowledge Graph application in {APP_DIR}")
    
    # Create trigger file if it doesn't exist
    if not os.path.exists(RESTART_TRIGGER_FILE):
        with open(RESTART_TRIGGER_FILE, 'w') as f:
            f.write(datetime.now().isoformat())
    
    # If app is not running on startup, start it
    if not is_process_running():
        logger.info("Application not running on watchdog startup")
        restart_app()
    
    # Main monitoring loop
    while True:
        try:
            health_status = check_health_status()
            
            if should_restart() or health_status == "RESTARTING":
                logger.info(f"Restart triggered (status: {health_status})")
                restart_app()
            elif not is_process_running():
                logger.warning("Application is not responsive, restarting")
                restart_app()
            else:
                logger.debug("Application is running normally")
            
        except Exception as e:
            logger.error(f"Error in watchdog: {e}")
        
        # Wait for next check
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
