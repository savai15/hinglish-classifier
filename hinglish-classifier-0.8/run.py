"""
Hinglish Complaint Classifier - One-Click Launcher
Starts both FastAPI backend and Vite frontend.
"""
import subprocess
import sys
import os
import time
import signal

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

def main():
    print("=" * 50)
    print("  Hinglish Complaint Classifier")
    print("  Starting backend + frontend...")
    print("=" * 50)

    procs = []

    # Start FastAPI backend
    print("\n[1/2] Starting FastAPI backend on port 8000...")
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--reload", "--port", "8000"],
        cwd=PROJECT_ROOT,
    )
    procs.append(("Backend", backend))

    # Wait for backend to start
    time.sleep(2)

    # Start Vite frontend
    print("[2/2] Starting Vite frontend on port 5173...")
    frontend = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=FRONTEND_DIR,
    )
    procs.append(("Frontend", frontend))

    print("\n" + "=" * 50)
    print("  Both servers running!")
    print("  Frontend: http://localhost:5173")
    print("  Backend:  http://localhost:8000")
    print("  API Docs: http://localhost:8000/docs")
    print("=" * 50)
    print("\n  Press Ctrl+C to stop both servers.\n")

    try:
        backend.wait()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        for name, p in procs:
            p.terminate()
            print(f"  {name} stopped.")
        print("Done!")

if __name__ == "__main__":
    main()
