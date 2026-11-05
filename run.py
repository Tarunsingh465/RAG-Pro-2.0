import socket
import subprocess
import sys
import time
import uvicorn


def is_port_busy(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def release_port(port: int) -> bool:
    """Attempts to release a lingering process on Windows."""
    try:
        cmd = f'powershell -Command "Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }}"'
        subprocess.run(cmd, shell=True, capture_output=True, timeout=5)
        time.sleep(1)
        return not is_port_busy(port)
    except Exception:
        return False


if __name__ == "__main__":
    PORT = 8000
    HOST = "127.0.0.1"

    if is_port_busy(PORT, HOST):
        print(f"⚠️  Port {PORT} is currently busy. Attempting to free it...")
        if release_port(PORT):
            print(f"✅ Successfully released port {PORT}!")
        else:
            print(f"❌ Could not automatically release port {PORT}.")
            print("   Please terminate the process manually or use another port.")
            sys.exit(1)

    print(f"🚀 Starting RAG-Pro API on http://{HOST}:{PORT} ...")
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=True,
    )
