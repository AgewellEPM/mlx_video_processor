#!/usr/bin/env python3
"""
Launch script for MLX Video Processor Web Interface.

Run this script to start the web server:
    python web/run_server.py

Then open http://localhost:8000 in your browser.
"""

import argparse
import os
import sys
import webbrowser
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="MLX Video Processor Web Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python web/run_server.py                    # Start on default port 8000
    python web/run_server.py --port 8080        # Start on port 8080
    python web/run_server.py --no-browser       # Don't open browser automatically
    python web/run_server.py --host 0.0.0.0     # Allow external connections
        """,
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1, use 0.0.0.0 for external access)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )

    args = parser.parse_args()

    # Print banner
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║         MLX Video Processor - Web Interface              ║
    ║                                                          ║
    ║         Apple ML Models on Apple Silicon                 ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    url = f"http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}"
    print(f"    Starting server at: {url}")
    print(f"    Host: {args.host}")
    print(f"    Port: {args.port}")
    print(f"    Auto-reload: {'enabled' if args.reload else 'disabled'}")
    print()
    print("    Available models:")
    print("      - FastViT T8 (Image Classification)")
    print("      - YOLOv3 (Object Detection)")
    print("      - DeepLabV3 (Semantic Segmentation)")
    print("      - Depth Anything V2 (Depth Estimation)")
    print("      - DETR ResNet50 (Detection Transformer)")
    print()
    print("    Press Ctrl+C to stop the server")
    print()

    # Open browser
    if not args.no_browser:
        print(f"    Opening browser at {url}...")
        webbrowser.open(url)

    # Import and run uvicorn
    try:
        import uvicorn
    except ImportError:
        print("\n    Error: uvicorn not installed.")
        print("    Please install it with: pip install uvicorn")
        sys.exit(1)

    # Change to web directory
    os.chdir(Path(__file__).parent)

    # Run server
    uvicorn.run(
        "backend:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
