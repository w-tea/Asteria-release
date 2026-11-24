"""Launcher-owned LAN entry; no independent generation engine or data root."""
import argparse
import asyncio
from pathlib import Path

from app.core.config import AppConfig
from app.services.lan_runtime import serve_gateway


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, default=3001)
    parser.add_argument('--cert', type=Path)
    parser.add_argument('--key', type=Path)
    args = parser.parse_args()
    asyncio.run(serve_gateway(AppConfig(), host=args.host, port=args.port, cert=args.cert, key=args.key))


if __name__ == '__main__':
    main()
