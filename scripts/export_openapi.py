"""Write the API's OpenAPI schema to frontend/openapi.json (the frontend's types are generated from it).

Usage:  python scripts/export_openapi.py && (cd frontend && npm run api:types)
"""

from __future__ import annotations

import json

from flowcast import paths
from flowcast.api.app import create_app

OUT = paths.PROJECT_ROOT / "frontend" / "openapi.json"


def main() -> None:
    OUT.write_text(json.dumps(create_app().openapi(), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
