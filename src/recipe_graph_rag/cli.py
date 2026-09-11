"""Entry point for local configuration validation."""
import argparse
import json

from .config import load_settings


def main():
    parser = argparse.ArgumentParser(description="Recipe Graph RAG")
    parser.add_argument("command", choices=["check-config"])
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()
    try:
        settings = load_settings(args.env_file)
    except ValueError as exc:
        parser.exit(2, f"Configuration error: {exc}\n")
    print(json.dumps(settings.safe_summary(), indent=2, ensure_ascii=False))
    print("Configuration format OK. External services have NOT been checked.")


if __name__ == "__main__":
    main()
