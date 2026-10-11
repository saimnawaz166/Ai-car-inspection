#!/usr/bin/env python
"""Django ka command-line tool (runserver, migrate, etc.)."""
import os
import sys


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Django install nahi hai. Pehle chalayein: pip install -r requirements.txt"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
