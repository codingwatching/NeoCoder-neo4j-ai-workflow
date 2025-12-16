#!/usr/bin/env python3
"""
Extract incarnation types directly from source files in the incarnations directory.
"""

import os
import re


def extract_incarnation_types() -> None:
    """Scan the incarnations directory and list available types."""
    # correct path determination
    current_dir = os.path.dirname(os.path.abspath(__file__))
    incarnations_dir = os.path.join(current_dir, "incarnations")

    if not os.path.exists(incarnations_dir):
        print(f"Error: Directory not found: {incarnations_dir}")
        return

    print(f"Scanning directory: {incarnations_dir}")
    print("\nIncarnation Types Found:")
    print("------------------------")

    found_count = 0

    # Scan for *_incarnation.py files
    for entry in sorted(os.listdir(incarnations_dir)):
        if entry.startswith("__") or not entry.endswith(".py"):
            continue

        # Skip base modules
        if entry in ("base_incarnation.py", "polymorphic_adapter.py"):
            continue

        # Check if it follows the pattern
        if entry.endswith("_incarnation.py"):
            # Extract type from filename
            type_name = entry.replace("_incarnation.py", "")

            # Read file to find description
            file_path = os.path.join(incarnations_dir, entry)
            description = "No description found"

            try:
                with open(file_path, "r") as f:
                    content = f.read()
                    # Try to find description class attribute
                    desc_match = re.search(
                        r'description\s*=\s*["\']([^"\']+)["\']', content
                    )
                    if desc_match:
                        description = desc_match.group(1)
            except Exception as e:
                description = f"Error reading file: {e}"

            print(f'{type_name.upper()} = "{type_name}"  # {description}')
            found_count += 1

    print(f"\nTotal types found: {found_count}")


def main() -> None:
    extract_incarnation_types()


if __name__ == "__main__":
    main()
