# Utilities

This folder contains utility tools and legacy interfaces for OpenAudit.

## Tools

### Data Viewing Utilities
- **`quick_view.py`** - Command-line tool for quick viewing of experiment results
  ```bash
  python3 utils/quick_view.py
  ```

### Legacy Interfaces
- **`web_interface.py`** - Original historical analysis web interface (pre-unified)
  ```bash
  python3 utils/web_interface.py
  # Visit: http://localhost:5001
  ```

## Notes

- **`quick_view.py`** is useful for command-line data exploration
- **`web_interface.py`** is the original interface, now superseded by the main unified interface
- These tools import from the `core/` modules for consistency

## Migration Note

Users should prefer the main `openaudit_interface.py` over these legacy tools for most use cases. 