# RAG Application - AI Agent Guide

## Architecture Overview

This is a **RAG (Retrieval-Augmented Generation)** CLI application with a service-oriented architecture. The codebase is split into two main parts:

- **`Common/`** - Shared business logic and services (singleton pattern, testable)
- **`rag_cli/`** - Click-based CLI interface that consumes Common services

Key principle: **CLI commands are thin wrappers** around Common services. Never put business logic in CLI command files.

## Core Patterns & Conventions

### 1. Singleton Services with Module-Level Access

All services use **lazy singleton pattern** with module-level convenience access:

```python
# From Common/services/config_service.py
class ConfigService:
    _instance = None
    def __new__(cls): ...  # Singleton

# Global lazy wrapper for convenience
config_service = _ConfigServiceLazy()

# Usage anywhere:
from Common.services.config_service import config_service
db_url = config_service.get_database_url()
```

**When writing new services**: Follow this exact pattern (see `config_service.py`, `neon_service.py`, `huggingface_service.py`).

### 2. Configuration Management

Configuration is **centralized** in `config_service.py` using Pydantic Settings:
- Loads from `.env` file (use `.env.example` as template)
- All settings have **validators** (see `AppSettings` validators for DB URLs, log levels, etc.)
- Access via: `config_service.settings.{property}` or helper methods like `config_service.get_youtube_config()`

**When adding new config**: Add to `AppSettings` with proper validation, default value, and type hints.

### 3. Path Management

Critical: The project adds `Common/` to `sys.path` in entry points:

```python
# From rag_cli.py and rag_cli/cli.py
sys.path.insert(0, str(Path(__file__).parent))
```

**Always import Common services as**: `from Common.services.X import Y` (not relative imports).

### 4. CLI Command Structure

Commands follow a **registration pattern**:

```python
# Two patterns in use:
# 1. Function-based registration (youtube.py, config.py)
def register_youtube_commands(cli):
    youtube = click.Group('youtube', help='...')
    cli.add_command(youtube)
    @youtube.command()
    def test(): ...

# 2. Group-based (neon.py, huggingface.py)
@click.group()
def neon():
    """Neon database commands."""
    pass

# Registered in rag_cli/cli.py:
cli.add_command(neon.neon)
```

**Commands should**:
- Check service availability with friendly error messages
- Support `--json` flag for machine-readable output
- Use emoji indicators (✅ ❌ 🔄 📝 etc.) for user feedback
- Print errors to stderr with troubleshooting tips

## Service Integration Points

### YouTube Service
- Uses YouTube Data API v3 (requires `YOUTUBE_API_KEY`)
- Retry strategy built-in (3 retries for 429, 500+ errors)
- Helper functions: `search_youtube_videos()`, `get_youtube_video_info()`

### Transcription Service
- **Priority strategy**: YouTube Transcript API (fast) → OpenAI Whisper (fallback)
- Lazy-loads Whisper model only when needed
- Supports multiple languages with auto-detection

### Neon Database Service
- PostgreSQL connection pooling (min=1, max=10)
- Context managers for connection safety: `with neon_service.get_cursor(): ...`
- Always requires SSL (`sslmode=require`)
- Uses `RealDictCursor` by default for dict-like results

### HuggingFace Service
- Singleton with module-level `huggingface_service` instance
- Methods: `test_connection()`, `generate_text()`, `get_embeddings()`, `list_models()`
- Optional auth via `HUGGINGFACE_TOKEN`

## Development Workflows

### Running the CLI
```bash
# From project root
python3 rag_cli.py [command]

# Examples:
python3 rag_cli.py config                        # Show config
python3 rag_cli.py youtube test --query "python" # Test YouTube API
python3 rag_cli.py neon backup --data > backup.sql
python3 rag_cli.py huggingface test
```

### Testing
Tests are in `Common/tests/` using pytest:

```bash
# Run all tests
pytest Common/tests/

# With coverage
pytest --cov=Common Common/tests/

# Specific service
pytest Common/tests/test_config_service.py -v
```

**Test patterns to follow**:
- Reset singletons in `setup`/`teardown`: `ConfigService._reset_instance()`
- Use `@patch` for external dependencies (API calls, DB connections)
- Test validation logic explicitly (see `test_log_level_validation`, `test_database_url_validation`)

### Adding a New Service

1. Create `Common/services/new_service.py`:
   - Singleton pattern with `_instance` and `__new__`
   - Config access via `config_service.get_X_config()`
   - Module-level instance: `new_service = NewService()`
   - Add `_reset_instance()` classmethod for tests

2. Add settings to `Common/services/config_service.py`:
   - New fields in `AppSettings` with validators
   - Helper method like `get_new_config()`

3. Create `Common/tests/test_new_service.py`:
   - Test singleton pattern
   - Test with/without configuration
   - Mock external dependencies

4. Add CLI commands in `rag_cli/commands/new.py`:
   - Follow group pattern
   - Support `--json` flag
   - Register in `rag_cli/cli.py`

## Project-Specific Quirks

1. **Mixed documentation**: README is in Polish (`Common/README.md`), code is in English
2. **Entry point**: `rag_cli.py` is the main entry (not `rag_cli/cli.py` directly)
3. **No async yet**: Services use sync code despite README mentioning async aspirations
4. **Connection pooling**: Neon service uses `SimpleConnectionPool`, not async pool
5. **Type hints required**: Full annotations everywhere per README guidelines
6. **Python 3.11+**: Can use modern features (pattern matching, enhanced type hints)

## Common Gotchas

- Don't forget `sys.path.insert(0, ...)` in new entry points
- CLI commands check `is_available()` before using services
- Singleton services need `_reset_instance()` for test isolation
- Database connection strings must start with `postgresql://` or `sqlite://`
- Neon requires SSL; connection strings need `?sslmode=require`
- Secret key must be 32+ chars (auto-generates if missing)

## File Organization Rules

- Business logic → `Common/services/`
- CLI interface → `rag_cli/commands/`
- Tests → `Common/tests/test_*.py`
- Config → `.env` (never committed, use `.env.example`)
- No models/ or utils/ directories exist yet (mentioned in README but not implemented)
