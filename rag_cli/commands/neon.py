"""
Neon database management commands for RAG CLI.

This module provides CLI commands for managing Neon PostgreSQL databases,
including backup, restore, and SQL execution capabilities.
"""

import sys
import click
from typing import Optional

from Common.services.neon_service import get_neon_service


@click.group()
def neon():
    """Neon database management commands."""
    pass


@neon.command()
@click.option('--data', is_flag=True, help='Include table data in backup')
def backup(data: bool):
    """Backup database schema (and optionally data) to stdout."""
    try:
        neon_service = get_neon_service()

        if not neon_service.is_available():
            click.echo("❌ Neon database service is not available. Check NEON_DB_URL configuration.", err=True)
            sys.exit(1)

        click.echo("🔄 Creating database backup...", err=True)

        # Create backup
        sql_backup = neon_service.backup_schema(include_data=data)

        if not sql_backup.strip():
            click.echo("⚠️  No tables found in database.", err=True)
            return

        # Output to stdout
        click.echo(sql_backup)

        click.echo(f"✅ Database backup completed ({'with data' if data else 'schema only'})", err=True)

    except Exception as e:
        click.echo(f"❌ Backup failed: {e}", err=True)
        sys.exit(1)


@neon.command()
@click.option('--input', '-i', help='Input file containing SQL commands (default: stdin)')
def execute(input: Optional[str]):
    """Execute SQL commands from stdin or file."""
    try:
        neon_service = get_neon_service()

        if not neon_service.is_available():
            click.echo("❌ Neon database service is not available. Check NEON_DB_URL configuration.", err=True)
            sys.exit(1)

        # Read SQL content
        if input:
            try:
                with open(input, 'r', encoding='utf-8') as f:
                    sql_content = f.read()
                click.echo(f"📖 Reading SQL from file: {input}", err=True)
            except FileNotFoundError:
                click.echo(f"❌ File not found: {input}", err=True)
                sys.exit(1)
            except Exception as e:
                click.echo(f"❌ Error reading file: {e}", err=True)
                sys.exit(1)
        else:
            click.echo("📖 Reading SQL from stdin... (Ctrl+D to finish)", err=True)
            sql_content = sys.stdin.read()
            click.echo("✅ SQL content read from stdin", err=True)

        if not sql_content.strip():
            click.echo("⚠️  No SQL content provided.", err=True)
            return

        click.echo("🔄 Executing SQL commands...", err=True)

        # Execute SQL
        results = neon_service.execute_sql_from_stdin(sql_content)

        # Report results
        click.echo(f"✅ Executed {results['commands_executed']} SQL commands", err=True)

        if results['errors']:
            click.echo(f"⚠️  {len(results['errors'])} errors occurred:", err=True)
            for error in results['errors']:
                click.echo(f"  - {error['error']}", err=True)
                if len(error['statement']) > 50:
                    click.echo(f"    Statement: {error['statement'][:50]}...", err=True)
                else:
                    click.echo(f"    Statement: {error['statement']}", err=True)

        if results['results']:
            total_rows = sum(r.get('rows_affected', 0) for r in results['results'] if 'rows_affected' in r)
            click.echo(f"📊 Total rows affected: {total_rows}", err=True)

    except Exception as e:
        click.echo(f"❌ SQL execution failed: {e}", err=True)
        sys.exit(1)


@neon.command()
def status():
    """Check Neon database connection status."""
    try:
        neon_service = get_neon_service()

        if not neon_service.is_available():
            click.echo("❌ Neon database service is not available")
            click.echo("   Make sure NEON_DB_URL is configured in your .env file")
            return

        click.echo("🔍 Checking database connection...", err=True)

        health = neon_service.health_check()

        if health['status'] == 'healthy':
            click.echo("✅ Database connection is healthy")
            click.echo(f"   Version: {health.get('version', 'unknown')}")
            click.echo(f"   Database: {health.get('database', 'unknown')}")
            click.echo(f"   User: {health.get('user', 'unknown')}")
        else:
            click.echo("❌ Database connection failed")
            click.echo(f"   Error: {health.get('error', 'unknown')}")

    except Exception as e:
        click.echo(f"❌ Status check failed: {e}")


@neon.command()
def tables():
    """List all tables in the database."""
    try:
        neon_service = get_neon_service()

        if not neon_service.is_available():
            click.echo("❌ Neon database service is not available. Check NEON_DB_URL configuration.", err=True)
            sys.exit(1)

        click.echo("📋 Listing database tables...", err=True)

        # Get all tables
        tables_query = """
            SELECT tablename, tableowner
            FROM pg_tables
            WHERE schemaname = 'public'
            ORDER BY tablename
        """

        result = neon_service.execute_query(tables_query)

        if not result:
            click.echo("📭 No tables found in database")
            return

        click.echo(f"📊 Found {len(result)} tables:")
        for table in result:
            click.echo(f"  • {table['tablename']} (owner: {table['tableowner']})")

    except Exception as e:
        click.echo(f"❌ Failed to list tables: {e}", err=True)
        sys.exit(1)