"""
Neon Database Service for PostgreSQL connections.

This module provides a service for connecting to and interacting with
Neon PostgreSQL databases with connection pooling and error handling.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple, Union
from contextlib import contextmanager
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor, execute_values
import psycopg2.extensions

from Common.services.config_service import config_service

logger = logging.getLogger(__name__)


class NeonDatabaseService:
    """Service for managing Neon PostgreSQL database connections and operations."""

    _instance: Optional['NeonDatabaseService'] = None
    _connection_pool: Optional[pool.SimpleConnectionPool] = None

    def __new__(cls) -> 'NeonDatabaseService':
        """Singleton pattern to ensure single database service instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the Neon database service."""
        if self._connection_pool is None:
            self._initialize_connection_pool()

    def _initialize_connection_pool(self) -> None:
        """Initialize the connection pool for Neon database."""
        neon_db_url = config_service.get_neon_db_url()

        if not neon_db_url:
            logger.warning("NEON_DB_URL not configured. Neon database service will not be available.")
            return

        try:
            # Parse the database URL to extract connection parameters
            # Neon URL format: postgresql://username:password@hostname:port/database?sslmode=require
            import urllib.parse
            parsed = urllib.parse.urlparse(neon_db_url)

            db_params = {
                'host': parsed.hostname,
                'port': parsed.port or 5432,
                'database': parsed.path.lstrip('/'),
                'user': parsed.username,
                'password': parsed.password,
                'sslmode': 'require',  # Neon requires SSL
                'connect_timeout': 10,
            }

            # Create connection pool
            self._connection_pool = pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                **db_params
            )

            logger.info("Neon database connection pool initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Neon database connection pool: {e}")
            raise RuntimeError(f"Neon database initialization failed: {e}")

    def is_available(self) -> bool:
        """Check if Neon database service is available."""
        return self._connection_pool is not None

    @contextmanager
    def get_connection(self):
        """Get a database connection from the pool."""
        if not self.is_available():
            raise RuntimeError("Neon database service is not available. Check NEON_DB_URL configuration.")

        conn = None
        try:
            conn = self._connection_pool.getconn()
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                self._connection_pool.putconn(conn)

    @contextmanager
    def get_cursor(self, cursor_factory=RealDictCursor):
        """Get a database cursor with automatic connection management."""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Database operation error: {e}")
                raise
            finally:
                cursor.close()

    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query and return results.

        Args:
            query: SQL query string
            params: Query parameters (optional)

        Returns:
            List of dictionaries representing rows
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params or ())
            if cursor.description:
                return [dict(row) for row in cursor.fetchall()]
            return []

    def execute_non_query(self, query: str, params: Optional[Tuple] = None) -> int:
        """
        Execute a non-SELECT query (INSERT, UPDATE, DELETE).

        Args:
            query: SQL query string
            params: Query parameters (optional)

        Returns:
            Number of affected rows
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params or ())
            return cursor.rowcount

    def execute_many(self, query: str, params_list: List[Tuple]) -> None:
        """
        Execute a query with multiple parameter sets.

        Args:
            query: SQL query string
            params_list: List of parameter tuples
        """
        with self.get_cursor() as cursor:
            cursor.executemany(query, params_list)

    def insert_data(self, table: str, data: Dict[str, Any]) -> int:
        """
        Insert a single row into a table.

        Args:
            table: Table name
            data: Dictionary of column-value pairs

        Returns:
            ID of inserted row (if table has auto-incrementing primary key)
        """
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['%s'] * len(data))
        values = tuple(data.values())

        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders}) RETURNING id"

        with self.get_cursor() as cursor:
            cursor.execute(query, values)
            result = cursor.fetchone()
            return result['id'] if result else None

    def insert_many(self, table: str, data_list: List[Dict[str, Any]]) -> None:
        """
        Insert multiple rows into a table efficiently.

        Args:
            table: Table name
            data_list: List of dictionaries with column-value pairs
        """
        if not data_list:
            return

        columns = list(data_list[0].keys())
        query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES %s"

        values = [tuple(row[col] for col in columns) for row in data_list]

        with self.get_cursor() as cursor:
            execute_values(cursor, query, values)

    def update_data(self, table: str, data: Dict[str, Any], where_clause: str, where_params: Tuple) -> int:
        """
        Update rows in a table.

        Args:
            table: Table name
            data: Dictionary of column-value pairs to update
            where_clause: WHERE clause (e.g., "id = %s")
            where_params: Parameters for WHERE clause

        Returns:
            Number of updated rows
        """
        set_clause = ', '.join([f"{col} = %s" for col in data.keys()])
        values = tuple(data.values()) + where_params

        query = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"

        return self.execute_non_query(query, values)

    def delete_data(self, table: str, where_clause: str, where_params: Tuple) -> int:
        """
        Delete rows from a table.

        Args:
            table: Table name
            where_clause: WHERE clause (e.g., "id = %s")
            where_params: Parameters for WHERE clause

        Returns:
            Number of deleted rows
        """
        query = f"DELETE FROM {table} WHERE {where_clause}"
        return self.execute_non_query(query, where_params)

    def create_table(self, table_name: str, schema: Dict[str, str]) -> None:
        """
        Create a table with the given schema.

        Args:
            table_name: Name of the table to create
            schema: Dictionary mapping column names to SQL types
        """
        columns = ', '.join([f"{col} {sql_type}" for col, sql_type in schema.items()])
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"

        self.execute_non_query(query)

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists.

        Args:
            table_name: Name of the table to check

        Returns:
            True if table exists, False otherwise
        """
        query = """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_name = %s
            )
        """
        result = self.execute_query(query, (table_name,))
        return result[0]['exists'] if result else False

    def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get information about table columns.

        Args:
            table_name: Name of the table

        Returns:
            List of column information dictionaries
        """
        query = """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position
        """
        return self.execute_query(query, (table_name,))

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the database connection.

        Returns:
            Dictionary with health check results
        """
        try:
            result = self.execute_query("SELECT version(), current_database(), current_user")
            if result:
                return {
                    'status': 'healthy',
                    'version': result[0]['version'],
                    'database': result[0]['current_database'],
                    'user': result[0]['current_user']
                }
            else:
                return {'status': 'unhealthy', 'error': 'No response from database'}
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}

    def close_all_connections(self) -> None:
        """Close all connections in the pool."""
        if self._connection_pool:
            self._connection_pool.closeall()
            logger.info("All Neon database connections closed")

    @classmethod
    def _reset_instance(cls) -> None:
        """Reset the singleton instance. Used for testing."""
        if cls._instance and cls._instance._connection_pool:
            cls._instance.close_all_connections()
        cls._instance = None
        cls._connection_pool = None


# Global Neon database service instance
neon_service = NeonDatabaseService()


def get_neon_service() -> NeonDatabaseService:
    """Get Neon database service instance."""
    return neon_service