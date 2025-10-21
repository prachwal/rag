"""
Tests for neon_service.py module.

This module contains comprehensive tests for the Neon database service,
including connection management, CRUD operations, and error handling.
"""

import pytest
from unittest.mock import patch, MagicMock, mock_open
from Common.services.neon_service import NeonDatabaseService, get_neon_service


class TestNeonDatabaseService:
    """Test cases for NeonDatabaseService class."""

    def setup_method(self):
        """Reset singleton instance before each test."""
        NeonDatabaseService._reset_instance()

    def teardown_method(self):
        """Reset singleton instance after each test."""
        NeonDatabaseService._reset_instance()

    def test_singleton_pattern(self):
        """Test that NeonDatabaseService follows singleton pattern."""
        with patch('Common.services.neon_service.config_service') as mock_config:
            mock_config.get_neon_db_url.return_value = "postgresql://test:test@localhost:5432/test"

            with patch('psycopg2.pool.SimpleConnectionPool'):
                service1 = NeonDatabaseService()
                service2 = NeonDatabaseService()
                assert service1 is service2

    def test_initialization_without_neon_url(self):
        """Test initialization when NEON_DB_URL is not configured."""
        with patch('Common.services.neon_service.config_service') as mock_config:
            mock_config.get_neon_db_url.return_value = None

            service = NeonDatabaseService()
            assert not service.is_available()
            assert service._connection_pool is None

    def test_initialization_with_neon_url(self):
        """Test initialization when NEON_DB_URL is configured."""
        with patch('Common.services.neon_service.config_service') as mock_config:
            mock_config.get_neon_db_url.return_value = "postgresql://user:pass@host:5432/db"

            with patch('psycopg2.pool.SimpleConnectionPool') as mock_pool:
                service = NeonDatabaseService()
                assert service.is_available()
                mock_pool.assert_called_once()

    def test_is_available(self):
        """Test is_available method."""
        with patch('Common.services.neon_service.config_service') as mock_config:
            # Test when not available
            mock_config.get_neon_db_url.return_value = None
            service = NeonDatabaseService()
            assert not service.is_available()

            # Test when available
            NeonDatabaseService._reset_instance()
            mock_config.get_neon_db_url.return_value = "postgresql://test:test@localhost:5432/test"
            with patch('psycopg2.pool.SimpleConnectionPool'):
                service = NeonDatabaseService()
                assert service.is_available()

    def test_get_connection_context_manager(self):
        """Test get_connection context manager."""
        with patch('Common.services.neon_service.config_service') as mock_config:
            mock_config.get_neon_db_url.return_value = "postgresql://test:test@localhost:5432/test"

            with patch('psycopg2.pool.SimpleConnectionPool') as mock_pool_class:
                mock_pool = MagicMock()
                mock_pool_class.return_value = mock_pool
                mock_conn = MagicMock()
                mock_pool.getconn.return_value = mock_conn

                service = NeonDatabaseService()

                with service.get_connection() as conn:
                    assert conn is mock_conn

                mock_pool.getconn.assert_called_once()
                mock_pool.putconn.assert_called_once_with(mock_conn)

    def test_get_connection_unavailable_service(self):
        """Test get_connection when service is not available."""
        with patch('Common.services.neon_service.config_service') as mock_config:
            mock_config.get_neon_db_url.return_value = None

            service = NeonDatabaseService()

            with pytest.raises(RuntimeError, match="Neon database service is not available"):
                with service.get_connection():
                    pass

    def test_execute_query(self):
        """Test execute_query method."""
        with patch('Common.services.neon_service.config_service') as mock_config:
            mock_config.get_neon_db_url.return_value = "postgresql://test:test@localhost:5432/test"

            with patch('psycopg2.pool.SimpleConnectionPool') as mock_pool_class:
                mock_pool = MagicMock()
                mock_pool_class.return_value = mock_pool
                mock_conn = MagicMock()
                mock_cursor = MagicMock()
                mock_pool.getconn.return_value = mock_conn
                mock_conn.cursor.return_value = mock_cursor
                mock_cursor.fetchall.return_value = [{'id': 1, 'name': 'test'}]
                mock_cursor.description = [('id',), ('name',)]

                service = NeonDatabaseService()
                result = service.execute_query("SELECT * FROM test")

                assert result == [{'id': 1, 'name': 'test'}]
                mock_cursor.execute.assert_called_once_with("SELECT * FROM test", ())
                mock_conn.commit.assert_called_once()

    def test_execute_non_query(self):
        """Test execute_non_query method."""
        with patch('Common.services.neon_service.config_service') as mock_config:
            mock_config.get_neon_db_url.return_value = "postgresql://test:test@localhost:5432/test"

            with patch('psycopg2.pool.SimpleConnectionPool') as mock_pool_class:
                mock_pool = MagicMock()
                mock_pool_class.return_value = mock_pool
                mock_conn = MagicMock()
                mock_cursor = MagicMock()
                mock_pool.getconn.return_value = mock_conn
                mock_conn.cursor.return_value = mock_cursor
                mock_cursor.rowcount = 5

                service = NeonDatabaseService()
                result = service.execute_non_query("DELETE FROM test WHERE id = %s", (1,))

                assert result == 5
                mock_cursor.execute.assert_called_once_with("DELETE FROM test WHERE id = %s", (1,))
                mock_conn.commit.assert_called_once()

    def test_insert_data(self):
        """Test insert_data method."""
        with patch('Common.services.neon_service.config_service') as mock_config:
            mock_config.get_neon_db_url.return_value = "postgresql://test:test@localhost:5432/test"

            with patch('psycopg2.pool.SimpleConnectionPool') as mock_pool_class:
                mock_pool = MagicMock()
                mock_pool_class.return_value = mock_pool
                mock_conn = MagicMock()
                mock_cursor = MagicMock()
                mock_pool.getconn.return_value = mock_conn
                mock_conn.cursor.return_value = mock_cursor
                mock_cursor.fetchone.return_value = {'id': 123}

                service = NeonDatabaseService()
                result = service.insert_data("users", {"name": "John", "email": "john@example.com"})

                assert result == 123
                expected_query = "INSERT INTO users (name, email) VALUES (%s, %s) RETURNING id"
                mock_cursor.execute.assert_called_once_with(expected_query, ("John", "john@example.com"))

    def test_update_data(self):
        """Test update_data method."""
        with patch('Common.services.neon_service.config_service') as mock_config:
            mock_config.get_neon_db_url.return_value = "postgresql://test:test@localhost:5432/test"

            with patch('psycopg2.pool.SimpleConnectionPool') as mock_pool_class:
                mock_pool = MagicMock()
                mock_pool_class.return_value = mock_pool
                mock_conn = MagicMock()
                mock_cursor = MagicMock()
                mock_pool.getconn.return_value = mock_conn
                mock_conn.cursor.return_value = mock_cursor
                mock_cursor.rowcount = 3

                service = NeonDatabaseService()
                result = service.update_data("users", {"name": "Jane"}, "id = %s", (1,))

                assert result == 3
                expected_query = "UPDATE users SET name = %s WHERE id = %s"
                mock_cursor.execute.assert_called_once_with(expected_query, ("Jane", 1))

    def test_delete_data(self):
        """Test delete_data method."""
        with patch('Common.services.neon_service.config_service') as mock_config:
            mock_config.get_neon_db_url.return_value = "postgresql://test:test@localhost:5432/test"

            with patch('psycopg2.pool.SimpleConnectionPool') as mock_pool_class:
                mock_pool = MagicMock()
                mock_pool_class.return_value = mock_pool
                mock_conn = MagicMock()
                mock_cursor = MagicMock()
                mock_pool.getconn.return_value = mock_conn
                mock_conn.cursor.return_value = mock_cursor
                mock_cursor.rowcount = 2

                service = NeonDatabaseService()
                result = service.delete_data("users", "id = %s", (1,))

                assert result == 2
                mock_cursor.execute.assert_called_once_with("DELETE FROM users WHERE id = %s", (1,))

    def test_table_exists(self):
        """Test table_exists method."""
        with patch('Common.services.neon_service.config_service') as mock_config:
            mock_config.get_neon_db_url.return_value = "postgresql://test:test@localhost:5432/test"

            with patch('psycopg2.pool.SimpleConnectionPool') as mock_pool_class:
                mock_pool = MagicMock()
                mock_pool_class.return_value = mock_pool
                mock_conn = MagicMock()
                mock_cursor = MagicMock()
                mock_pool.getconn.return_value = mock_conn
                mock_conn.cursor.return_value = mock_cursor
                mock_cursor.fetchall.return_value = [{'exists': True}]

                service = NeonDatabaseService()
                result = service.table_exists("users")

                assert result is True

    def test_health_check_success(self):
        """Test health_check method when database is healthy."""
        with patch('Common.services.neon_service.config_service') as mock_config:
            mock_config.get_neon_db_url.return_value = "postgresql://test:test@localhost:5432/test"

            with patch('psycopg2.pool.SimpleConnectionPool') as mock_pool_class:
                mock_pool = MagicMock()
                mock_pool_class.return_value = mock_pool
                mock_conn = MagicMock()
                mock_cursor = MagicMock()
                mock_pool.getconn.return_value = mock_conn
                mock_conn.cursor.return_value = mock_cursor
                mock_cursor.fetchall.return_value = [{
                    'version': 'PostgreSQL 15.4',
                    'current_database': 'testdb',
                    'current_user': 'testuser'
                }]

                service = NeonDatabaseService()
                result = service.health_check()

                assert result['status'] == 'healthy'
                assert 'PostgreSQL 15.4' in result['version']
                assert result['database'] == 'testdb'
                assert result['user'] == 'testuser'

    def test_health_check_failure(self):
        """Test health_check method when database connection fails."""
        with patch('Common.services.neon_service.config_service') as mock_config:
            mock_config.get_neon_db_url.return_value = "postgresql://test:test@localhost:5432/test"

            with patch('psycopg2.pool.SimpleConnectionPool') as mock_pool_class:
                mock_pool = MagicMock()
                mock_pool_class.return_value = mock_pool
                mock_conn = MagicMock()
                mock_cursor = MagicMock()
                mock_pool.getconn.return_value = mock_conn
                mock_conn.cursor.return_value = mock_cursor
                mock_cursor.execute.side_effect = Exception("Connection failed")

                service = NeonDatabaseService()
                result = service.health_check()

                assert result['status'] == 'unhealthy'
                assert 'Connection failed' in result['error']

    def test_close_all_connections(self):
        """Test close_all_connections method."""
        with patch('Common.services.neon_service.config_service') as mock_config:
            mock_config.get_neon_db_url.return_value = "postgresql://test:test@localhost:5432/test"

            with patch('psycopg2.pool.SimpleConnectionPool') as mock_pool_class:
                mock_pool = MagicMock()
                mock_pool_class.return_value = mock_pool

                service = NeonDatabaseService()
                service.close_all_connections()

                mock_pool.closeall.assert_called_once()


class TestNeonServiceConvenienceFunctions:
    """Test cases for convenience functions."""

    def setup_method(self):
        """Reset singleton instance before each test."""
        NeonDatabaseService._reset_instance()

    def teardown_method(self):
        """Reset singleton instance after each test."""
        NeonDatabaseService._reset_instance()

    def test_get_neon_service(self):
        """Test get_neon_service convenience function."""
        with patch('Common.services.neon_service.config_service') as mock_config:
            mock_config.get_neon_db_url.return_value = "postgresql://test:test@localhost:5432/test"

            with patch('psycopg2.pool.SimpleConnectionPool'):
                service = get_neon_service()
                assert isinstance(service, NeonDatabaseService)


class TestNeonServiceIntegration:
    """Integration tests for Neon service (mocked)."""

    def setup_method(self):
        """Reset singleton instance before each test."""
        NeonDatabaseService._reset_instance()

    def teardown_method(self):
        """Reset singleton instance after each test."""
        NeonDatabaseService._reset_instance()

    def test_full_workflow(self):
        """Test a complete workflow with Neon service."""
        with patch('Common.services.neon_service.config_service') as mock_config:
            mock_config.get_neon_db_url.return_value = "postgresql://test:test@localhost:5432/test"

            with patch('psycopg2.pool.SimpleConnectionPool') as mock_pool_class:
                mock_pool = MagicMock()
                mock_pool_class.return_value = mock_pool
                mock_conn = MagicMock()
                mock_cursor = MagicMock()
                mock_pool.getconn.return_value = mock_conn
                mock_conn.cursor.return_value = mock_cursor

                # Setup mock responses
                mock_cursor.rowcount = 1
                mock_cursor.fetchone.return_value = {'id': 1}

                service = NeonDatabaseService()

                # Test table creation
                service.create_table("test_table", {"id": "SERIAL PRIMARY KEY", "name": "VARCHAR(100)"})

                # Test data insertion
                user_id = service.insert_data("users", {"name": "Alice", "email": "alice@example.com"})
                assert user_id == 1

                # Test data update
                updated = service.update_data("users", {"name": "Alice Smith"}, "id = %s", (1,))
                assert updated == 1

                # Test data deletion
                deleted = service.delete_data("users", "id = %s", (1,))
                assert deleted == 1

                # Test health check
                mock_cursor.fetchall.return_value = [{
                    'version': 'PostgreSQL 15.4',
                    'current_database': 'testdb',
                    'current_user': 'testuser'
                }]
                health = service.health_check()
                assert health['status'] == 'healthy'