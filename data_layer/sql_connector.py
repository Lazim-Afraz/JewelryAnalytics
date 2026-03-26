"""
SQL Server Connector Module
data_layer/sql_connector.py

Handles all connections to SQL Server from WSL.
Uses TCP/IP with host IP + port — named instances (\\SQLEXPRESS) don't
resolve from WSL, so the connection goes through 172.x.x.x,1433 instead.
"""

import pyodbc
import pandas as pd
from typing import Optional, List
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SQLServerConnector:
    """
    SQL Server connector for WSL environments.

    Key differences from a native Windows connector:
    - Uses SQL Server authentication (not Windows auth)
    - Uses IP address + port instead of server name or instance name
    - Driver name must be 'ODBC Driver 17 for SQL Server' (Linux driver)
    """

    def __init__(self,
                 server: str,
                 database: str,
                 username: Optional[str] = None,
                 password: Optional[str] = None,
                 use_windows_auth: bool = False,
                 timeout: int = 30):
        """
        Initialize SQL Server connection.

        Args:
            server:           Host IP + port  e.g. "172.20.144.1,1433"
                              NOT "localhost\\SQLEXPRESS" — that only works on Windows.
            database:         Database name   e.g. "JewelryPortfolioDB"
            username:         SQL login username
            password:         SQL login password
            use_windows_auth: Must be False in WSL
            timeout:          Connection timeout in seconds

        Example:
            connector = SQLServerConnector(
                server   = "172.20.144.1,1433",
                database = "JewelryPortfolioDB",
                username = "jewelry_user",
                password = "YourStrongPassword!"
            )
        """
        self.server           = server
        self.database         = database
        self.username         = username
        self.password         = password
        self.use_windows_auth = use_windows_auth
        self.timeout          = timeout

        self.connection = None
        self.cursor     = None
        self._build_connection_string()

    def _build_connection_string(self):
        """
        Build ODBC connection string.

        Both branches use 'ODBC Driver 17 for SQL Server' — this is the
        Linux driver installed via Microsoft's apt package. The old
        '{SQL Server}' driver is Windows-only and will not be found in WSL.
        """
        driver = "ODBC Driver 17 for SQL Server"

        if self.use_windows_auth:
            # Windows auth works only when running natively on Windows.
            # If you reach this branch from WSL the connection will fail —
            # set use_windows_auth=False and supply username + password instead.
            self.conn_str = (
                f"DRIVER={{{driver}}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"Trusted_Connection=yes;"
            )
        else:
            # SQL Server authentication — the correct mode for WSL
            self.conn_str = (
                f"DRIVER={{{driver}}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"UID={self.username};"
                f"PWD={self.password};"
            )

    def connect(self) -> bool:
        """
        Establish connection to SQL Server.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            logger.info(f"Connecting to {self.database} on {self.server}...")
            self.connection = pyodbc.connect(self.conn_str, timeout=self.timeout)
            self.cursor     = self.connection.cursor()
            logger.info("✅ Connection successful!")
            return True

        except pyodbc.Error as e:
            logger.error(f"❌ Connection failed: {e}")
            logger.error("Common causes:")
            logger.error("  1. Wrong IP in SERVER — run: cat /etc/resolv.conf | grep nameserver")
            logger.error("  2. TCP/IP not enabled in SQL Server Configuration Manager")
            logger.error("  3. Port 1433 blocked by Windows Firewall")
            logger.error("  4. SQL Server auth mode not enabled (must allow both modes)")
            logger.error("  5. Login not created in SSMS")
            return False

    def test_connection(self) -> dict:
        """
        Test database connection and return status details.

        Returns:
            dict with keys: success, message, server, database, timestamp,
            and (on success) sql_version.
        """
        result = {
            'success':   False,
            'message':   '',
            'server':    self.server,
            'database':  self.database,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        try:
            if self.connect():
                cursor  = self.connection.cursor()
                version = cursor.execute("SELECT @@VERSION").fetchone()[0]

                result['success']     = True
                result['message']     = f"Connected successfully to {self.database}"
                result['sql_version'] = version.split('\n')[0]

                logger.info("✅ Connection test passed")
            else:
                result['message'] = "Connection failed — check logs above for cause"

        except Exception as e:
            result['message'] = f"Error: {str(e)}"
            logger.error(f"Connection test exception: {e}")

        return result

    def execute_query(self,
                      query: str,
                      params: Optional[tuple] = None) -> pd.DataFrame:
        """
        Execute a SQL query and return results as a DataFrame.

        Args:
            query:  SQL query string
            params: Optional parameters for parameterised queries

        Returns:
            pandas DataFrame with query results

        Example:
            df = connector.execute_query(
                "SELECT TOP 10 * FROM BRANCH_PERFORMANCE_SUMMARY1 WHERE DATE >= ?",
                ('2024-01-01',)
            )
        """
        if not self.connection:
            self.connect()

        try:
            logger.info("Executing query...")
            if params:
                df = pd.read_sql(query, self.connection, params=params)
            else:
                df = pd.read_sql(query, self.connection)

            logger.info(f"✅ Query returned {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            raise

    def execute_query_chunked(self,
                               query: str,
                               chunk_size: int = 100_000,
                               params: Optional[tuple] = None) -> pd.DataFrame:
        """
        Execute a large query in chunks to avoid memory issues.

        Args:
            query:      SQL query
            chunk_size: Rows per chunk (default 100,000)
            params:     Query parameters

        Returns:
            Complete DataFrame (all chunks combined)
        """
        if not self.connection:
            self.connect()

        try:
            logger.info(f"Executing chunked query (chunk_size={chunk_size:,})...")
            chunks = []

            for chunk_df in pd.read_sql(query, self.connection,
                                        chunksize=chunk_size,
                                        params=params):
                chunks.append(chunk_df)
                logger.info(f"  Loaded chunk: {len(chunk_df):,} rows")

            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"✅ Total rows loaded: {len(df):,}")
            return df

        except Exception as e:
            logger.error(f"❌ Chunked query failed: {e}")
            raise

    def get_table_info(self, table_name: str) -> pd.DataFrame:
        """
        Return column names, types, and nullability for a given table.

        Args:
            table_name: Table name (e.g. 'BRANCH_PERFORMANCE_SUMMARY1')

        Returns:
            DataFrame with columns: COLUMN_NAME, DATA_TYPE,
            IS_NULLABLE, CHARACTER_MAXIMUM_LENGTH
        """
        query = """
            SELECT
                COLUMN_NAME,
                DATA_TYPE,
                IS_NULLABLE,
                CHARACTER_MAXIMUM_LENGTH
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = ?
            ORDER BY ORDINAL_POSITION
        """
        return self.execute_query(query, (table_name,))

    def get_row_count(self, table_name: str, where_clause: str = '') -> int:
        """
        Return the row count for a table, with an optional WHERE filter.

        Args:
            table_name:   Table name
            where_clause: Optional WHERE condition (without the WHERE keyword)

        Returns:
            Integer row count
        """
        query = f"SELECT COUNT(*) AS count FROM {table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"

        result = self.execute_query(query)
        return int(result.iloc[0]['count'])

    def list_tables(self) -> List[str]:
        """
        Return a list of all base tables in the connected database.

        Returns:
            List of table name strings
        """
        query = """
            SELECT TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_NAME
        """
        df = self.execute_query(query)
        return df['TABLE_NAME'].tolist()

    def close(self):
        """Close cursor and connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            logger.info("Connection closed.")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ─── Convenience function ────────────────────────────────────────────────────

def quick_query(query: str,
                server: str = None,
                database: str = None) -> pd.DataFrame:
    """
    Run a one-off query using settings from DatabaseConfig.

    Example:
        df = quick_query("SELECT TOP 5 * FROM BRANCH_PERFORMANCE_SUMMARY1")
    """
    from config.database_config import DatabaseConfig

    server   = server   or DatabaseConfig.SERVER
    database = database or DatabaseConfig.DATABASE

    with SQLServerConnector(
        server           = server,
        database         = database,
        username         = DatabaseConfig.USERNAME,
        password         = DatabaseConfig.PASSWORD,
        use_windows_auth = DatabaseConfig.USE_WINDOWS_AUTH
    ) as conn:
        return conn.execute_query(query)


# ─── Self-test ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== SQL Server Connector — WSL connection test ===\n")

    from config.database_config import DatabaseConfig

    print(f"Server  : {DatabaseConfig.SERVER}")
    print(f"Database: {DatabaseConfig.DATABASE}")
    print(f"User    : {DatabaseConfig.USERNAME}")
    print(f"Windows auth: {DatabaseConfig.USE_WINDOWS_AUTH}\n")

    connector = SQLServerConnector(
        server           = DatabaseConfig.SERVER,
        database         = DatabaseConfig.DATABASE,
        username         = DatabaseConfig.USERNAME,
        password         = DatabaseConfig.PASSWORD,
        use_windows_auth = DatabaseConfig.USE_WINDOWS_AUTH
    )

    result = connector.test_connection()
    print(f"Success : {result['success']}")
    print(f"Message : {result['message']}")

    if result['success']:
        print(f"Version : {result.get('sql_version', 'n/a')}\n")

        tables = connector.list_tables()
        print(f"Tables in {DatabaseConfig.DATABASE}:")
        for t in tables:
            print(f"  - {t}")

        if tables:
            print(f"\nSample row from {tables[0]}:")
            df = connector.execute_query(f"SELECT TOP 1 * FROM {tables[0]}")
            for col in df.columns:
                print(f"  {col}: {df.iloc[0][col]}")

    connector.close()
    print("\n=== Test complete ===")
