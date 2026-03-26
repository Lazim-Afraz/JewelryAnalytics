"""
Database Configuration for SQL Server Connection
config/database_config.py

WSL-compatible configuration for localhost\\SQLEXPRESS / JewelryPortfolioDB.

CHANGES FROM YOUR ORIGINAL FILE:
    1. SERVER  — changed from 'localhost\\SQLEXPRESS' to IP + port
                 (named instances don't resolve from WSL)
    2. DRIVER  — changed from '{SQL Server}' to '{ODBC Driver 17 for SQL Server}'
                 ('{SQL Server}' is a Windows-only driver, not available in WSL)
    3. USE_WINDOWS_AUTH — changed to False
                 (Windows Authentication doesn't work cross-environment from WSL)
    4. USERNAME / PASSWORD — you must fill these in
                 (create this login in SSMS first — see setup steps below)

HOW TO GET YOUR SERVER IP (run this in WSL terminal every time you reboot):
    cat /etc/resolv.conf | grep nameserver | awk '{print $2}'
    Copy the result, e.g. 172.20.144.1, and set SERVER = "172.20.144.1,1433"
"""

import os
from typing import Optional


class DatabaseConfig:
    """SQL Server configuration settings — WSL edition"""

    # =========================================================================
    # STEP 1: Paste your WSL host IP here
    #   Run in WSL:  cat /etc/resolv.conf | grep nameserver | awk '{print $2}'
    #   Then set:    SERVER = "172.20.x.x,1433"
    #
    # DO NOT use 'localhost\\SQLEXPRESS' — that string only works on Windows.
    # DO NOT use a backslash or instance name — use IP,PORT format.
    # =========================================================================
    SERVER = "172.28.160.1,1433"    # e.g. "172.20.144.1,1433"

    # Your database name — already correct, no change needed
    DATABASE = "JewelryPortfolioDB"

    # =========================================================================
    # STEP 2: Set to False — Windows Auth does not work from WSL
    # =========================================================================
    USE_WINDOWS_AUTH = False

    # =========================================================================
    # STEP 3: Fill in the SQL login you created in SSMS
    #   In SSMS run:
    #       CREATE LOGIN jewelry_user WITH PASSWORD = 'YourStrongPassword!';
    #       USE JewelryPortfolioDB;
    #       CREATE USER jewelry_user FOR LOGIN jewelry_user;
    #       ALTER ROLE db_datareader ADD MEMBER jewelry_user;
    #       ALTER ROLE db_datawriter ADD MEMBER jewelry_user;
    # =========================================================================
    USERNAME = "jewelry_user"             # replace if you used a different name
    PASSWORD = "12345678"      # replace with your actual password

    # =========================================================================
    # Table names — update if yours differ
    # =========================================================================
    TRANSACTIONS_TABLE = "BRANCH_PERFORMANCE_SUMMARY1"
    PRODUCTS_TABLE     = "products"
    BRANCHES_TABLE     = "branches"

    # Default date filter
    DEFAULT_START_DATE = "2024-01-01"

    # Query timeout in seconds
    QUERY_TIMEOUT = 300

    # =========================================================================
    # Connection string builder
    # =========================================================================

    @classmethod
    def get_connection_string(cls) -> str:
        """
        Build pyodbc connection string based on configuration.

        DRIVER is always 'ODBC Driver 17 for SQL Server' — this is the
        Linux driver installed via Microsoft's apt package.
        '{SQL Server}' (your original driver) is Windows-only and will
        raise 'Data source name not found' in WSL.

        Returns:
            str: pyodbc-compatible connection string
        """
        driver = "ODBC Driver 17 for SQL Server"

        if cls.USE_WINDOWS_AUTH:
            # Only works when running natively on Windows, not from WSL
            return (
                f"DRIVER={{{driver}}};"
                f"SERVER={cls.SERVER};"
                f"DATABASE={cls.DATABASE};"
                f"Trusted_Connection=yes;"
            )
        else:
            # SQL Server authentication — use this in WSL
            return (
                f"DRIVER={{{driver}}};"
                f"SERVER={cls.SERVER};"
                f"DATABASE={cls.DATABASE};"
                f"UID={cls.USERNAME};"
                f"PWD={cls.PASSWORD};"
            )


# =============================================================================
# Environment variable loader (optional — for keeping credentials out of code)
# =============================================================================

def load_from_env():
    """
    Override DatabaseConfig values from a .env file.

    Create a file called .env in your project root:
        SQL_SERVER=172.20.144.1,1433
        SQL_DATABASE=JewelryPortfolioDB
        SQL_USERNAME=jewelry_user
        SQL_PASSWORD=YourStrongPassword!

    Then call load_from_env() before instantiating any connector.
    Requires: pip install python-dotenv
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("python-dotenv not installed — skipping .env load")
        return

    DatabaseConfig.SERVER   = os.getenv("SQL_SERVER",   DatabaseConfig.SERVER)
    DatabaseConfig.DATABASE = os.getenv("SQL_DATABASE", DatabaseConfig.DATABASE)
    DatabaseConfig.USERNAME = os.getenv("SQL_USERNAME", DatabaseConfig.USERNAME)
    DatabaseConfig.PASSWORD = os.getenv("SQL_PASSWORD", DatabaseConfig.PASSWORD)


# =============================================================================
# Quick sanity check — run with: python -m config.database_config
# =============================================================================

if __name__ == "__main__":
    print("=== Database Configuration ===")
    print(f"Server   : {DatabaseConfig.SERVER}")
    print(f"Database : {DatabaseConfig.DATABASE}")
    print(f"Auth     : {'Windows' if DatabaseConfig.USE_WINDOWS_AUTH else 'SQL Server'}")
    print(f"Username : {DatabaseConfig.USERNAME}")

    conn_str = DatabaseConfig.get_connection_string()
    if "PWD=" in conn_str:
        masked = conn_str.split("PWD=")[0] + "PWD=*****;"
    else:
        masked = conn_str

    print(f"\nConnection string (password masked):\n{masked}")

    # Attempt a live connection test
    try:
        import pyodbc
        print("\nTesting connection...")
        conn    = pyodbc.connect(conn_str, timeout=10)
        version = conn.cursor().execute("SELECT @@VERSION").fetchone()[0]
        print(f"✅ Connected! SQL Server: {version.split(chr(10))[0]}")
        conn.close()
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nCheck:")
        print("  1. SERVER IP is correct (rerun: cat /etc/resolv.conf | grep nameserver)")
        print("  2. TCP/IP is enabled in SQL Server Configuration Manager")
        print("  3. Port 1433 is open in Windows Firewall")
        print("  4. SQL Server is set to 'SQL Server and Windows Authentication mode'")
        print("  5. Login exists in SSMS (CREATE LOGIN ...)")
        print("  6. ODBC Driver 17 is installed in WSL (odbcinst -q -d)")
