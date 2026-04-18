"""
connector.py
------------
Encapsulates the connection to Azure SQL and data extraction.
"""

import pyodbc
import pandas as pd


class DatabaseConnector:

    def __init__(self, server: str, database: str, username: str, password: str):
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self._connection = None

    def connect(self) -> None:
        connection_string = (
            f"DRIVER={{SQL Server}};"
            f"SERVER={self.server},1433;"
            f"DATABASE={self.database};"
            f"UID={self.username};"
            f"PWD={self.password};"
            "Connection Timeout=60;"
        )
        self._connection = pyodbc.connect(connection_string)
        print(f"[DatabaseConnector] Conectado a '{self.database}'")

    def disconnect(self) -> None:
        if self._connection:
            self._connection.close()
            self._connection = None
            print("[DatabaseConnector] Conexión cerrada.")

    def fetch_data(self, query: str) -> pd.DataFrame:
        if self._connection is None:
            raise RuntimeError("Debes llamar a connect() antes de fetch_data().")
        df = pd.read_sql(query, self._connection)
        print(f"[DatabaseConnector] {len(df)} registros obtenidos.")
        return df

    def get_churn_data(self, table: str = "SalesLT.Product") -> pd.DataFrame:
        return self.fetch_data(f"SELECT * FROM {table};")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
