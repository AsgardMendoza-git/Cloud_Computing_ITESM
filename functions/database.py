"""
database.py
-----------
Clase DatabaseConnector: encapsula la conexión a Azure SQL y la extracción
de datos para el modelo de churn bancario.

Dependencias: pyodbc, pandas
  pip install pyodbc pandas
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

    # ------------------------------------------------------------------
    # Conexión
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Abre la conexión al servidor Azure SQL."""
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
        """Cierra la conexión si está abierta."""
        if self._connection:
            self._connection.close()
            self._connection = None
            print("[DatabaseConnector] Conexión cerrada.")

    # ------------------------------------------------------------------
    # Extracción de datos
    # ------------------------------------------------------------------

    def fetch_data(self, query: str) -> pd.DataFrame:
        """
        Ejecuta una query SQL y regresa un DataFrame de pandas.

        Parámetros
        ----------
        query : Sentencia SELECT a ejecutar

        Retorna
        -------
        pd.DataFrame con los resultados
        """
        if self._connection is None:
            raise RuntimeError("Debes llamar a connect() antes de fetch_data().")
        df = pd.read_sql(query, self._connection)
        print(f"[DatabaseConnector] {len(df)} registros obtenidos.")
        return df

    def get_churn_data(self, table: str = "SalesLT.Product") -> pd.DataFrame:
        """
        Carga la tabla de pedidos de Adventure Works.

        Parámetros
        ----------
        table : Nombre de la tabla con schema (por defecto 'SalesLT.SalesOrderHeader')

        Retorna
        -------
        pd.DataFrame con todas las columnas de la tabla
        """
        query = f"SELECT * FROM {table};"
        return self.fetch_data(query)

    # ------------------------------------------------------------------
    # Context manager (uso con 'with')
    # ------------------------------------------------------------------

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
