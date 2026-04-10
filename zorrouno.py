import pandas as pd

class processor:
    @staticmethod
    def embbed(d):
        """
        Preprocesa la tabla SalesLT.Product:
        - Crea la variable objetivo binaria: IsBlack (1 si Color='Black', 0 si no)
        - Conserva solo features numéricas relevantes
        - Elimina filas con NaN en features clave
        """
        # Crear variable objetivo binaria a partir de Color
        if "Color" in d.columns:
            d = d.copy()
            d["IsBlack"] = (d["Color"] == "Black").astype(int)

        # Conservar solo columnas numéricas útiles como features + target
        keep = ["StandardCost", "ListPrice", "Weight",
                "ProductCategoryID", "ProductModelID", "IsBlack"]
        existing = [c for c in keep if c in d.columns]
        d = d[existing]

        # Eliminar filas donde falten valores en features
        d = d.dropna()
        return d