"""Sales KPIs and Business Aggregations using Pandas."""

import pandas as pd


def kpis(df: pd.DataFrame) -> dict:
    """Compute core store performance KPIs."""
    order_totals = df.groupby("InvoiceNo")["Revenue"].sum()
    return {
        "revenue": round(float(df["Revenue"].sum()), 2),
        "orders": int(df["InvoiceNo"].nunique()),
        "customers": int(df["CustomerID"].nunique()) if "CustomerID" in df.columns else 0,
        "avg_order": round(float(order_totals.mean()), 2) if len(order_totals) else 0.0,
        "units_sold": int(df["Quantity"].sum()),
    }


def top_products(df: pd.DataFrame, n: int = 10, ascending: bool = False, by: str = "Revenue") -> pd.DataFrame:
    """Return top or bottom N products ranked by Revenue or Quantity."""
    grouped = (
        df.groupby("Description", as_index=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Quantity=("Quantity", "sum"),
            Orders=("InvoiceNo", "nunique"),
        )
    )
    grouped["Revenue"] = grouped["Revenue"].round(2)
    return grouped.sort_values(by, ascending=ascending).head(n).reset_index(drop=True)


def monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly revenue and order counts."""
    m = (
        df.groupby("Month", as_index=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("InvoiceNo", "nunique"),
            Units=("Quantity", "sum"),
        )
        .sort_values("Month")
    )
    m["Revenue"] = m["Revenue"].round(2)
    return m


def by_country(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Aggregate revenue and order volume by Country / Region."""
    c = (
        df.groupby("Country", as_index=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("InvoiceNo", "nunique"),
        )
        .sort_values("Revenue", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    c["Revenue"] = c["Revenue"].round(2)
    return c
