"""Data Loading, Cleaning & Built-in Synthetic Store Dataset Generation.

Note: No external CSV dataset download is required! If no file is uploaded,
this module generates a realistic, deterministic 12-month retail sales dataset
and bilingual customer reviews dataset in memory so the project runs out-of-the-box.
"""

import numpy as np
import pandas as pd


PRODUCTS = [
    ("SKU-101", "Wireless Noise-Canceling Headphones", 89.99, 1.45),
    ("SKU-102", "Smart Fitness Watch Pro", 129.50, 1.35),
    ("SKU-103", "Mechanical RGB Gaming Keyboard", 64.00, 1.15),
    ("SKU-104", "Ergonomic Vertical Wireless Mouse", 34.99, 1.10),
    ("SKU-105", "USB-C 8-in-1 Aluminum Hub", 42.00, 1.25),
    ("SKU-106", "Portable 20000mAh Fast Power Bank", 49.99, 1.30),
    ("SKU-107", "Full HD 1080p Auto-Focus Webcam", 54.50, 0.90),
    ("SKU-108", "Adjustable Aluminum Laptop Stand", 29.99, 1.05),
    ("SKU-109", "Waterproof Bluetooth Speaker", 39.99, 0.85),
    ("SKU-110", "Smart LED Desk Lamp with Wireless Charger", 45.00, 0.70),
    ("SKU-111", "Microfiber Laptop Sleeve 15-inch", 19.99, 0.45),
    ("SKU-112", "Braided HDMI 2.1 High-Speed Cable", 14.50, 0.35),
]

COUNTRIES = [
    ("Egypt", 0.38),
    ("Saudi Arabia", 0.24),
    ("United Arab Emirates", 0.16),
    ("United Kingdom", 0.10),
    ("Germany", 0.07),
    ("United States", 0.05),
]


def make_sample_sales(n_days=365, seed=42):
    """Generate a realistic 12-month store transactions DataFrame with clear trend & seasonality."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp("2026-09-30"), periods=n_days, freq="D")

    skus = [p[0] for p in PRODUCTS]
    names = [p[1] for p in PRODUCTS]
    prices = [p[2] for p in PRODUCTS]
    weights = np.array([p[3] for p in PRODUCTS])
    weights = weights / weights.sum()

    country_names = [c[0] for c in COUNTRIES]
    country_probs = [c[1] for c in COUNTRIES]

    dow_multipliers = {0: 0.85, 1: 0.90, 2: 0.95, 3: 1.15, 4: 1.35, 5: 1.40, 6: 0.80}

    rows = []
    invoice_counter = 500000

    for idx, dt in enumerate(dates):
        # Clear upward growth trend + weekly cycle + monthly cycle
        trend_factor = 1.0 + 0.55 * (idx / n_days)
        dow_factor = dow_multipliers[dt.dayofweek]
        seasonal_factor = 1.0 + 0.22 * np.sin(2 * np.pi * idx / 30.0)
        target_orders = int(round(16 * trend_factor * dow_factor * seasonal_factor + rng.normal(0, 1.2)))
        n_orders = max(6, target_orders)

        for _ in range(n_orders):
            invoice_counter += 1
            inv_no = str(invoice_counter)
            cust_id = int(rng.integers(1001, 1450))
            country = rng.choice(country_names, p=country_probs)
            n_items = int(rng.choice([1, 2, 3], p=[0.55, 0.32, 0.13]))
            chosen_indices = rng.choice(len(PRODUCTS), size=n_items, replace=False, p=weights)

            for p_idx in chosen_indices:
                qty = int(rng.choice([1, 2, 3, 4], p=[0.50, 0.30, 0.14, 0.06]))
                unit_price = prices[p_idx]
                rows.append(
                    {
                        "InvoiceNo": inv_no,
                        "StockCode": skus[p_idx],
                        "Description": names[p_idx],
                        "Quantity": qty,
                        "InvoiceDate": dt + pd.Timedelta(hours=int(rng.integers(9, 22))),
                        "UnitPrice": unit_price,
                        "CustomerID": cust_id,
                        "Country": country,
                    }
                )

    df = pd.DataFrame(rows)
    return clean_sales(df)


def load_sales(file_or_path=None):
    """Load sales data from an uploaded CSV/XLSX file, or generate synthetic store data."""
    if file_or_path is None:
        return make_sample_sales()

    name = getattr(file_or_path, "name", str(file_or_path)).lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(file_or_path)
    else:
        df = pd.read_csv(file_or_path, encoding="utf-8", encoding_errors="ignore")
    return clean_sales(df)


def clean_sales(df):
    """Clean and standardize transaction records and compute Revenue & Month."""
    df = df.copy()
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("invoiceno", "invoice_no", "order_id", "orderid", "invoice"):
            col_map[c] = "InvoiceNo"
        elif cl in ("description", "product", "product_name", "item"):
            col_map[c] = "Description"
        elif cl in ("quantity", "qty", "units"):
            col_map[c] = "Quantity"
        elif cl in ("invoicedate", "invoice_date", "date", "order_date"):
            col_map[c] = "InvoiceDate"
        elif cl in ("unitprice", "unit_price", "price"):
            col_map[c] = "UnitPrice"
        elif cl in ("customerid", "customer_id", "client_id"):
            col_map[c] = "CustomerID"
        elif cl in ("country", "region", "city"):
            col_map[c] = "Country"
    df = df.rename(columns=col_map)

    df = df.dropna(subset=["Description", "InvoiceDate"]).drop_duplicates()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df = df[~df["InvoiceNo"].str.upper().str.startswith("C")]
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce").fillna(0.0)
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Description"] = df["Description"].astype(str).str.strip()
    if "Country" not in df.columns:
        df["Country"] = "Global"
    if "CustomerID" not in df.columns:
        df["CustomerID"] = range(1, len(df) + 1)
    df["Revenue"] = (df["Quantity"] * df["UnitPrice"]).round(2)
    df["Month"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    return df


def make_sample_reviews(n=1600, seed=42):
    """Generate a realistic bilingual (EN + AR) customer reviews dataset."""
    rng = np.random.default_rng(seed)

    pos_en = [
        "great build quality", "very fast delivery", "love this product",
        "excellent value for money", "works smoothly without issues",
        "highly recommend this store", "sleek and modern design",
        "customer support was super helpful", "battery life is impressive",
        "exceeded my expectations",
    ]
    neg_en = [
        "late delivery took over two weeks", "arrived damaged in the box",
        "poor plastic quality feels cheap", "price is too expensive for what you get",
        "stopped working after three days", "bad customer service never replied",
        "missing charging cable in package", "overheats quickly during normal use",
        "wrong item sent by mistake", "bluetooth connection keeps dropping",
    ]
    neu_en = [
        "product is okay for basic use", "average quality nothing special",
        "matches the description fairly well", "arrived on time with standard packaging",
        "does the job as expected", "decent price for a backup device",
        "normal performance neither great nor terrible", "acceptable build for entry level",
    ]

    pos_ar = [
        "جودة ممتازة جداً", "التوصيل سريع والخدمة رائعة", "منتج ممتاز وأنصح بشرائه",
        "سعر مناسب جداً مقابل الجودة", "يعمل بكفاءة عالية وبدون مشاكل",
    ]
    neg_ar = [
        "التوصيل متأخر جداً استغرق أسبوعين", "الجودة سيئة والخامة ضعيفة",
        "خدمة العملاء لا ترد على الشكاوى", "توقف عن العمل بعد أيام قليلة",
        "المنتج وصل مكسور والتغليف سيئ",
    ]
    neu_ar = [
        "المنتج عادي ومقبول السعر", "جودة متوسطة تؤدي الغرض",
        "وصل في الموعد المحدد والتغليف عادي", "أداء مقبول مقارنة بالسعر",
    ]

    rows = []
    for i in range(n):
        label = rng.choice(["positive", "negative", "neutral"], p=[0.52, 0.28, 0.20])
        use_ar = (i % 5 == 0)
        if use_ar:
            pool = {"positive": pos_ar, "negative": neg_ar, "neutral": neu_ar}[label]
            parts = rng.choice(pool, size=2, replace=False)
            text = f"{parts[0]}، {parts[1]}."
        else:
            pool = {"positive": pos_en, "negative": neg_en, "neutral": neu_en}[label]
            parts = rng.choice(pool, size=2, replace=False)
            # Introduce realistic mixed phrasing in ~3% of reviews
            if i % 33 == 0:
                extra = rng.choice(neu_en)
                text = f"{parts[0].capitalize()}, though {extra}."
            else:
                text = f"{parts[0].capitalize()}, and {parts[1]}."
        rows.append({"review": text, "sentiment": label})

    return pd.DataFrame(rows)


def load_reviews(file_or_path=None):
    """Load customer reviews from an uploaded CSV or generate realistic sample reviews."""
    if file_or_path is None:
        return make_sample_reviews()

    df = pd.read_csv(file_or_path, encoding="utf-8", encoding_errors="ignore")
    df.columns = [c.lower().strip() for c in df.columns]
    text_col = next((c for c in df.columns if c in ("review", "text", "review_text", "comment")), df.columns[0])
    label_col = next((c for c in df.columns if c in ("sentiment", "label", "rating_label")), df.columns[-1])
    df = df[[text_col, label_col]].rename(columns={text_col: "review", label_col: "sentiment"})
    df = df.dropna()
    df["review"] = df["review"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()
    return df
