"""Smart Bilingual Business Chatbot & Actionable Recommendation Engine.

Features priority intent routing so questions like 'What are customers complaining
about the most?' or 'Which country has the highest sales?' route to the exact right
domain instead of colliding on generic words like 'most' or 'highest'.
"""

import re
from . import analytics as A
from .sentiment import top_complaints
from .i18n import T


def compute_forecast_growth(future) -> float:
    """Calculate percentage change between first 7 days and last 7 days of forecast."""
    if future is None or len(future) < 14:
        return 0.0
    first_week = float(future["Forecast"].iloc[:7].mean())
    last_week = float(future["Forecast"].iloc[-7:].mean())
    if first_week <= 0:
        return 0.0
    return ((last_week - first_week) / first_week) * 100.0


def recommendations(sales, reviews, future, lang: str = "en") -> list:
    """Generate prioritized, data-driven business recommendations."""
    t = T.get(lang, T["en"])
    recs = []

    # 1. Best seller stock protection
    top_df = A.top_products(sales, 1, ascending=False)
    if not top_df.empty:
        top_row = top_df.iloc[0]
        recs.append(t["rec_stock"].format(top_row["Description"], top_row["Revenue"]))

    # 2. Forecast trend recommendation
    if future is not None and not future.empty:
        growth_pct = compute_forecast_growth(future)
        if growth_pct >= 0:
            recs.append(t["rec_up"].format(growth_pct))
        else:
            recs.append(t["rec_down"].format(growth_pct))

    # 3. Customer complaints action item
    comp = top_complaints(reviews, 3)
    if comp:
        phrases = ", ".join(f"'{c}'" for c, _ in comp)
        recs.append(t["rec_complaint"].format(phrases))

    # 4. Low performing products review
    low_df = A.top_products(sales, 3, ascending=True)
    if not low_df.empty:
        low_names = ", ".join(low_df["Description"].str[:35])
        recs.append(t["rec_low"].format(low_names))

    # 5. Average Order Value (AOV) growth strategy
    k = A.kpis(sales)
    recs.append(t["rec_aov"].format(k["avg_order"]))

    return recs


def _match_specific_product(q_clean: str, sales):
    """Check if the user is asking about a specific product in the catalog."""
    all_prods = A.top_products(sales, n=100, ascending=False)
    total_rev = float(sales["Revenue"].sum()) or 1.0
    q_tokens = set(re.findall(r"[a-z0-9]{3,}", q_clean))
    ignore_tokens = {
        "what", "which", "are", "the", "top", "best", "most", "lowest", "worst",
        "selling", "product", "products", "item", "items", "sales", "revenue",
        "how", "much", "many", "about", "for", "from", "with", "pro", "set",
    }
    search_tokens = q_tokens - ignore_tokens
    if not search_tokens:
        return None

    for idx, row in all_prods.iterrows():
        desc_lower = str(row["Description"]).lower()
        desc_tokens = set(re.findall(r"[a-z0-9]{3,}", desc_lower))
        if desc_lower in q_clean or (search_tokens & desc_tokens):
            return {
                "name": row["Description"],
                "revenue": float(row["Revenue"]),
                "share": (float(row["Revenue"]) / total_rev) * 100.0,
                "quantity": int(row["Quantity"]),
                "orders": int(row["Orders"]),
                "rank": int(idx) + 1,
                "total_products": len(all_prods),
            }
    return None


def answer(q: str, sales, reviews, future, lang: str = "en") -> str:
    """Answer natural-language business questions in English or Arabic using real data."""
    t = T.get(lang, T["en"])
    q_clean = str(q).lower().strip()
    has = lambda *words: any(w in q_clean for w in words)

    # 1. PRIORITY 1: Customer complaints, negative reviews & sentiment analysis
    # Checked FIRST so "What are customers complaining about the most?" routes here, not to "most" -> top products!
    if has(
        "complain", "complaint", "complaints", "problem", "problems", "issue", "issues",
        "negative", "sentiment", "review", "reviews", "feedback", "satisfaction", "angry", "bad",
        "بيشتكوا", "شكاوى", "شكوى", "مشاكل", "مشكلة", "سلبي", "سلبية", "آراء", "تقييمات", "المشاعر", "زعلانين",
    ):
        c = top_complaints(reviews, 5)
        pos_share = float((reviews["sentiment"] == "positive").mean() * 100)
        neu_share = float((reviews["sentiment"] == "neutral").mean() * 100)
        neg_share = float((reviews["sentiment"] == "negative").mean() * 100)
        lines = [f"- **{phrase}** ({count} mentions)" for phrase, count in c]
        return t["ans_complaints"].format(pos_share, neu_share, neg_share) + "\n" + "\n".join(lines)

    # 2. PRIORITY 2: Business recommendations, advice & growth strategy
    if has(
        "recommend", "recommendation", "recommendations", "advice", "suggest", "suggestion",
        "boost", "improve", "strategy", "action", "solutions",
        "توصي", "توصيات", "توصية", "نصيحة", "نصائح", "اقترح", "أعمل إيه", "اعمل ايه", "تحسين", "زيادة المبيعات",
    ):
        return "\n".join(f"- {r}" for r in recommendations(sales, reviews, future, lang))

    # 3. PRIORITY 3: Sales forecast, future predictions & expected trends
    if has(
        "forecast", "predict", "prediction", "future", "next", "upcoming", "trend",
        "increase", "decrease", "will sales", "30 days",
        "متوقع", "توقع", "توقعات", "تزيد", "تقل", "هتزيد", "هتقل", "القادم", "الجاي", "المستقبل",
    ):
        if future is None or future.empty:
            return t["ans_noforecast"]
        tot = float(future["Forecast"].sum())
        avg_daily = float(future["Forecast"].mean())
        growth_pct = compute_forecast_growth(future)
        trend_str = t["up"].format(growth_pct) if growth_pct >= 0 else t["down"].format(growth_pct)
        peak_row = future.loc[future["Forecast"].idxmax()]
        peak_date = str(peak_row["Date"])[:10]
        peak_val = float(peak_row["Forecast"])
        return t["ans_forecast"].format(len(future), tot, avg_daily, trend_str, peak_date, peak_val)

    # 4. PRIORITY 4: Country / Region / Geographic breakdown
    if has("country", "countries", "region", "regions", "where", "location", "market", "دولة", "الدول", "بلد", "منطقة", "المناطق", "فين"):
        c = A.by_country(sales, 6)
        lines = [f"- **{r.Country}**: ${r.Revenue:,.2f} ({r.Orders:,} orders)" for r in c.itertuples()]
        return t["ans_country"] + "\n" + "\n".join(lines)

    # 5. PRIORITY 5: Monthly sales trend / Best month
    if has("month", "monthly", "months", "season", "شهر", "شهري", "الشهور", "أفضل شهر", "اشهر"):
        m = A.monthly_sales(sales)
        best_m = m.loc[m["Revenue"].idxmax()]
        last_m = m.iloc[-1]
        best_str = str(best_m["Month"])[:7]
        last_str = str(last_m["Month"])[:7]
        recent = m.tail(5)
        lines = [f"- **{str(r.Month)[:7]}**: ${r.Revenue:,.2f} ({r.Orders:,} orders)" for r in recent.itertuples()]
        return (
            t["ans_monthly"].format(
                best_str, float(best_m["Revenue"]), int(best_m["Orders"]),
                last_str, float(last_m["Revenue"]), int(last_m["Orders"]),
            )
            + "\n"
            + "\n".join(lines)
        )

    # 6. PRIORITY 6: Lowest / Worst selling products
    if has("worst", "least", "lowest", "slow", "slowest", "bottom", "underperform", "weak", "أقل", "اقل", "الأقل", "اسوأ", "أسوأ", "ضعيف"):
        p = A.top_products(sales, 5, ascending=True)
        lines = [
            f"- **{r.Description}**: ${r.Revenue:,.2f} ({r.Quantity:,} units | {r.Orders:,} orders)"
            for r in p.itertuples()
        ]
        return t["ans_low"] + "\n" + "\n".join(lines)

    # 7. PRIORITY 7: Check if user asked about a specific product by name (when not asking generic top/best)
    if not has("top", "best", "most", "highest", "أكتر", "اكتر", "أكثر", "افضل", "أفضل", "الأعلى"):
        prod_match = _match_specific_product(q_clean, sales)
        if prod_match:
            return t["ans_product_detail"].format(
                prod_match["name"],
                prod_match["revenue"],
                prod_match["share"],
                prod_match["quantity"],
                prod_match["orders"],
                prod_match["rank"],
                prod_match["total_products"],
            )

    # 8. PRIORITY 8: Top / Best selling products
    if has("top", "best", "most", "highest", "popular", "product", "products", "أكتر", "اكتر", "أكثر", "افضل", "أفضل", "الأعلى", "منتج", "المنتجات"):
        p = A.top_products(sales, 5, ascending=False)
        lines = [
            f"- **{r.Description}**: ${r.Revenue:,.2f} ({r.Quantity:,} units | {r.Orders:,} orders)"
            for r in p.itertuples()
        ]
        return t["ans_top"] + "\n" + "\n".join(lines)

    # 9. PRIORITY 9: Overall revenue, orders, customers & KPIs
    if has(
        "total", "revenue", "sales", "order", "orders", "average", "aov", "kpi",
        "customer", "customers", "how much", "summary", "overview",
        "إجمالي", "اجمالي", "مبيعات", "إيرادات", "ايرادات", "طلبات", "متوسط", "عملاء", "العملاء", "كم", "ملخص",
    ):
        k = A.kpis(sales)
        return t["ans_total"].format(k["revenue"], k["orders"], k["customers"], k["units_sold"], k["avg_order"])

    return t["ans_unknown"]
