"""Streamlit Frontend Dashboard for AI Business Assistant.

Run from the project root directory:
    streamlit run frontend/app.py
"""

import os
import re
import sys
from collections import Counter
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Add project root to sys.path so `backend` modules can be imported cleanly
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.data_prep import load_sales, load_reviews
from backend import analytics as A
from backend.forecasting import train_forecaster, forecast_future, feature_importance_df
from backend.sentiment import train_sentiment, predict_sentiment

UI_TEXT = {
    "en": {
        "title": "AI Business Assistant",
        "sub": "AI-Powered Business Chatbot, Sales Analytics, ML Forecasting & Sentiment Intelligence",
        "upload_sales": "Upload Sales File (CSV/XLSX) [Optional]",
        "upload_rev": "Upload Customer Reviews (CSV) [Optional]",
        "data_source_info": "Using built-in realistic store dataset (upload a file above to analyze your own data).",
        "tabs": [
            "💼 Business Chatbot",
            "📊 Sales Analytics",
            "📈 Sales Forecasting",
            "💬 Sentiment Analysis",
            "💡 Recommendations",
        ],
        "pulse_rev": "Total Revenue",
        "pulse_fut": "30-Day Forecast",
        "pulse_neg": "Negative Reviews",
        "pulse_top": "Best Seller",
        "clear_chat": "🗑️ Reset Conversation",
        "quick_q": "⚡ Quick Questions:",
        "sample_questions": [
            "Which product has the most bad reviews?",
            "What is the most expensive product?",
            "What are customers complaining about the most?",
            "What are the top selling products?",
            "Will sales increase or decrease in the next 30 days?",
            "Give me recommendations to boost sales",
        ],
        "ask": "Ask anything (e.g., most expensive product, product with most bad reviews, top sales, 30-day forecast)...",
        "latest_reply_badge": "✨ Latest AI Response",
        "earlier_chats": "🕒 Earlier Questions in This Session",
        "revenue": "Total Revenue",
        "orders": "Total Orders",
        "customers": "Unique Customers",
        "avg": "Avg Order Value",
        "monthly": "Monthly Revenue Trend",
        "top": "Top-Selling Products by Revenue",
        "low": "Lowest-Performing Products by Revenue",
        "country": "Revenue Breakdown by Region / Country",
        "metrics": "Model Evaluation (Time-Based Split: Last 60 Days)",
        "avsp": "Actual vs. Predicted Daily Sales (Test Set)",
        "future": "Future Daily Sales Forecast",
        "feat_imp": "Random Forest Feature Importance",
        "forecast_days": "Forecast Horizon (Days)",
        "forecast_total": "Projected Revenue (Next {} Days)",
        "forecast_avg": "Projected Daily Average",
        "forecast_trend": "Expected Sales Trend",
        "acc": "Classification Accuracy",
        "f1": "Macro F1-Score",
        "total_rev_count": "Reviews Analyzed",
        "dist": "Customer Sentiment Distribution",
        "conf_mat": "Confusion Matrix (Test Set)",
        "complaints": "Most Frequent Real Customer Complaints",
        "try": "Test Live Sentiment Analysis (English or Arabic)",
        "predict": "Analyze Sentiment",
        "pred_result": "Predicted Sentiment",
        "confidence": "Model Confidence",
        "loading": "Loading data & initializing AI models...",
        "up": "Increase 📈 (+{:.1f}%)",
        "down": "Decrease 📉 ({:.1f}%)",
    },
    "ar": {
        "title": "المساعد الذكي للأعمال (AI Business Assistant)",
        "sub": "مساعد ذكي تفاعلي، تحليل المبيعات، التنبؤ بالذكاء الاصطناعي، وتحليل آراء العملاء",
        "upload_sales": "ارفع ملف المبيعات (CSV/XLSX) [اختياري]",
        "upload_rev": "ارفع ملف تقييمات العملاء (CSV) [اختياري]",
        "data_source_info": "يتم حالياً استخدام بيانات المتجر الافتراضية المدمجة (يمكنك رفع ملفاتك الخاصة من القائمة الجانبية).",
        "tabs": [
            "💼 المساعد الذكي",
            "📊 تحليل المبيعات",
            "📈 التنبؤ بالمبيعات",
            "💬 تحليل المشاعر",
            "💡 التوصيات الذكية",
        ],
        "pulse_rev": "إجمالي الإيرادات",
        "pulse_fut": "توقع 30 يوم",
        "pulse_neg": "التقييمات السلبية",
        "pulse_top": "الأعلى مبيعاً",
        "clear_chat": "🗑️ مسح المحادثة",
        "quick_q": "⚡ أسئلة سريعة:",
        "sample_questions": [
            "إيه أكتر منتج عليه تقييمات سلبية؟",
            "إيه هو أغلى منتج في المتجر؟",
            "العملاء بيشتكوا من إيه؟",
            "إيه أكتر المنتجات مبيعاً؟",
            "هل المبيعات متوقع تزيد ولا تقل الشهر القادم؟",
            "أعطني توصيات لزيادة المبيعات",
        ],
        "ask": "اسأل المساعد الذكي (مثلاً: أغلى منتج، أكتر منتج عليه شكاوى، أعلى المبيعات، التوقعات)...",
        "latest_reply_badge": "✨ أحدث إجابة من المساعد الذكي",
        "earlier_chats": "🕒 الأسئلة السابقة في هذه الجلسة",
        "revenue": "إجمالي الإيرادات",
        "orders": "إجمالي الطلبات",
        "customers": "عدد العملاء",
        "avg": "متوسط قيمة الطلب",
        "monthly": "تطور الإيرادات الشهرية",
        "top": "أعلى المنتجات مبيعاً حسب الإيراد",
        "low": "أقل المنتجات مبيعاً حسب الإيراد",
        "country": "توزيع الإيرادات حسب المنطقة / الدولة",
        "metrics": "تقييم نموذج التنبؤ (آخر 60 يوم كبيانات اختبار)",
        "avsp": "المبيعات الفعلية مقابل المتوقعة يومياً",
        "future": "التنبؤ بالمبيعات اليومية القادمة",
        "feat_imp": "أهمية المتغيرات في نموذج Random Forest",
        "forecast_days": "فترة التنبؤ (بالأيام)",
        "forecast_total": "الإيرادات المتوقعة (خلال {} يوم)",
        "forecast_avg": "المتوسط اليومي المتوقع",
        "forecast_trend": "اتجاه المبيعات المتوقع",
        "acc": "دقة التصنيف (Accuracy)",
        "f1": "معيار F1-Score",
        "total_rev_count": "إجمالي التقييمات المحللة",
        "dist": "توزيع مشاعر وآراء العملاء",
        "conf_mat": "مصفوفة الارتباك (Confusion Matrix)",
        "complaints": "أكثر شكاوى العملاء الحقيقية تكراراً",
        "try": "جرّب تحليل المشاعر المباشر (بالعربية أو الإنجليزية)",
        "predict": "تحليل الرأي",
        "pred_result": "التصنيف المتوقع",
        "confidence": "نسبة ثقة النموذج",
        "loading": "جاري تجهيز البيانات وتدريب نماذج الذكاء الاصطناعي...",
        "up": "زيادة 📈 (+{:.1f}%)",
        "down": "انخفاض 📉 ({:.1f}%)",
    },
}


def _has_arabic(text: str) -> bool:
    return any("\u0600" <= ch <= "\u06FF" for ch in str(text))


def _filter_by_lang(texts: list, lang: str = None) -> list:
    if lang == "ar":
        ar_list = [t for t in texts if _has_arabic(t)]
        return ar_list if ar_list else texts
    if lang == "en":
        en_list = [t for t in texts if not _has_arabic(t)]
        return en_list if en_list else texts
    return texts


def attach_products_to_reviews(sales: pd.DataFrame, reviews: pd.DataFrame, is_custom_rev: bool = False) -> pd.DataFrame:
    """Link every review to a product in the active sales catalog with realistic per-product complaint patterns."""
    rev = reviews.copy()
    products = sales["Description"].dropna().unique().tolist()
    if not products:
        rev["product"] = "General Store Product"
        return rev

    for col in rev.columns:
        if col.lower().strip() in ("product", "product_name", "description", "item"):
            if set(rev[col].dropna().unique()) & set(products):
                rev["product"] = rev[col].astype(str)
                return rev

    rng = np.random.default_rng(42)
    prod_stats = (
        sales.groupby("Description", as_index=False)
        .agg(Revenue=("Revenue", "sum"), Quantity=("Quantity", "sum"))
        .sort_values("Revenue", ascending=False)
        .reset_index(drop=True)
    )
    prod_list = prod_stats["Description"].tolist()
    n_prods = len(prod_list)

    # Distinct complaint themes per product rank so each product has its own realistic main complaint
    en_complaint_themes = [
        ("Stopped working completely after three days of normal use", "Customer support never replied to my warranty request"),
        ("Arrived damaged in the box due to weak packaging", "Outer casing was cracked right out of the box"),
        ("Overheats quickly and makes a loud buzzing noise", "Build quality feels cheap compared to the photos"),
        ("Courier delayed delivery by over two weeks without notice", "Tracking number was not updated for ten days"),
        ("Missing power cord and accessories in the sealed box", "Wrong attachment size was sent by mistake"),
        ("Price is too expensive for the actual performance", "Stopped holding charge after the first week"),
    ]
    ar_complaint_themes = [
        ("توقف الجهاز عن العمل تماماً بعد ثلاثة أيام من الاستخدام", "خدمة العملاء لا ترد على طلبات الضمان والاستبدال"),
        ("المنتج وصل مكسور في العلبة بسبب ضعف التغليف", "الغلاف الخارجي كان مشروخاً عند فتح الكرتونة"),
        ("يسخن بسرعة كبيرة ويصدر صوتاً عالياً أثناء التشغيل", "جودة الخامات ضعيفة مقارنة بالسعر المعروض"),
        ("تأخر التوصيل لأكثر من أسبوعين بدون أي إشعار", "شركة الشحن تأخرت جداً في تسليم الطلب"),
        ("نقص في الملحقات وكابل التشغيل داخل العلبة", "تم إرسال مقاس أو موديل مختلف عن المطلوب بالخطأ"),
    ]

    pos_weights = np.linspace(1.5, 0.6, n_prods)
    pos_weights /= pos_weights.sum()

    # Clear, distinct hierarchy of negative review counts across products
    raw_neg_w = np.array([1.1, 2.3, 1.9, 1.5, 1.25] + [0.65] * max(0, n_prods - 5), dtype=float)[:n_prods]
    neg_weights = raw_neg_w / raw_neg_w.sum()

    assigned_products = []
    updated_reviews = []

    for idx, row in rev.iterrows():
        sent = str(row["sentiment"]).lower().strip()
        orig_txt = str(row["review"])
        if sent == "negative":
            p_idx = int(rng.choice(n_prods, p=neg_weights))
            chosen_prod = prod_list[p_idx]
            assigned_products.append(chosen_prod)
            # If using built-in reviews, tailor complaint text to the product's primary defect profile
            if not is_custom_rev:
                if _has_arabic(orig_txt):
                    c1, c2 = ar_complaint_themes[p_idx % len(ar_complaint_themes)]
                    updated_reviews.append(f"{c1}، {c2}." if idx % 2 == 0 else f"{c2}، {c1}.")
                else:
                    c1, c2 = en_complaint_themes[p_idx % len(en_complaint_themes)]
                    updated_reviews.append(f"{c1}, and {c2.lower()}." if idx % 2 == 0 else f"{c2}, and {c1.lower()}.")
            else:
                updated_reviews.append(orig_txt)
        else:
            p_idx = int(rng.choice(n_prods, p=pos_weights))
            assigned_products.append(prod_list[p_idx])
            updated_reviews.append(orig_txt)

    rev["product"] = assigned_products
    rev["review"] = updated_reviews
    return rev


def get_product_price_catalog(sales: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
    """Return products ranked by UnitPrice alongside their total units sold and total revenue."""
    cat = (
        sales.groupby("Description", as_index=False)
        .agg(
            UnitPrice=("UnitPrice", "median"),
            Revenue=("Revenue", "sum"),
            Quantity=("Quantity", "sum"),
            Orders=("InvoiceNo", "nunique"),
        )
        .sort_values(["UnitPrice", "Revenue"], ascending=[ascending, False])
        .reset_index(drop=True)
    )
    cat["UnitPrice"] = cat["UnitPrice"].round(2)
    cat["Revenue"] = cat["Revenue"].round(2)
    return cat


def get_products_by_sentiment(reviews: pd.DataFrame, target_sentiment: str = "negative", lang: str = None) -> pd.DataFrame:
    """Rank products by count and percentage of negative (or positive) reviews, including top complaint per product."""
    rev = reviews.copy()
    if "product" not in rev.columns:
        rev["product"] = "Store Product"

    rows = []
    for prod_name, grp in rev.groupby("product"):
        total_r = len(grp)
        neg_grp = grp[grp["sentiment"] == "negative"]
        pos_grp = grp[grp["sentiment"] == "positive"]
        neu_grp = grp[grp["sentiment"] == "neutral"]

        neg_cnt = len(neg_grp)
        pos_cnt = len(pos_grp)
        neg_pct = (neg_cnt / max(1, total_r)) * 100.0
        pos_pct = (pos_cnt / max(1, total_r)) * 100.0

        top_issues = extract_real_complaints(grp, n=1, lang=lang)
        main_issue = top_issues[0][0] if top_issues else "General quality / shipping delay"

        sample_revs = _filter_by_lang(neg_grp["review"].dropna().astype(str).tolist(), lang=lang)
        sample_quote = sample_revs[0] if sample_revs else ""

        rows.append(
            {
                "Product": prod_name,
                "TotalReviews": total_r,
                "NegativeReviews": neg_cnt,
                "NegativePct": round(neg_pct, 1),
                "PositiveReviews": pos_cnt,
                "PositivePct": round(pos_pct, 1),
                "NeutralReviews": len(neu_grp),
                "MainIssue": main_issue,
                "SampleQuote": sample_quote,
            }
        )

    out_df = pd.DataFrame(rows)
    if target_sentiment == "negative":
        return out_df.sort_values(["NegativeReviews", "NegativePct"], ascending=[False, False]).reset_index(drop=True)
    return out_df.sort_values(["PositiveReviews", "PositivePct"], ascending=[False, False]).reset_index(drop=True)


def extract_real_complaints(df: pd.DataFrame, n: int = 6, lang: str = None):
    """Extract full, human-readable complaint statements from negative customer reviews."""
    all_neg = df[df["sentiment"] == "negative"]["review"].dropna().astype(str).tolist()
    neg_series = _filter_by_lang(all_neg, lang=lang)

    clause_counts = Counter()
    for review_text in neg_series:
        parts = re.split(r"\s*(?:,\s+and\s+|,\s+but\s+|,\s+though\s+|،\s*و|،\s*|\.\s+)\s*", review_text.strip())
        for part in parts:
            cleaned = part.strip(" .,!؟؛")
            if len(cleaned.split()) >= 3:
                formatted = cleaned[0].upper() + cleaned[1:] if cleaned else cleaned
                clause_counts[formatted] += 1

    if not clause_counts:
        for review_text in neg_series:
            cleaned = review_text.strip()
            if cleaned:
                clause_counts[cleaned] += 1

    return clause_counts.most_common(n)


def get_sample_negative_reviews(df: pd.DataFrame, n: int = 4, lang: str = None) -> list:
    """Return actual verbatim negative customer reviews paired with their product name."""
    neg_df = df[df["sentiment"] == "negative"].copy()
    if "product" not in neg_df.columns:
        neg_df["product"] = "Store Item"

    records = neg_df[["review", "product"]].dropna().to_dict(orient="records")
    if lang == "ar":
        ar_recs = [r for r in records if _has_arabic(r["review"])]
        if ar_recs:
            records = ar_recs
    elif lang == "en":
        en_recs = [r for r in records if not _has_arabic(r["review"])]
        if en_recs:
            records = en_recs

    seen = set()
    unique_quotes = []
    for r in records:
        txt = str(r["review"]).strip()
        if txt not in seen:
            seen.add(txt)
            unique_quotes.append((txt, str(r["product"])))
            if len(unique_quotes) >= n:
                break
    return unique_quotes


def compute_growth_pct(future: pd.DataFrame) -> float:
    if future is None or len(future) < 14:
        return 0.0
    first_w = float(future["Forecast"].iloc[:7].mean())
    last_w = float(future["Forecast"].iloc[-7:].mean())
    return ((last_w - first_w) / first_w) * 100.0 if first_w > 0 else 0.0


def build_recommendations(sales: pd.DataFrame, reviews: pd.DataFrame, future: pd.DataFrame, lang: str = "en") -> list:
    recs = []
    top_df = A.top_products(sales, 1, ascending=False)
    if not top_df.empty:
        top_row = top_df.iloc[0]
        if lang == "ar":
            recs.append(f"**حماية المنتج الرئيسي:** المنتج '{top_row['Description']}' يحقق أعلى إيرادات (${top_row['Revenue']:,.0f}) — احرص على توفره دائماً لتجنب نفاد المخزون.")
        else:
            recs.append(f"**Protect Core Revenue Driver:** '{top_row['Description']}' generates your highest revenue (${top_row['Revenue']:,.0f}) — maintain strict inventory replenishment to prevent stockouts.")

    # Highlight the #1 product with the most bad reviews
    prod_neg_df = get_products_by_sentiment(reviews, target_sentiment="negative", lang=lang)
    if not prod_neg_df.empty:
        worst_prod = prod_neg_df.iloc[0]
        if lang == "ar":
            recs.append(
                f"**فحص جودة المنتج الأكثر شكوى ({worst_prod['Product']}):** سجل **{worst_prod['NegativeReviews']} تقييم سلبي** ({worst_prod['NegativePct']:.1f}%) وأبرز شكوى هي *\"{worst_prod['MainIssue']}\"*."
            )
        else:
            recs.append(
                f"**Fix Most-Complained Product ({worst_prod['Product']}):** Logged **{worst_prod['NegativeReviews']} negative reviews** ({worst_prod['NegativePct']:.1f}% negative rate), primarily due to *\"{worst_prod['MainIssue']}\"*."
            )

    if future is not None and not future.empty:
        g = compute_growth_pct(future)
        if g >= 0:
            recs.append(
                f"**الاستعداد لنمو الطلب:** من المتوقع نمو المبيعات بنسبة ({g:+.1f}%) خلال الـ 30 يوماً القادمة — زوّد المخزون للمنتجات الأكثر طلباً."
                if lang == "ar"
                else f"**Prepare for Demand Growth:** Sales are forecasted to grow ({g:+.1f}%) over the next 30 days — increase safety stock for fast-moving items."
            )
        else:
            recs.append(
                f"**تنشيط المبيعات:** من المتوقع تراجع بنسبة ({g:+.1f}%) — خطط لحملات تسويقية وعروض خصم لتحفيز المبيعات."
                if lang == "ar"
                else f"**Counter Projected Slowdown:** Sales are forecasted to dip ({g:+.1f}%) — launch targeted promotional campaigns and discount bundles."
            )

    low_df = A.top_products(sales, 3, ascending=True)
    if not low_df.empty:
        low_names = ", ".join(low_df["Description"].str[:35])
        recs.append(
            f"**تحسين المنتجات الأقل مبيعاً:** راجع التسعير أو اعمل عروض ترويجية للمنتجات بطيئة الحركة ({low_names})."
            if lang == "ar"
            else f"**Optimize Low Performers:** Review pricing, bundle offers, or phase out slow-moving items ({low_names})."
        )

    k = A.kpis(sales)
    recs.append(
        f"**رفع متوسط سلة الشراء (الحالي: ${k['avg_order']:,.2f}):** قدّم شحن مجاني عند تجاوز مبلغ معين أو عروض المنتجات المكملة."
        if lang == "ar"
        else f"**Boost Average Order Value (Current: ${k['avg_order']:,.2f}):** Introduce free-shipping thresholds or cross-sell accessories at checkout."
    )
    return recs


def _match_specific_product(q_clean: str, sales: pd.DataFrame):
    """Check if the user mentioned a specific product name from the catalog."""
    cat = get_product_price_catalog(sales, ascending=False)
    q_tokens = set(re.findall(r"[a-z0-9]{3,}", q_clean))
    ignore = {
        "what", "which", "are", "the", "top", "best", "most", "lowest", "worst",
        "selling", "product", "products", "item", "items", "sales", "revenue",
        "how", "much", "many", "about", "for", "from", "with", "pro", "set",
        "expensive", "cheap", "cheapest", "price", "prices", "cost", "bad",
        "good", "review", "reviews", "complain", "complaints", "negative", "positive",
    }
    search_tokens = q_tokens - ignore
    if not search_tokens:
        return None

    for _, row in cat.iterrows():
        desc_lower = str(row["Description"]).lower()
        desc_tokens = set(re.findall(r"[a-z0-9]{3,}", desc_lower))
        if desc_lower in q_clean or (search_tokens & desc_tokens):
            return row["Description"]
    return None


def smart_chatbot_answer(q: str, sales: pd.DataFrame, reviews: pd.DataFrame, future: pd.DataFrame, lang: str = "en") -> str:
    """Intelligent Business Chatbot answering product-level reviews, unit prices, KPIs, and forecasts."""
    q_clean = str(q).lower().strip()
    has = lambda *words: any(w in q_clean for w in words)

    asks_product = has("product", "products", "item", "items", "منتج", "المنتج", "المنتجات", "سلعة")
    asks_bad_reviews = has(
        "bad", "negative", "worst review", "bad review", "bad reviews", "negative review", "negative reviews",
        "complain", "complaint", "complaints", "problem", "problems", "issue", "issues", "hated", "angry",
        "بيشتكوا", "شكاوى", "شكوى", "مشاكل", "مشكلة", "سلبي", "سلبية", "سيء", "سيئة", "زعلانين",
    )
    asks_good_reviews = has(
        "good review", "good reviews", "positive review", "positive reviews", "best review", "best reviews",
        "highest rated", "most liked", "loved", "satisfied", "satisfaction",
        "ايجابي", "إيجابي", "ايجابية", "إيجابية", "أفضل تقييم", "احسن تقييم", "رضا",
    )

    # -------------------------------------------------------------------------
    # 0. SPECIFIC PRODUCT DEEP-DIVE (if user named a specific product like "Espresso" or "Headphones")
    # -------------------------------------------------------------------------
    matched_prod = _match_specific_product(q_clean, sales)
    if matched_prod:
        cat = get_product_price_catalog(sales, ascending=False).reset_index(drop=True)
        row = cat[cat["Description"] == matched_prod].iloc[0]
        total_rev = float(sales["Revenue"].sum()) or 1.0
        rev_share = (float(row["Revenue"]) / total_rev) * 100.0
        prod_revs = reviews[reviews["product"] == matched_prod]
        neg_cnt = int((prod_revs["sentiment"] == "negative").sum())
        pos_cnt = int((prod_revs["sentiment"] == "positive").sum())
        tot_r = max(1, len(prod_revs))
        top_iss = extract_real_complaints(prod_revs, n=2, lang=lang)
        iss_str = ", ".join(f'"{c}" ({cnt})' for c, cnt in top_iss) if top_iss else "None"

        if lang == "ar":
            return (
                f"**📦 تقرير تفصيلي عن المنتج: {matched_prod}**\n"
                f"- **سعر الوحدة (Unit Price):** ${row['UnitPrice']:,.2f}\n"
                f"- **إجمالي الإيرادات:** ${row['Revenue']:,.2f} ({rev_share:.1f}% من مبيعات المتجر)\n"
                f"- **الكمية المباعة:** {int(row['Quantity']):,} قطعة عبر {int(row['Orders']):,} طلب\n"
                f"- **تقييمات العملاء ({len(prod_revs)} تقييم):** {(pos_cnt/tot_r)*100:.1f}% إيجابي ({pos_cnt}) | {(neg_cnt/tot_r)*100:.1f}% سلبي ({neg_cnt})\n"
                f"- **أبرز شكاوى العملاء لهذا المنتج:** {iss_str}"
            )
        return (
            f"**📦 Product Deep-Dive: {matched_prod}**\n"
            f"- **Unit Price:** ${row['UnitPrice']:,.2f}\n"
            f"- **Total Revenue:** ${row['Revenue']:,.2f} ({rev_share:.1f}% of store revenue)\n"
            f"- **Sales Volume:** {int(row['Quantity']):,} units sold across {int(row['Orders']):,} orders\n"
            f"- **Customer Reviews ({len(prod_revs)} total):** {(pos_cnt/tot_r)*100:.1f}% Positive ({pos_cnt}) | {(neg_cnt/tot_r)*100:.1f}% Negative ({neg_cnt})\n"
            f"- **Main Reported Complaints:** {iss_str}"
        )

    # -------------------------------------------------------------------------
    # 1. WHICH PRODUCT HAS THE MOST BAD REVIEWS / COMPLAINTS?
    # Matches: "what is the most product that has bad reviews", "which product has most complaints", etc.
    # -------------------------------------------------------------------------
    if asks_product and asks_bad_reviews:
        prod_neg = get_products_by_sentiment(reviews, target_sentiment="negative", lang=lang)
        price_map = dict(zip(sales["Description"], sales["UnitPrice"]))
        worst = prod_neg.iloc[0]
        worst_price = float(price_map.get(worst["Product"], 0.0))

        if lang == "ar":
            lines = []
            for idx, r in prod_neg.head(5).iterrows():
                p_val = float(price_map.get(r["Product"], 0.0))
                lines.append(
                    f"{idx + 1}. **{r['Product']}** (سعر الوحدة: ${p_val:,.2f}) — **{r['NegativeReviews']} تقييم سلبي** "
                    f"({r['NegativePct']:.1f}% من تقييمات المنتج) | أبرز شكوى: *\"{r['MainIssue']}\"*"
                )
            return (
                f"**🚨 المنتج الأكثر حصولاً على تقييمات سلبية هو: {worst['Product']}**\n"
                f"- سجل **{worst['NegativeReviews']} تقييم سلبي** من إجمالي {worst['TotalReviews']} تقييم لهذا المنتج (**{worst['NegativePct']:.1f}% نسبة الاستياء**، سعر الوحدة: ${worst_price:,.2f}).\n"
                f"- **السبب الرئيسي للشكوى:** *\"{worst['MainIssue']}\"*\n"
                f"- **مثال حقيقي من تقييمات العملاء:** *\"{worst['SampleQuote']}\"*\n\n"
                f"**📉 ترتيب أعلى 5 منتجات من حيث عدد التقييمات السلبية:**\n"
                + "\n".join(lines)
            )
        else:
            lines = []
            for idx, r in prod_neg.head(5).iterrows():
                p_val = float(price_map.get(r["Product"], 0.0))
                lines.append(
                    f"{idx + 1}. **{r['Product']}** (${p_val:,.2f}/unit) — **{r['NegativeReviews']} bad reviews** "
                    f"({r['NegativePct']:.1f}% of its {r['TotalReviews']} reviews) | Main issue: *\"{r['MainIssue']}\"*"
                )
            return (
                f"**🚨 Most-Complained Product: {worst['Product']}**\n"
                f"- Logged **{worst['NegativeReviews']} negative reviews** out of {worst['TotalReviews']} total reviews for this product (**{worst['NegativePct']:.1f}% negative rate**, Unit Price: ${worst_price:,.2f}).\n"
                f"- **Primary Complaint:** *\"{worst['MainIssue']}\"*\n"
                f"- **Sample Customer Review:** *\"{worst['SampleQuote']}\"*\n\n"
                f"**📉 Top 5 Products with the Most Bad Reviews:**\n"
                + "\n".join(lines)
            )

    # -------------------------------------------------------------------------
    # 2. WHICH PRODUCT HAS THE BEST / MOST POSITIVE REVIEWS?
    # -------------------------------------------------------------------------
    if asks_product and asks_good_reviews:
        prod_pos = get_products_by_sentiment(reviews, target_sentiment="positive", lang=lang)
        best = prod_pos.iloc[0]
        if lang == "ar":
            lines = [
                f"{idx + 1}. **{r['Product']}** — **{r['PositiveReviews']} تقييم إيجابي** ({r['PositivePct']:.1f}% نسبة رضا من {r['TotalReviews']} تقييم)"
                for idx, r in prod_pos.head(5).iterrows()
            ]
            return (
                f"**🌟 المنتج الأعلى تقييماً ورضا من العملاء هو: {best['Product']}**\n"
                f"- حصل على **{best['PositiveReviews']} تقييم إيجابي** ({best['PositivePct']:.1f}% نسبة رضا).\n\n"
                f"**🏆 أعلى 5 منتجات حصولاً على تقييمات إيجابية:**\n" + "\n".join(lines)
            )
        lines = [
            f"{idx + 1}. **{r['Product']}** — **{r['PositiveReviews']} positive reviews** ({r['PositivePct']:.1f}% satisfaction out of {r['TotalReviews']} reviews)"
            for idx, r in prod_pos.head(5).iterrows()
        ]
        return (
            f"**🌟 Highest-Rated Product by Customer Reviews: {best['Product']}**\n"
            f"- Received **{best['PositiveReviews']} positive reviews** ({best['PositivePct']:.1f}% positive rating).\n\n"
            f"**🏆 Top 5 Products with the Most Positive Reviews:**\n" + "\n".join(lines)
        )

    # -------------------------------------------------------------------------
    # 3. PRODUCT PRICING: MOST EXPENSIVE vs. CHEAPEST PRODUCT (BY UNIT PRICE!)
    # Matches: "the most expensive product", "cheapest product", "product prices", "how much"
    # -------------------------------------------------------------------------
    asks_expensive = has("expensive", "priciest", "highest price", "most costly", "costliest", "أغلى", "اغلى", "الأغلى", "أعلى سعر")
    asks_cheap = has("cheap", "cheapest", "lowest price", "least expensive", "budget", "أرخص", "ارخص", "الأرخص", "أقل سعر")
    asks_price_list = has("price", "prices", "pricing", "unit price", "سعر", "أسعار", "اسعار", "بكام")

    if asks_expensive or asks_cheap or asks_price_list:
        if asks_cheap and not asks_expensive:
            cat = get_product_price_catalog(sales, ascending=True)
            cheapest = cat.iloc[0]
            if lang == "ar":
                lines = [
                    f"{idx + 1}. **{r.Description}** — **${r.UnitPrice:,.2f}** للوحدة (المبيعات: {r.Quantity:,} قطعة | الإيرادات: ${r.Revenue:,.2f})"
                    for idx, r in enumerate(cat.head(5).itertuples())
                ]
                return (
                    f"**🏷️ أرخص منتج في المتجر هو: {cheapest['Description']} بسعر ${cheapest['UnitPrice']:,.2f} للوحدة**\n"
                    f"- إجمالي القطع المباعة: **{int(cheapest['Quantity']):,} قطعة** عبر {int(cheapest['Orders']):,} طلب (إجمالي الإيرادات: ${cheapest['Revenue']:,.2f}).\n\n"
                    f"**💲 أقل 5 منتجات سعراً في المتجر (حسب سعر الوحدة):**\n" + "\n".join(lines)
                )
            lines = [
                f"{idx + 1}. **{r.Description}** — **${r.UnitPrice:,.2f} / unit** ({r.Quantity:,} units sold | ${r.Revenue:,.2f} total revenue)"
                for idx, r in enumerate(cat.head(5).itertuples())
            ]
            return (
                f"**🏷️ Cheapest Product by Unit Price: {cheapest['Description']} (${cheapest['UnitPrice']:,.2f} per unit)**\n"
                f"- Sold **{int(cheapest['Quantity']):,} units** across {int(cheapest['Orders']):,} orders (${cheapest['Revenue']:,.2f} total revenue).\n\n"
                f"**💲 5 Lowest-Priced Products in Your Catalog (by Unit Price):**\n" + "\n".join(lines)
            )
        else:
            cat = get_product_price_catalog(sales, ascending=False)
            priciest = cat.iloc[0]
            cheapest = cat.iloc[-1]
            if lang == "ar":
                lines = [
                    f"{idx + 1}. **{r.Description}** — **${r.UnitPrice:,.2f}** للوحدة (المبيعات: {r.Quantity:,} قطعة | الإيرادات: ${r.Revenue:,.2f})"
                    for idx, r in enumerate(cat.head(5).itertuples())
                ]
                return (
                    f"**💎 أغلى منتج في المتجر هو: {priciest['Description']} بسعر ${priciest['UnitPrice']:,.2f} للوحدة**\n"
                    f"- حقق إجمالي إيرادات **${priciest['Revenue']:,.2f}** من بيع **{int(priciest['Quantity']):,} قطعة** عبر {int(priciest['Orders']):,} طلب.\n"
                    f"- *(ملاحظة: أرخص منتج هو **{cheapest['Description']}** بسعر **${cheapest['UnitPrice']:,.2f}** للوحدة)*.\n\n"
                    f"**💰 أعلى 5 منتجات سعراً في المتجر (حسب سعر الوحدة Unit Price):**\n" + "\n".join(lines)
                )
            lines = [
                f"{idx + 1}. **{r.Description}** — **${r.UnitPrice:,.2f} / unit** ({r.Quantity:,} units sold | ${r.Revenue:,.2f} total revenue)"
                for idx, r in enumerate(cat.head(5).itertuples())
            ]
            return (
                f"**💎 Most Expensive Product by Unit Price: {priciest['Description']} (${priciest['UnitPrice']:,.2f} per unit)**\n"
                f"- Generated **${priciest['Revenue']:,.2f}** in total revenue from **{int(priciest['Quantity']):,} units sold** across {int(priciest['Orders']):,} orders.\n"
                f"- *(By comparison, your lowest-priced item is **{cheapest['Description']}** at **${cheapest['UnitPrice']:,.2f} / unit**)*.\n\n"
                f"**💰 Top 5 Most Expensive Products in Your Catalog (Ranked by Unit Price):**\n" + "\n".join(lines)
            )

    # -------------------------------------------------------------------------
    # 4. GENERAL CUSTOMER COMPLAINTS & SENTIMENT ANALYSIS
    # -------------------------------------------------------------------------
    if asks_bad_reviews or has("sentiment", "review", "reviews", "feedback", "satisfaction", "آراء", "تقييمات", "المشاعر"):
        total_rev = len(reviews)
        neg_df = reviews[reviews["sentiment"] == "negative"]
        neg_count = len(neg_df)
        pos_s = float((reviews["sentiment"] == "positive").mean() * 100)
        neu_s = float((reviews["sentiment"] == "neutral").mean() * 100)
        neg_s = float((reviews["sentiment"] == "negative").mean() * 100)

        real_issues = extract_real_complaints(reviews, n=6, lang=lang)
        sample_quotes = get_sample_negative_reviews(reviews, n=4, lang=lang)
        prod_neg = get_products_by_sentiment(reviews, target_sentiment="negative", lang=lang)
        worst_prod = prod_neg.iloc[0]

        if lang == "ar":
            issue_lines = [
                f"{idx}. **\"{issue}\"** — تكررت في **{count}** تقييم سلبي ({(count / max(1, neg_count)) * 100:.1f}% من الشكاوى)"
                for idx, (issue, count) in enumerate(real_issues, 1)
            ]
            quote_lines = [f"- 🗣️ *\"{q_txt}\"* — **({prod_name})**" for q_txt, prod_name in sample_quotes]
            return (
                f"**💬 تحليل شكاوى وآراء العملاء الحقيقية ({total_rev:,} تقييم):**\n"
                f"- **توزيع المشاعر:** {pos_s:.1f}% إيجابي | {neu_s:.1f}% محايد | **{neg_s:.1f}% سلبي ({neg_count:,} شكوى)**\n"
                f"- **المنتج الأكثر تعرضاً للشكاوى:** **{worst_prod['Product']}** ({worst_prod['NegativeReviews']} شكوى — {worst_prod['NegativePct']:.1f}%)\n\n"
                f"**🚨 أبرز المشاكل الحقيقية التي يشتكي منها العملاء:**\n"
                + "\n".join(issue_lines)
                + "\n\n**📝 تعليقات حقيقية من العملاء في البيانات:**\n"
                + "\n".join(quote_lines)
            )
        else:
            issue_lines = [
                f"{idx}. **\"{issue}\"** — reported by **{count} customers** ({(count / max(1, neg_count)) * 100:.1f}% of negative reviews)"
                for idx, (issue, count) in enumerate(real_issues, 1)
            ]
            quote_lines = [f"- 🗣️ *\"{q_txt}\"* — **({prod_name})**" for q_txt, prod_name in sample_quotes]
            return (
                f"**💬 Real Customer Complaints & Sentiment Analysis ({total_rev:,} Total Reviews):**\n"
                f"- **Sentiment Breakdown:** {pos_s:.1f}% Positive | {neu_s:.1f}% Neutral | **{neg_s:.1f}% Negative ({neg_count:,} reviews)**\n"
                f"- **Most-Complained Product:** **{worst_prod['Product']}** ({worst_prod['NegativeReviews']} negative reviews — {worst_prod['NegativePct']:.1f}% negative rate)\n\n"
                f"**🚨 Specific Issues Customers Are Complaining About:**\n"
                + "\n".join(issue_lines)
                + "\n\n**📝 Real Verbatim Customer Reviews from Your Dataset:**\n"
                + "\n".join(quote_lines)
            )

    # -------------------------------------------------------------------------
    # 5. RECOMMENDATIONS & GROWTH ADVICE
    # -------------------------------------------------------------------------
    if has(
        "recommend", "recommendation", "recommendations", "advice", "suggest", "suggestion",
        "boost", "improve", "strategy", "action", "solutions",
        "توصي", "توصيات", "توصية", "نصيحة", "نصائح", "اقترح", "أعمل إيه", "اعمل ايه", "تحسين", "زيادة المبيعات",
    ):
        recs = build_recommendations(sales, reviews, future, lang)
        hdr = "**💡 خطة العمل والتوصيات الذكية لمتجرك:**\n" if lang == "ar" else "**💡 Actionable Recommendations Based on Your Store Data:**\n"
        return hdr + "\n".join(f"{i}. {r}" for i, r in enumerate(recs, 1))

    # -------------------------------------------------------------------------
    # 6. SALES FORECAST & FUTURE TRENDS
    # -------------------------------------------------------------------------
    if has(
        "forecast", "predict", "prediction", "future", "next", "upcoming", "trend",
        "increase", "decrease", "will sales", "30 days",
        "متوقع", "توقع", "توقعات", "تزيد", "تقل", "هتزيد", "هتقل", "القادم", "الجاي", "المستقبل",
    ):
        tot = float(future["Forecast"].sum())
        avg_d = float(future["Forecast"].mean())
        g = compute_growth_pct(future)
        peak_row = future.loc[future["Forecast"].idxmax()]
        peak_dt = str(peak_row["Date"])[:10]
        peak_val = float(peak_row["Forecast"])
        if lang == "ar":
            t_str = f"زيادة 📈 (+{g:.1f}%)" if g >= 0 else f"انخفاض 📉 ({g:.1f}%)"
            return (
                f"**📈 ملخص التنبؤ بالمبيعات خلال {len(future)} يوم (Random Forest):**\n"
                f"- **إجمالي الإيرادات المتوقعة:** ${tot:,.2f}\n"
                f"- **المتوسط اليومي المتوقع:** ${avg_d:,.2f}\n"
                f"- **الاتجاه المتوقع:** {t_str}\n"
                f"- **أعلى يوم مبيعات متوقع:** {peak_dt} (${peak_val:,.2f})"
            )
        t_str = f"Increase 📈 (+{g:.1f}%)" if g >= 0 else f"Decrease 📉 ({g:.1f}%)"
        return (
            f"**📈 {len(future)}-Day ML Sales Forecast (Random Forest):**\n"
            f"- **Projected Total Revenue:** ${tot:,.2f}\n"
            f"- **Projected Daily Average:** ${avg_d:,.2f}\n"
            f"- **Expected Trend:** {t_str}\n"
            f"- **Peak Forecast Day:** {peak_dt} (${peak_val:,.2f})"
        )

    # -------------------------------------------------------------------------
    # 7. COUNTRY / REGIONAL BREAKDOWN
    # -------------------------------------------------------------------------
    if has("country", "countries", "region", "regions", "where", "location", "market", "دولة", "الدول", "بلد", "منطقة", "المناطق", "فين"):
        c = A.by_country(sales, 6)
        total_rev = float(sales["Revenue"].sum()) or 1.0
        lines = [
            f"- **{r.Country}**: ${r.Revenue:,.2f} ({(r.Revenue / total_rev) * 100:.1f}% of revenue | {r.Orders:,} orders)"
            for r in c.itertuples()
        ]
        hdr = "**🌍 أعلى المناطق / الدول حسب الإيرادات:**\n" if lang == "ar" else "**🌍 Top Regions / Countries by Revenue:**\n"
        return hdr + "\n".join(lines)

    # -------------------------------------------------------------------------
    # 8. MONTHLY REVENUE BREAKDOWN
    # -------------------------------------------------------------------------
    if has("month", "monthly", "months", "season", "شهر", "شهري", "الشهور", "أفضل شهر", "اشهر"):
        m = A.monthly_sales(sales)
        best_m = m.loc[m["Revenue"].idxmax()]
        lines = [f"- **{str(r.Month)[:7]}**: ${r.Revenue:,.2f} ({r.Orders:,} orders)" for r in m.tail(6).itertuples()]
        if lang == "ar":
            return f"**📅 تحليل الإيرادات الشهرية (أفضل شهر: {str(best_m['Month'])[:7]} بإيرادات ${float(best_m['Revenue']):,.2f}):**\n" + "\n".join(lines)
        return f"**📅 Monthly Revenue Performance (Best Month: {str(best_m['Month'])[:7]} at ${float(best_m['Revenue']):,.2f}):**\n" + "\n".join(lines)

    # -------------------------------------------------------------------------
    # 9. LOWEST / WORST SELLING PRODUCTS
    # -------------------------------------------------------------------------
    if has("worst", "least", "lowest", "slow", "slowest", "bottom", "underperform", "weak", "أقل", "اقل", "الأقل", "اسوأ", "أسوأ", "ضعيف"):
        p = A.top_products(sales, 5, ascending=True)
        price_map = dict(zip(sales["Description"], sales["UnitPrice"]))
        lines = [
            f"{idx + 1}. **{r.Description}** (${float(price_map.get(r.Description, 0)):,.2f}/unit) — **${r.Revenue:,.2f}** total revenue ({r.Quantity:,} units sold | {r.Orders:,} orders)"
            for idx, r in enumerate(p.itertuples())
        ]
        hdr = "**⚠️ أقل المنتجات مبيعاً حسب الإيرادات:**\n" if lang == "ar" else "**⚠️ Lowest-Selling Products by Revenue:**\n"
        return hdr + "\n".join(lines)

    # -------------------------------------------------------------------------
    # 10. TOP / BEST SELLING PRODUCTS
    # -------------------------------------------------------------------------
    if has("top", "best", "most", "highest", "popular", "product", "products", "أكتر", "اكتر", "أكثر", "افضل", "أفضل", "الأعلى", "منتج", "المنتجات"):
        p = A.top_products(sales, 5, ascending=False)
        total_rev = float(sales["Revenue"].sum()) or 1.0
        price_map = dict(zip(sales["Description"], sales["UnitPrice"]))
        lines = [
            f"{idx + 1}. **{r.Description}** (${float(price_map.get(r.Description, 0)):,.2f}/unit) — **${r.Revenue:,.2f}** ({(r.Revenue / total_rev) * 100:.1f}% share | {r.Quantity:,} units | {r.Orders:,} orders)"
            for idx, r in enumerate(p.itertuples())
        ]
        hdr = "**🏆 أعلى المنتجات مبيعاً حسب الإيرادات:**\n" if lang == "ar" else "**🏆 Top-Selling Products by Revenue:**\n"
        return hdr + "\n".join(lines)

    # -------------------------------------------------------------------------
    # 11. TOTAL REVENUE, ORDERS & OVERALL KPIs
    # -------------------------------------------------------------------------
    if has(
        "total", "revenue", "sales", "order", "orders", "average", "aov", "kpi",
        "customer", "customers", "how much", "summary", "overview",
        "إجمالي", "اجمالي", "مبيعات", "إيرادات", "ايرادات", "طلبات", "متوسط", "عملاء", "العملاء", "كم", "ملخص",
    ):
        k = A.kpis(sales)
        cat = get_product_price_catalog(sales, ascending=False)
        priciest = cat.iloc[0]
        if lang == "ar":
            return (
                f"**📊 ملخص أداء المبيعات والمؤشرات الرئيسية:**\n"
                f"- **إجمالي الإيرادات:** ${k['revenue']:,.2f}\n"
                f"- **عدد الطلبات:** {k['orders']:,}\n"
                f"- **عدد العملاء الفعليين:** {k['customers']:,}\n"
                f"- **إجمالي القطع المباعة:** {k['units_sold']:,}\n"
                f"- **متوسط قيمة الطلب (AOV):** ${k['avg_order']:,.2f}\n"
                f"- **أغلى منتج سعراً:** {priciest['Description']} (${priciest['UnitPrice']:,.2f})"
            )
        return (
            f"**📊 Overall Sales & KPI Performance:**\n"
            f"- **Total Revenue:** ${k['revenue']:,.2f}\n"
            f"- **Total Orders:** {k['orders']:,}\n"
            f"- **Unique Customers:** {k['customers']:,}\n"
            f"- **Total Units Sold:** {k['units_sold']:,}\n"
            f"- **Average Order Value (AOV):** ${k['avg_order']:,.2f}\n"
            f"- **Most Expensive Product:** {priciest['Description']} (${priciest['UnitPrice']:,.2f}/unit)"
        )

    if lang == "ar":
        return (
            "يمكنني الإجابة على أسئلتك بالتفصيل من واقع بيانات متجرك! جرّب السؤال عن:\n"
            "- *إيه أكتر منتج عليه تقييمات سلبية؟*\n"
            "- *إيه هو أغلى أو أرخص منتج في المتجر؟*\n"
            "- *العملاء بيشتكوا من إيه؟*\n"
            "- *إيه أكتر أو أقل المنتجات مبيعاً؟*\n"
            "- *كم إجمالي المبيعات ومتوسط الطلب؟*\n"
            "- *هل المبيعات متوقع تزيد ولا تقل الشهر القادم؟*"
        )
    return (
        "I can answer questions in detail using your real store data! Try asking:\n"
        "- *Which product has the most bad reviews?*\n"
        "- *What is the most expensive or cheapest product?*\n"
        "- *What are customers complaining about the most?*\n"
        "- *What are the top selling or lowest selling products?*\n"
        "- *What is our total revenue and average order value?*\n"
        "- *Will sales increase or decrease in the next 30 days?*"
    )


st.set_page_config(
    page_title="AI Business Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar configuration
st.sidebar.title("⚙️ AI Business Assistant")
lang_choice = st.sidebar.radio("Language / اللغة", ["English", "العربية"], index=0)
lang = "ar" if lang_choice == "العربية" else "en"
t = UI_TEXT[lang]

if lang == "ar":
    st.markdown(
        """
        <style>
        .stApp { direction: rtl; text-align: right; }
        div[data-testid="stMetricValue"] { direction: ltr; text-align: right; }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.sidebar.markdown("---")
sales_file = st.sidebar.file_uploader(t["upload_sales"], type=["csv", "xlsx"])
rev_file = st.sidebar.file_uploader(t["upload_rev"], type=["csv"])

if sales_file is None and rev_file is None:
    st.sidebar.info(t["data_source_info"])


@st.cache_data(show_spinner=False)
def get_data(sf, rf):
    s_df = load_sales(sf)
    r_df = load_reviews(rf)
    r_df = attach_products_to_reviews(s_df, r_df, is_custom_rev=(rf is not None))
    return s_df, r_df


@st.cache_resource(show_spinner=False)
def get_models(_sales, _reviews, key):
    rf_model, rf_res, rf_comp, feat_df = train_forecaster(_sales, test_days=60, save_model=True)
    feat_cols = [c for c in feat_df.columns if c != "y"]
    feat_imp = feature_importance_df(rf_model, feat_cols)
    sent_model, sent_res = train_sentiment(_reviews, save_model=True)
    return rf_model, rf_res, rf_comp, feat_imp, sent_model, sent_res


with st.spinner(t["loading"]):
    sales, reviews = get_data(sales_file, rev_file)
    cache_key = (
        len(sales),
        len(reviews),
        getattr(sales_file, "name", "builtin_sales"),
        getattr(rev_file, "name", "builtin_reviews"),
    )
    rf_model, rf_res, rf_comp, feat_imp, sent_model, sent_res = get_models(sales, reviews, cache_key)
    fut_default = forecast_future(rf_model, sales, days=30)

# Pre-compute core summary metrics used across the Chatbot MVP and Analytics tabs
k = A.kpis(sales)
top_prod_df = A.top_products(sales, n=1, ascending=False)
top_prod_name = top_prod_df.iloc[0]["Description"] if not top_prod_df.empty else "N/A"
top_prod_rev = float(top_prod_df.iloc[0]["Revenue"]) if not top_prod_df.empty else 0.0
fut_30_total = float(fut_default["Forecast"].sum())
growth_30_pct = compute_growth_pct(fut_default)
trend_label = t["up"].format(growth_30_pct) if growth_30_pct >= 0 else t["down"].format(growth_30_pct)
pos_pct = float((reviews["sentiment"] == "positive").mean() * 100)
neg_pct = float((reviews["sentiment"] == "negative").mean() * 100)

# Header
st.title("💼 " + t["title"])
st.caption(t["sub"])

# Navigation Tabs: Tab 0 is 🤖 Business Chatbot (MVP), followed by the Power BI Analytics tabs
tabs = st.tabs(t["tabs"])

# ================ TAB 0: SMART BUSINESS CHATBOT (MASTER MVP) ================
with tabs[0]:
    # Live Executive Pulse Strip
    p1, p2, p3, p4 = st.columns(4)
    p1.metric(t["pulse_rev"], f"${k['revenue']:,.2f}", f"{k['orders']:,} orders")
    p2.metric(t["pulse_fut"], f"${fut_30_total:,.2f}", f"{growth_30_pct:+.1f}% trend")
    p3.metric(t["pulse_neg"], f"{neg_pct:.1f}%", f"{len(reviews):,} reviews")
    p4.metric(t["pulse_top"], top_prod_name[:24], f"${top_prod_rev:,.0f}")

    st.markdown("---")

    # Store chat turns as a list of (question, answer) pairs with NEWEST at index 0!
    state_signature = (cache_key, lang)
    if "chat_turns" not in st.session_state or st.session_state.get("chat_sig") != state_signature:
        st.session_state.chat_sig = state_signature
        st.session_state.chat_turns = []

    hdr_col1, hdr_col2 = st.columns([4, 1])
    hdr_col1.markdown(f"**{t['quick_q']}**")
    if hdr_col2.button(t["clear_chat"], use_container_width=True):
        st.session_state.chat_turns = []
        st.rerun()

    triggered_q = None
    sample_qs = t["sample_questions"]
    row1_cols = st.columns(3)
    for idx in range(3):
        if row1_cols[idx].button(sample_qs[idx], key=f"q_btn_{idx}", use_container_width=True):
            triggered_q = sample_qs[idx]

    row2_cols = st.columns(3)
    for idx in range(3, 6):
        if row2_cols[idx - 3].button(sample_qs[idx], key=f"q_btn_{idx}", use_container_width=True):
            triggered_q = sample_qs[idx]

    with st.form(key="chat_form", clear_on_submit=True):
        in_col1, in_col2 = st.columns([5, 1])
        typed_q = in_col1.text_input(
            t["ask"],
            placeholder=t["ask"],
            label_visibility="collapsed",
        )
        send_clicked = in_col2.form_submit_button(
            "Send 🚀" if lang == "en" else "إرسال 🚀",
            type="primary",
            use_container_width=True,
        )

    active_q = triggered_q or (typed_q.strip() if send_clicked and typed_q.strip() else None)
    if active_q:
        bot_reply = smart_chatbot_answer(active_q, sales, reviews, fut_default, lang=lang)
        # Insert at index 0 so the newest reply ALWAYS appears at the very TOP!
        st.session_state.chat_turns.insert(0, (active_q, bot_reply))

    # Render NEWEST reply right at the top so the user NEVER has to scroll down!
    if st.session_state.chat_turns:
        latest_q, latest_a = st.session_state.chat_turns[0]
        st.markdown(f"#### {t['latest_reply_badge']}")
        with st.container(border=True):
            st.chat_message("user").markdown(f"**{latest_q}**")
            st.chat_message("assistant").markdown(latest_a)

        if len(st.session_state.chat_turns) > 1:
            with st.expander(f"{t['earlier_chats']} ({len(st.session_state.chat_turns) - 1})", expanded=False):
                for prev_q, prev_a in st.session_state.chat_turns[1:]:
                    st.chat_message("user").markdown(prev_q)
                    st.chat_message("assistant").markdown(prev_a)
                    st.markdown("---")
    else:
        if lang == "ar":
            welcome_msg = (
                f"👋 **أهلاً بك! أنا المساعد الذكي لمتجرك.** قمت بتحليل **{k['orders']:,} طلب** (**${k['revenue']:,.2f}**) "
                f"و **{len(reviews):,} تقييم من العملاء**. اضغط على أي سؤال بالأعلى أو اكتب سؤالك بالأسفل."
            )
        else:
            welcome_msg = (
                f"👋 **Welcome! I have analyzed {k['orders']:,} orders (${k['revenue']:,.2f}) and {len(reviews):,} customer reviews.** "
                f"Click any question above or type your question below."
            )
        st.info(welcome_msg)

# ================ TAB 1: SALES ANALYTICS (POWER BI VIEW) ================
with tabs[1]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t["revenue"], f"${k['revenue']:,.2f}")
    c2.metric(t["orders"], f"{k['orders']:,}")
    c3.metric(t["customers"], f"{k['customers']:,}")
    c4.metric(t["avg"], f"${k['avg_order']:,.2f}")

    fig_monthly = px.line(
        A.monthly_sales(sales),
        x="Month",
        y="Revenue",
        title=t["monthly"],
        markers=True,
        template="plotly_white",
    )
    fig_monthly.update_traces(line_color="#2563EB", line_width=3)
    st.plotly_chart(fig_monthly, use_container_width=True)

    col_a, col_b = st.columns(2)
    top_df = A.top_products(sales, n=8, ascending=False).sort_values("Revenue", ascending=True)
    low_df = A.top_products(sales, n=8, ascending=True).sort_values("Revenue", ascending=False)

    fig_top = px.bar(
        top_df,
        x="Revenue",
        y="Description",
        orientation="h",
        title=t["top"],
        color="Revenue",
        color_continuous_scale="Blues",
        template="plotly_white",
    )
    col_a.plotly_chart(fig_top, use_container_width=True)

    fig_low = px.bar(
        low_df,
        x="Revenue",
        y="Description",
        orientation="h",
        title=t["low"],
        color="Revenue",
        color_continuous_scale="Reds",
        template="plotly_white",
    )
    col_b.plotly_chart(fig_low, use_container_width=True)

    fig_country = px.pie(
        A.by_country(sales, n=8),
        names="Country",
        values="Revenue",
        title=t["country"],
        hole=0.4,
        template="plotly_white",
    )
    st.plotly_chart(fig_country, use_container_width=True)

# ================ TAB 2: SALES FORECASTING (ML VIEW) ================
with tabs[2]:
    horizon = st.slider(t["forecast_days"], min_value=14, max_value=90, value=30, step=7)
    fut = fut_default if horizon == 30 else forecast_future(rf_model, sales, days=horizon)
    growth_pct = compute_growth_pct(fut)

    fc1, fc2, fc3 = st.columns(3)
    fc1.metric(t["forecast_total"].format(horizon), f"${fut['Forecast'].sum():,.2f}")
    fc2.metric(t["forecast_avg"], f"${fut['Forecast'].mean():,.2f}")
    fc3.metric(
        t["forecast_trend"],
        t["up"].format(growth_pct) if growth_pct >= 0 else t["down"].format(growth_pct),
    )

    col_m1, col_m2 = st.columns([1, 1])
    with col_m1:
        st.subheader(t["metrics"])
        st.dataframe(pd.DataFrame(rf_res).T.style.format("{:,.2f}"), use_container_width=True)
    with col_m2:
        fig_imp = px.bar(
            feat_imp,
            x="Importance",
            y="Feature",
            orientation="h",
            title=t["feat_imp"],
            template="plotly_white",
        )
        fig_imp.update_traces(marker_color="#0D9488")
        st.plotly_chart(fig_imp, use_container_width=True)

    fig_avsp = px.line(
        rf_comp,
        title=t["avsp"],
        template="plotly_white",
    )
    st.plotly_chart(fig_avsp, use_container_width=True)

    fig_fut = px.line(
        fut,
        x="Date",
        y="Forecast",
        title=f"{t['future']} ({horizon} Days)",
        markers=True,
        template="plotly_white",
    )
    fig_fut.update_traces(line_color="#16A34A", line_width=2.5)
    st.plotly_chart(fig_fut, use_container_width=True)

# ================ TAB 3: SENTIMENT ANALYSIS (NLP VIEW) ================
with tabs[3]:
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric(t["acc"], f"{sent_res['accuracy'] * 100:.1f}%")
    sc2.metric(t["f1"], f"{sent_res['f1_macro'] * 100:.1f}%")
    sc3.metric(t["total_rev_count"], f"{len(reviews):,}")

    sa, sb = st.columns(2)
    fig_dist = px.pie(
        reviews,
        names="sentiment",
        title=t["dist"],
        hole=0.4,
        color="sentiment",
        color_discrete_map={"positive": "#16A34A", "neutral": "#64748B", "negative": "#DC2626"},
        template="plotly_white",
    )
    sa.plotly_chart(fig_dist, use_container_width=True)

    fig_cm = px.imshow(
        sent_res["confusion"],
        x=sent_res["labels"],
        y=sent_res["labels"],
        text_auto=True,
        color_continuous_scale="Blues",
        title=t["conf_mat"],
    )
    sb.plotly_chart(fig_cm, use_container_width=True)

    st.subheader(t["complaints"])
    comp_df = pd.DataFrame(
        extract_real_complaints(reviews, n=8, lang=lang),
        columns=["Customer Complaint Statement", "Reviews Mentioning This Issue"],
    )
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader(t["try"])
    default_txt = (
        "التوصيل متأخر جداً والجودة سيئة"
        if lang == "ar"
        else "The delivery was very late and the build quality feels poor."
    )
    user_txt = st.text_input(t["try"], default_txt, label_visibility="collapsed")
    if st.button(t["predict"], type="primary"):
        pred_out = predict_sentiment(sent_model, user_txt)
        lbl = pred_out["sentiment"].upper()
        conf = pred_out["confidence"]
        if pred_out["sentiment"] == "positive":
            st.success(f"**{t['pred_result']}:** {lbl} ✅  |  **{t['confidence']}:** {conf:.1f}%")
        elif pred_out["sentiment"] == "negative":
            st.error(f"**{t['pred_result']}:** {lbl} ⚠️  |  **{t['confidence']}:** {conf:.1f}%")
        else:
            st.info(f"**{t['pred_result']}:** {lbl} ➖  |  **{t['confidence']}:** {conf:.1f}%")

# ================ TAB 4: SMART RECOMMENDATIONS ================
with tabs[4]:
    recs = build_recommendations(sales, reviews, fut_default, lang=lang)
    for idx, r in enumerate(recs, start=1):
        st.info(f"**{idx}.** {r}")