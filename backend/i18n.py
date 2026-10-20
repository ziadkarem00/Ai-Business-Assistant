"""Bilingual Translations (English & Arabic) for AI Business Assistant."""

T = {
    "en": {
        "title": "AI Business Assistant",
        "sub": "AI-Powered Business Chatbot, Sales Analytics, ML Forecasting & Sentiment Intelligence",
        "lang": "Language",
        "upload_sales": "Upload Sales File (CSV/XLSX) [Optional]",
        "upload_rev": "Upload Customer Reviews (CSV) [Optional]",
        "data_source_info": "Using built-in realistic store dataset (upload a file above to analyze your own data).",
        "tabs": [
            "🤖 Business Chatbot (MVP)",
            "📊 Sales Analytics",
            "📈 Sales Forecasting",
            "💬 Sentiment Analysis",
            "💡 Recommendations",
        ],
        "chat_hero_title": "🤖 Smart AI Business Assistant",
        "chat_hero_sub": "Ask anything about your store's sales, products, 30-day ML forecast, customer complaints, or growth recommendations.",
        "chat_welcome": (
            "👋 **Welcome! I am your AI Business Assistant.**\n\n"
            "I have analyzed **{orders:,} orders** (**${rev:,.2f}** total revenue across **{products}** products) "
            "and **{reviews:,} customer reviews**.\n"
            "- **Top Product:** {top_prod} (${top_rev:,.2f})\n"
            "- **Next 30-Day Forecast:** ${fut_tot:,.2f} ({trend})\n"
            "- **Customer Satisfaction:** {pos_pct:.1f}% Positive | {neg_pct:.1f}% Negative\n\n"
            "*Click any quick question above or type your own question below!*"
        ),
        "clear_chat": "🗑️ Reset Conversation",
        "pulse_rev": "Total Revenue",
        "pulse_fut": "30-Day Forecast",
        "pulse_neg": "Negative Reviews",
        "pulse_top": "Best Seller",
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
        "complaints": "Most Frequent Customer Complaints (Negative Bigrams)",
        "try": "Test Live Sentiment Analysis (English or Arabic)",
        "predict": "Analyze Sentiment",
        "pred_result": "Predicted Sentiment",
        "confidence": "Model Confidence",
        "ask": "Ask your AI Business Assistant about sales, products, forecasts, complaints, or recommendations...",
        "quick_q": "⚡ Quick Questions to Ask the Assistant:",
        "sample_questions": [
            "What are customers complaining about the most?",
            "What are the top selling products?",
            "What are the lowest selling products?",
            "What is our total revenue and average order value?",
            "Will sales increase or decrease in the next 30 days?",
            "Give me recommendations to boost sales",
        ],
        "loading": "Loading data & initializing AI models...",
        "ans_top": "**🏆 Top-Selling Products by Revenue:**",
        "ans_low": "**⚠️ Lowest-Selling Products (Candidates for Promotion/Review):**",
        "ans_total": (
            "**📊 Overall Sales & KPI Performance:**\n"
            "- **Total Revenue:** ${:,.2f}\n"
            "- **Total Orders:** {:,}\n"
            "- **Unique Customers:** {:,}\n"
            "- **Total Units Sold:** {:,}\n"
            "- **Average Order Value (AOV):** ${:,.2f}"
        ),
        "ans_noforecast": "Forecast data is currently unavailable.",
        "up": "Increase 📈 (+{:.1f}%)",
        "down": "Decrease 📉 ({:.1f}%)",
        "ans_forecast": (
            "**📈 {}-Day ML Sales Forecast (Random Forest):**\n"
            "- **Projected Total Revenue:** ${:,.2f}\n"
            "- **Projected Daily Average:** ${:,.2f}\n"
            "- **Expected Trend:** {}\n"
            "- **Peak Forecast Day:** {} (${:,.2f})"
        ),
        "ans_complaints": (
            "**💬 Customer Sentiment & Top Complaints Analysis:**\n"
            "- **Sentiment Breakdown:** {:.1f}% Positive | {:.1f}% Neutral | {:.1f}% Negative\n"
            "- **Most Frequent Customer Complaints:**"
        ),
        "ans_country": "**🌍 Top Regions / Countries by Revenue:**",
        "ans_monthly": (
            "**📅 Monthly Revenue Performance:**\n"
            "- **Best Performing Month:** {} (${:,.2f} across {:,} orders)\n"
            "- **Latest Month ({}):** ${:,.2f} ({:,} orders)\n"
            "- **Recent Monthly Breakdown:**"
        ),
        "ans_product_detail": (
            "**📦 Product Performance — {}:**\n"
            "- **Total Revenue:** ${:,.2f} ({:.1f}% of store revenue)\n"
            "- **Units Sold:** {:,} units across {:,} orders\n"
            "- **Rank:** #{} out of {} products"
        ),
        "ans_unknown": (
            "I can answer questions using your real store data! Try asking:\n"
            "- *What are customers complaining about the most?*\n"
            "- *What are the top selling or lowest selling products?*\n"
            "- *What is our total revenue and average order value?*\n"
            "- *Will sales increase or decrease in the next 30 days?*\n"
            "- *Which country or month has the highest sales?*\n"
            "- *Give me recommendations to boost sales*"
        ),
        "rec_low": "**Optimize Low Performers:** Review pricing, bundle offers, or phase out slow-moving items ({}).",
        "rec_complaint": "**Resolve Customer Pain Points:** Prioritize fixing the top customer complaints reported in negative reviews ({}).",
        "rec_up": "**Prepare for Demand Growth:** Sales are forecasted to grow ({:+.1f}%) over the next 30 days — increase safety stock for fast-moving items.",
        "rec_down": "**Counter Projected Slowdown:** Sales are forecasted to dip ({:+.1f}%) — launch targeted promotional campaigns and discount bundles.",
        "rec_stock": "**Protect Core Revenue Driver:** '{}' generates your highest revenue (${:,.0f}) — maintain strict inventory replenishment to prevent stockouts.",
        "rec_aov": "**Boost Average Order Value (Current: ${:,.2f}):** Introduce free-shipping thresholds or cross-sell accessories at checkout.",
    },
    "ar": {
        "title": "المساعد الذكي للأعمال (AI Business Assistant)",
        "sub": "مساعد ذكي تفاعلي، تحليل المبيعات، التنبؤ بالذكاء الاصطناعي، وتحليل آراء العملاء",
        "lang": "اللغة",
        "upload_sales": "ارفع ملف المبيعات (CSV/XLSX) [اختياري]",
        "upload_rev": "ارفع ملف تقييمات العملاء (CSV) [اختياري]",
        "data_source_info": "يتم حالياً استخدام بيانات المتجر الافتراضية المدمجة (يمكنك رفع ملفاتك الخاصة من القائمة الجانبية).",
        "tabs": [
            "🤖 المساعد الذكي (الرئيسية)",
            "📊 تحليل المبيعات",
            "📈 التنبؤ بالمبيعات",
            "💬 تحليل المشاعر",
            "💡 التوصيات الذكية",
        ],
        "chat_hero_title": "🤖 المساعد الذكي لإدارة الأعمال",
        "chat_hero_sub": "اسأل مباشرة عن المبيعات، المنتجات، توقعات الـ 30 يوماً القادمة، شكاوى العملاء، أو توصيات زيادة الأرباح.",
        "chat_welcome": (
            "👋 **أهلاً بك! أنا المساعد الذكي لمتجرك.**\n\n"
            "قمت بتحليل **{orders:,} طلب** بإجمالي إيرادات **${rev:,.2f}** (عبر **{products}** منتج) "
            "بالإضافة إلى **{reviews:,} تقييم من العملاء**.\n"
            "- **المنتج الأعلى مبيعاً:** {top_prod} (${top_rev:,.2f})\n"
            "- **توقعات الـ 30 يوماً القادمة:** ${fut_tot:,.2f} ({trend})\n"
            "- **رضا العملاء:** {pos_pct:.1f}% إيجابي | {neg_pct:.1f}% سلبي\n\n"
            "*اضغط على أي سؤال سريع بالأعلى أو اكتب سؤالك بالأسفل!*"
        ),
        "clear_chat": "🗑️ مسح المحادثة",
        "pulse_rev": "إجمالي الإيرادات",
        "pulse_fut": "توقع 30 يوم",
        "pulse_neg": "التقييمات السلبية",
        "pulse_top": "الأعلى مبيعاً",
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
        "complaints": "أكثر شكاوى العملاء تكراراً",
        "try": "جرّب تحليل المشاعر المباشر (بالعربية أو الإنجليزية)",
        "predict": "تحليل الرأي",
        "pred_result": "التصنيف المتوقع",
        "confidence": "نسبة ثقة النموذج",
        "ask": "اسأل المساعد الذكي عن المبيعات، المنتجات، التوقعات، شكاوى العملاء، أو التوصيات...",
        "quick_q": "⚡ أسئلة سريعة لتجربة المساعد الذكي:",
        "sample_questions": [
            "العملاء بيشتكوا من إيه؟",
            "إيه أكتر المنتجات مبيعاً؟",
            "إيه أقل المنتجات مبيعاً؟",
            "كم إجمالي المبيعات ومتوسط الطلب؟",
            "هل المبيعات متوقع تزيد ولا تقل الشهر القادم؟",
            "أعطني توصيات لزيادة المبيعات",
        ],
        "loading": "جاري تجهيز البيانات وتدريب نماذج الذكاء الاصطناعي...",
        "ans_top": "**🏆 أعلى المنتجات مبيعاً حسب الإيرادات:**",
        "ans_low": "**⚠️ أقل المنتجات مبيعاً (تحتاج لمراجعة التسعير أو العروض):**",
        "ans_total": (
            "**📊 ملخص أداء المبيعات والمؤشرات الرئيسية:**\n"
            "- **إجمالي الإيرادات:** ${:,.2f}\n"
            "- **عدد الطلبات:** {:,}\n"
            "- **عدد العملاء:** {:,}\n"
            "- **إجمالي القطع المباعة:** {:,}\n"
            "- **متوسط قيمة الطلب (AOV):** ${:,.2f}"
        ),
        "ans_noforecast": "بيانات التنبؤ غير متاحة حالياً.",
        "up": "زيادة 📈 (+{:.1f}%)",
        "down": "انخفاض 📉 ({:.1f}%)",
        "ans_forecast": (
            "**📈 ملخص التنبؤ بالمبيعات خلال {} يوم (Random Forest):**\n"
            "- **إجمالي الإيرادات المتوقعة:** ${:,.2f}\n"
            "- **المتوسط اليومي المتوقع:** ${:,.2f}\n"
            "- **الاتجاه المتوقع:** {}\n"
            "- **أعلى يوم متوقع:** {} (${:,.2f})"
        ),
        "ans_complaints": (
            "**💬 ملخص تحليل مشاعر وشكاوى العملاء:**\n"
            "- **توزيع التقييمات:** {:.1f}% إيجابي | {:.1f}% محايد | {:.1f}% سلبي\n"
            "- **أكثر الشكاوى والمشاكل تكراراً:**"
        ),
        "ans_country": "**🌍 أعلى المناطق / الدول حسب الإيرادات:**",
        "ans_monthly": (
            "**📅 تحليل الإيرادات الشهرية:**\n"
            "- **أفضل شهر في المبيعات:** {} (${:,.2f} عبر {:,} طلب)\n"
            "- **الشهر الأخير ({}):** ${:,.2f} ({:,} طلب)\n"
            "- **تفاصيل الأشهر الأخيرة:**"
        ),
        "ans_product_detail": (
            "**📦 تحليل أداء المنتج — {}:**\n"
            "- **إجمالي الإيرادات:** ${:,.2f} ({:.1f}% من إجمالي المتجر)\n"
            "- **الكمية المباعة:** {:,} قطعة عبر {:,} طلب\n"
            "- **الترتيب:** رقم #{} من إجمالي {} منتج"
        ),
        "ans_unknown": (
            "يمكنني الإجابة على أسئلتك بناءً على بيانات متجرك الفعلية! جرّب السؤال عن:\n"
            "- *العملاء بيشتكوا من إيه؟*\n"
            "- *إيه أكتر أو أقل المنتجات مبيعاً؟*\n"
            "- *كم إجمالي المبيعات ومتوسط الطلب؟*\n"
            "- *هل المبيعات متوقع تزيد ولا تقل الشهر القادم؟*\n"
            "- *إيه أكتر دولة أو شهر حقق مبيعات؟*\n"
            "- *أعطني توصيات لزيادة المبيعات*"
        ),
        "rec_low": "**تحسين المنتجات الأقل مبيعاً:** راجع التسعير أو اعمل عروض ترويجية للمنتجات بطيئة الحركة ({}).",
        "rec_complaint": "**معالجة شكاوى العملاء:** أعطِ أولوية لحل أكثر المشاكل تكراراً في التقييمات السلبية ({}).",
        "rec_up": "**الاستعداد لنمو الطلب:** من المتوقع نمو المبيعات بنسبة ({:+.1f}%) خلال الفترة القادمة — زوّد المخزون للمنتجات الأكثر طلباً.",
        "rec_down": "**تنشيط المبيعات:** من المتوقع تراجع طفيف ({:+.1f}%) — خطط لحملات تسويقية وخصومات لتحفيز الشراء.",
        "rec_stock": "**حماية المنتج الرئيسي:** المنتج '{}' يحقق أعلى إيرادات (${:,.0f}) — احرص على توفره دائماً لتجنب نفاد المخزون.",
        "rec_aov": "**رفع متوسط سلة الشراء (الحالي: ${:,.2f}):** قدّم شحن مجاني عند تجاوز مبلغ معين أو عروض المنتجات المكملة.",
    },
}
