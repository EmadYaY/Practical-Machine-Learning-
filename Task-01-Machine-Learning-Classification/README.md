## Reference Report

The following report is provided as a reference for this assignment:

[View the Reference Report (Google Drive)](https://drive.google.com/file/d/1_TCKn9jrwUNjK58ZOlgNBC9UiMYBoPho/view)

-----------------
# Task-01: تشخیص وب‌سایت‌های فیشینگ (KNN در برابر Decision Tree)

دسته‌بندی دودویی روی داده‌ی UCI Phishing Websites (فایل `phishing.csv`: ۱۱٬۰۵۴ نمونه، ۳۰ ویژگی با مقادیر {-1, 0, 1}). برچسب طوری نگاشت شده که `y = 1` یعنی فیشینگ. هایپرپارامترهای KNN و Decision Tree فقط با 5-fold CV روی داده‌ی آموزش انتخاب می‌شوند و مجموعه‌ی تست یک بار باز می‌شود. MLP فقط برای رسم منحنی loss و accuracy در هر epoch اضافه شده است (KNN و درخت epoch ندارند).

منبع داده: R. Mohammad and L. McCluskey, UCI Machine Learning Repository, DOI 10.24432/C51W2X (CC BY 4.0).

## دو پیاده‌سازی

پروتکل هر دو یکی است: ادغام نمونه‌های تکراری، تقسیم طبقه‌ای ۶۰/۲۰/۲۰، انتخاب مدل فقط با CV، و باز شدن تست فقط یک بار.

| | نسخه‌ی چندمرحله‌ای (`step01` تا `step07` + `common.py`) | نسخه‌ی فشرده (`ml_task01.py`) |
|---|---|---|
| اجرا | یک اسکریپت برای هر مرحله، به ترتیب | یک دستور، چند دقیقه |
| انتقال بین مراحل | فایل‌های CSV/JSON در `outputs/data` | ندارد (یک فرایند) |
| پوشه‌ی خروجی | `outputs/` | `outputs_compact/` |
| EDA (همبستگی، اطلاعات متقابل، heatmap سطوح) | دارد | فقط توزیع کلاس‌ها |
| تکراری‌ها و نشت داده | مقایسه‌ی raw و dedup در baseline، پروتکل‌ها و جاروی K | نرخ نشت + جدول KNN روی raw و dedup |
| آزمایش‌های KNN | جاروی K (raw و dedup)، ۴ metric از جمله Chebyshev، وزن‌دهی، scaling، انتخاب ویژگی، ۴ بازنمایی، گرید ۱۶۲ تایی، منحنی یادگیری | جاروی K (dedup)، ۴ بازنمایی، گرید ۱۶۲ تایی (Euclidean / Manhattan / Hamming با وزن uniform / distance) |
| آزمایش‌های Decision Tree | baseline پیش‌فرض و سبک کلاس، جاروی عمق، گرید criterion × عمق × حداقل برگ، pruning، بازنمایی‌ها، اهمیت ویژگی، منحنی یادگیری، خروجی `.dot` | جاروی عمق، pruning، ۴ بازنمایی، عمق‌های ۳/۴/۱۰/نامحدود با ماتریس درهم‌ریختگی، رسم درخت عمق ۴ |
| مدل‌های مرحله‌ی تست | ۷ مدل (KNN baseline، KNN best، DT پیش‌فرض، DT سبک کلاس، DT نهایی، DT یک‌انحراف‌معیار، MLP) | ۵ مدل (KNN baseline، KNN best، DT سبک کلاس، DT نهایی، MLP) |
| معیارهای تست | Acc، P، R، F1، macro-F1، MCC، ROC-AUC، PR-AUC، BCE، Brier؛ نقاط کاری | Acc، P، R، F1، ROC-AUC، BCE؛ خطای train و test |
| عدم قطعیت و آزمون‌ها | بوت‌استرپ (Acc، F1، AUC)، McNemar (۶ جفت)، leave-one-out در برابر CV و تست | بوت‌استرپ (Acc، F1)، McNemar (۳ جفت) |
| اهمیت ویژگی | impurity و permutation گروهی (DT و KNN) و توافق رتبه‌ها | ندارد |
| لاگ | یک لاگ برای هر مرحله با classification report و ماتریس درهم‌ریختگی کامل | یک `run.log` |
| جدول‌های LaTeX | همه‌ی جدول‌های گزارش (`outputs/tex`) | جدول‌های مورد استفاده در گزارش (`outputs_compact/tex`) |

**مزیت نسخه‌ی چندمرحله‌ای:** هر مرحله جداگانه قابل اجراست (مثلاً فقط مرحله‌ی درخت)، چون نتایج قبلی ذخیره می‌شود؛ هر آزمایش لاگ، جدول و شکل خودش را دارد؛ و آزمایش‌هایی را پوشش می‌دهد که نسخه‌ی فشرده ندارد (scaling، انتخاب ویژگی، Chebyshev، منحنی یادگیری، اهمیت ویژگی، PR، نقاط کاری، leave-one-out).

**مزیت نسخه‌ی فشرده:** یک دستور همه‌ی اعداد گزارش را بازتولید می‌کند، به فایل‌های مرحله‌ی قبل وابسته نیست و توابع مشترک فقط یک بار تعریف شده‌اند.

برای مدل‌های مشترک، نتایج تست هر دو نسخه یکسان است (جدول پایین).

## ساختار خروجی (نسخه‌ی فشرده)

```
ml_task01.py
phishing.csv
outputs_compact/
  run.log                 # پیکربندی‌های انتخاب‌شده و معیارهای تست
  figures/                # fig01 ... fig08 (.png و .pdf، ۳۰۰ dpi)
  tables/                 # هر جدول به صورت .csv + mlp_history.csv + test_predictions.csv
  tex/                    # همان جدول‌ها به صورت tabular لاتک (بدون caption)
```

| شکل | محتوا |
|---|---|
| `fig01_class_distribution` | تعداد کلاس‌ها در داده‌ی اصلی |
| `fig02_knn_vs_k` | دقت KNN (train / CV / validation) و BCE بر حسب K |
| `fig03_dt_vs_depth` | دقت درخت بر حسب عمق (gini و entropy) |
| `fig04_dt_depth_confusion` | ماتریس درهم‌ریختگی validation برای عمق ۳، ۴، ۱۰ و نامحدود |
| `fig05_tree_depth4` | درخت entropy با عمق حداکثر ۴ |
| `fig06_mlp_curves` | BCE و accuracy در هر epoch برای MLP |
| `fig07_test_confusion` | ماتریس درهم‌ریختگی تست همه‌ی مدل‌ها |
| `fig08_test_roc` | منحنی ROC تست (کامل و بزرگ‌نمایی‌شده) |

جدول‌ها در `tables/` و `tex/`: `data_sample`، `split_summary`، `leakage_effect`، `knn_k_sweep`، `knn_representations`، `knn_grid_top10`، `knn_factors`، `dt_representations`، `dt_depth_compare`، `test_metrics`، `test_bootstrap_ci`، `mcnemar`، `hyperparameters`.

## توضیح کد، بلاک به بلاک (`ml_task01.py`)

**۰. تنظیمات و توابع کمکی**
ثابت‌ها (seed، شیء CV پنج‌لایه‌ی طبقه‌ای، نام بازنمایی‌ها، لیست ۸ ویژگی سه‌مقداری) و چهار تابع اصلی:
- `tex(name, df)`: جدول را هم CSV و هم tabular لاتک ذخیره می‌کند.
- `make_rep(mode, ref)`: بازنمایی سطوح را می‌سازد (`ordinal`، `onehot`، `merge0_neg`، `merge0_pos`). تبدیل روی train یاد گرفته و روی بقیه اعمال می‌شود.
- `evaluate(est, A, y_tr, B, y_va)`: میانگین و انحراف CV روی train و دقت validation را برمی‌گرداند.
- `bce`: لگاریتم‌خطا با احتمال‌های clip شده، چون برگ‌های خالص درخت احتمال ۰ یا ۱ می‌دهند.

**۱. داده**
ستون `class` به `y` نگاشت می‌شود (فیشینگ = ۱). بردارهای ویژگی تکراری با رأی اکثریت ادغام می‌شوند و تساوی‌ها حذف می‌شوند (۵۷۷۲ نمونه‌ی یکتا). تقسیم ۶۰/۲۰/۲۰ طبقه‌ای هم روی داده‌ی raw و هم dedup انجام می‌شود و نرخ نشت (سهم نمونه‌های validation/test که بردار مشابهشان در train هست) اندازه گرفته می‌شود.

**۲. KNN**
- جاروی K با دقت و BCE هموار‌شده (Laplace).
- اثر نشت: همان جاروی K روی تقسیم raw و dedup (جدول `leakage_effect`).
- مقایسه‌ی چهار بازنمایی.
- گرید مشترک روی سه بازنمایی برتر × metric × وزن × K؛ انتخاب فقط با میانگین CV (تساوی: انحراف کمتر، سپس K بزرگ‌تر).

**۳. Decision Tree**
جاروی عمق برای gini و entropy؛ pruning با `ccp_alpha`؛ بهترین از میان آن‌ها با CV (تساوی: برگ کمتر)؛ انتخاب بازنمایی (تساوی: ordinal)؛ مقایسه‌ی عمق‌های ۳/۴/۱۰/نامحدود با ماتریس درهم‌ریختگی؛ رسم درخت عمق ۴.

**۴. MLP**
آموزش با `partial_fit` روی ویژگی‌های one-hot و ثبت BCE و accuracy روی train و validation در هر epoch. epochی که کمترین BCE روی validation را دارد نگه داشته می‌شود (۴۵).

**۵. ارزیابی نهایی روی تست**
پنج مدل روی train+validation دوباره آموزش می‌بینند و تست یک بار امتیازدهی می‌شود. خروجی: جدول معیارها، بوت‌استرپ (۲۰۰۰ نمونه‌گیری)، آزمون دقیق McNemar، ماتریس‌های درهم‌ریختگی، ROC و جدول هایپرپارامترها.

## نتایج تست (در هر دو نسخه یکسان)

| مدل | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| KNN baseline (k=1، Euclidean، ordinal) | 0.9186 | 0.9372 | 0.9025 | 0.9195 | 0.9191 |
| KNN best (one-hot، Hamming، distance، K=9) | 0.9567 | 0.9706 | 0.9445 | 0.9574 | 0.9933 |
| DT سبک کلاس (entropy، عمق ۴) | 0.9255 | 0.9321 | 0.9227 | 0.9274 | 0.9756 |
| DT نهایی (entropy، pruned، one-hot) | 0.9472 | 0.9525 | 0.9445 | 0.9485 | 0.9677 |
| MLP (۴۵ epoch، مکمل) | 0.9619 | 0.9615 | 0.9647 | 0.9631 | 0.9938 |

## نحوه‌ی اجرا

```bash
# نسخه‌ی فشرده
python ml_task01.py phishing.csv

# نسخه‌ی چندمرحله‌ای (به همین ترتیب)
python step01_env_eda.py
python step02_preprocess.py
python step03_knn_baseline.py
python step04_knn_experiments.py
python step05_decision_tree.py
python step06_final_evaluation.py
python step07_importance_and_tex.py
```

نیازمندی‌ها: Python 3، numpy، pandas، scipy، scikit-learn، matplotlib، seaborn.
