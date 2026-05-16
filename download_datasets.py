"""
download_datasets.py — fetch diverse public datasets for pipeline testing.

Datasets downloaded
-------------------
Domain                     Source  Name                                Folder
------------------------   ------  ---------------------------------   ---------------------------
Customer / campaign        UCI     Bank Marketing (41k customers)      data/raw/bank_marketing/
E-commerce transactions    UCI     Online Retail II (1M transactions)  data/raw/online_retail/
Sensor / IoT (air)         UCI     Air Quality (9k readings)           data/raw/air_quality/
Sensor / IoT (room)        UCI     Occupancy Detection (20k records)   data/raw/occupancy/
Product spend              UCI     Wholesale Customers (440 buyers)    data/raw/wholesale_customers/
Customer seg (small)       Kaggle  Mall Customers (200 customers)      data/raw/mall_customers/
Employee / HR              Kaggle  IBM HR Attrition (1.5k employees)   data/raw/ibm_hr/

── NEW: extended coverage ──
Humans / demographics      UCI     Adult Census Income (48k)           data/raw/adult_census/
Humans / financial         UCI     Default of Credit Card (30k)        data/raw/credit_default/
Products / articles        UCI     Online News Popularity (40k)        data/raw/news_popularity/
Products / wine            UCI     Wine Quality red+white (6.5k)       data/raw/wine_quality/
Signals / physics          UCI     MAGIC Gamma Telescope (19k×10)      data/raw/magic_gamma/
Signals / clinical         UCI     Breast Cancer WDBC (569×30)         data/raw/breast_cancer_wdbc/
Signals / handwriting      UCI     Pen-Based Handwritten Digits (11k)  data/raw/pendigits/
Images as tabular          Kaggle  Fashion-MNIST (70k images × 784px)  data/raw/fashion_mnist/

Usage
-----
    python download_datasets.py              # download everything
    python download_datasets.py --list       # list datasets without downloading
    python download_datasets.py --only uci   # only UCI datasets (no Kaggle key needed)
    python download_datasets.py --only kaggle
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import textwrap
import zipfile
from pathlib import Path

import requests

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour helpers ────────────────────────────────────────────────────────────

def _green(s): return f"\033[32m{s}\033[0m"
def _yellow(s): return f"\033[33m{s}\033[0m"
def _red(s):   return f"\033[31m{s}\033[0m"
def _bold(s):  return f"\033[1m{s}\033[0m"


# ── README writer ─────────────────────────────────────────────────────────────

def _write_readme(folder: Path, title: str, source: str, description: str,
                  entity_col: str, use_case: str) -> None:
    readme = folder / "README.md"
    readme.write_text(textwrap.dedent(f"""\
        # {title}

        **Source**: {source}
        **Entity column**: `{entity_col}`
        **Use case for this pipeline**: {use_case}

        ## Description
        {description}
    """))


# ─────────────────────────────────────────────────────────────────────────────
#  UCI downloads (via ucimlrepo — no API key needed)
# ─────────────────────────────────────────────────────────────────────────────

def _uci(dataset_id: int, folder_name: str, title: str, entity_col: str,
         use_case: str, description: str) -> bool:
    folder = RAW_DIR / folder_name
    if any(folder.glob("*.csv")):
        print(f"  {_yellow('skip')}  {title} (already downloaded)")
        return True
    try:
        from ucimlrepo import fetch_ucirepo
        print(f"  {_bold('fetch')} {title} (UCI #{dataset_id}) …", end=" ", flush=True)
        ds = fetch_ucirepo(id=dataset_id)
        df = ds.data.original if hasattr(ds.data, "original") else ds.data.features
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / f"{folder_name}.csv"
        df.to_csv(out, index=False)
        _write_readme(folder, title, f"UCI ML Repository (id={dataset_id})",
                      description, entity_col, use_case)
        print(_green(f"✓  {len(df):,} rows → {out.name}"))
        return True
    except Exception as exc:
        print(_red(f"✗  {exc}"))
        return False


def download_uci_datasets() -> list[str]:
    results = []

    ok = _uci(
        dataset_id=222,
        folder_name="bank_marketing",
        title="Bank Marketing (Customer + Campaign)",
        entity_col="client ID (row index)",
        use_case="Cluster customers by demographics and campaign-response behaviour",
        description=(
            "41,188 records from a Portuguese bank's telemarketing campaigns. "
            "Features: age, job, marital status, education, balance, call duration, "
            "number of contacts, previous campaign outcome, subscription outcome (y)."
        ),
    )
    results.append("bank_marketing" if ok else "bank_marketing:FAILED")

    ok = _uci(
        dataset_id=352,
        folder_name="online_retail",
        title="Online Retail II (E-commerce transactions)",
        entity_col="CustomerID",
        use_case="Aggregate per-customer RFM features (recency, frequency, monetary) "
                 "to cluster buyer personas",
        description=(
            "1,067,371 transactions from a UK online retailer (2009-2011). "
            "Each row is one invoice line: CustomerID, StockCode, Description, "
            "Quantity, UnitPrice, Country."
        ),
    )
    results.append("online_retail" if ok else "online_retail:FAILED")

    ok = _uci(
        dataset_id=360,
        folder_name="air_quality",
        title="Air Quality UCI (Sensor data)",
        entity_col="sensor location (implicit, single site)",
        use_case="Cluster hourly sensor reading patterns across the day/week",
        description=(
            "9,358 hourly readings from 5 chemical sensors in an Italian city (2004-2005). "
            "Features: CO, NMHC, NOx, NO2, O3 sensor responses, temperature, "
            "relative humidity, absolute humidity."
        ),
    )
    results.append("air_quality" if ok else "air_quality:FAILED")

    ok = _uci(
        dataset_id=357,
        folder_name="occupancy",
        title="Occupancy Detection (IoT / Sensor)",
        entity_col="Timestamp (time-windowed rows)",
        use_case="Cluster building occupancy patterns by environmental sensor profile",
        description=(
            "20,560 readings from a room equipped with environmental sensors. "
            "Features: Temperature, Humidity, Light, CO2, HumidityRatio, Occupancy "
            "(ground truth label)."
        ),
    )
    results.append("occupancy" if ok else "occupancy:FAILED")

    ok = _uci(
        dataset_id=292,
        folder_name="wholesale_customers",
        title="Wholesale Customers (Product categories)",
        entity_col="row index (one row per wholesale customer)",
        use_case="Cluster wholesale buyers by annual spend across 6 product categories",
        description=(
            "440 wholesale customers of a Portuguese distributor. "
            "Features: Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen "
            "annual spend (monetary units). Classic product-preference segmentation benchmark."
        ),
    )
    results.append("wholesale_customers" if ok else "wholesale_customers:FAILED")

    # ── Extended coverage: humans, products, signals ──────────────────────────

    ok = _uci(
        dataset_id=2,
        folder_name="adult_census",
        title="Adult Census Income (Humans / Demographics)",
        entity_col="row index (one row per person)",
        use_case="Cluster adults by demographic + work profile (age, education, "
                 "occupation, race, sex, income bracket)",
        description=(
            "48,842 records from the 1994 US Census. "
            "Features: age, workclass, education, marital-status, occupation, "
            "relationship, race, sex, capital-gain, capital-loss, hours-per-week, "
            "native-country, income (>50K or <=50K)."
        ),
    )
    results.append("adult_census" if ok else "adult_census:FAILED")

    ok = _uci(
        dataset_id=350,
        folder_name="credit_default",
        title="Default of Credit Card Clients (Humans / Financial)",
        entity_col="ID",
        use_case="Cluster credit-card customers by payment history + credit usage; "
                 "identify default-risk personas",
        description=(
            "30,000 credit card holders in Taiwan. 24 features: credit limit, "
            "sex, education, marital status, age, 6 months of payment status, "
            "6 months of bill amounts, 6 months of payments, default label."
        ),
    )
    results.append("credit_default" if ok else "credit_default:FAILED")

    ok = _uci(
        dataset_id=332,
        folder_name="news_popularity",
        title="Online News Popularity (Products / Articles)",
        entity_col="url (article identifier)",
        use_case="Cluster articles by content style and channel; predict popularity tier",
        description=(
            "39,797 Mashable.com articles. 60 numeric features describing the "
            "article: word counts, links, images/videos, channel, day of week, "
            "keyword polarity scores, plus the share count (popularity target)."
        ),
    )
    results.append("news_popularity" if ok else "news_popularity:FAILED")

    ok = _uci(
        dataset_id=186,
        folder_name="wine_quality",
        title="Wine Quality red + white (Products / Sensory)",
        entity_col="row index (one row per wine sample)",
        use_case="Cluster wines by chemistry profile; correlate with quality score",
        description=(
            "6,497 Portuguese 'Vinho Verde' wine samples. 11 physicochemical "
            "features (fixed acidity, volatile acidity, citric acid, residual "
            "sugar, chlorides, free SO2, total SO2, density, pH, sulphates, "
            "alcohol) + quality (0-10) + type (red/white)."
        ),
    )
    results.append("wine_quality" if ok else "wine_quality:FAILED")

    # Signals — physics (replacement for HAR which UCI hasn't enabled for Python import)
    ok = _uci(
        dataset_id=159,
        folder_name="magic_gamma",
        title="MAGIC Gamma Telescope (Signals / Physics)",
        entity_col="row index (one row per Cherenkov burst measurement)",
        use_case="Cluster gamma-ray detector signal windows; 2 ground-truth "
                 "classes (gamma signal vs hadron background) hidden in 10 "
                 "physical signal descriptors",
        description=(
            "19,020 high-energy gamma-ray detector measurements from a "
            "Cherenkov telescope. 10 numeric features: shower length, width, "
            "size, concentrations, asymmetries, plus class label. Classic "
            "signal-vs-noise benchmark on real physics data."
        ),
    )
    results.append("magic_gamma" if ok else "magic_gamma:FAILED")

    # Signals / clinical — Wisconsin breast cancer cell-shape descriptors
    ok = _uci(
        dataset_id=17,
        folder_name="breast_cancer_wdbc",
        title="Breast Cancer Wisconsin Diagnostic (Clinical signal features)",
        entity_col="ID (one row per tumour sample)",
        use_case="Cluster tumour samples by cell-nucleus shape descriptors; "
                 "2 ground-truth classes (malignant / benign)",
        description=(
            "569 tumour samples. 30 numeric features computed from digitised "
            "cell-nucleus images: mean, std, worst of radius / texture / "
            "perimeter / area / smoothness / compactness / concavity / "
            "concave points / symmetry / fractal dimension. Plus diagnosis label."
        ),
    )
    results.append("breast_cancer_wdbc" if ok else "breast_cancer_wdbc:FAILED")

    # Signals / handwriting — pen-based handwritten digit features
    ok = _uci(
        dataset_id=81,
        folder_name="pendigits",
        title="Pen-Based Recognition of Handwritten Digits (Signals / Handwriting)",
        entity_col="row index (one row per drawn digit)",
        use_case="Cluster handwritten-digit pen-stroke signals; expect ~10 "
                 "natural clusters (one per digit 0-9)",
        description=(
            "10,992 handwritten digits, each represented by 16 features: 8 (x, y) "
            "coordinate pairs sampled along the pen trajectory. Resampled to a "
            "common length and normalised. Plus the digit label (0-9)."
        ),
    )
    results.append("pendigits" if ok else "pendigits:FAILED")

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Kaggle downloads
# ─────────────────────────────────────────────────────────────────────────────

def _kaggle_available() -> bool:
    try:
        from kaggle import KaggleApi  # noqa: F401
        return True
    except Exception:
        return False


def _kaggle_download(slug: str, folder_name: str, title: str,
                     expected_file: str, entity_col: str,
                     use_case: str, description: str) -> bool:
    folder = RAW_DIR / folder_name
    if (folder / expected_file).exists():
        print(f"  {_yellow('skip')}  {title} (already downloaded)")
        return True
    folder.mkdir(parents=True, exist_ok=True)
    print(f"  {_bold('fetch')} {title} (Kaggle: {slug}) …", end=" ", flush=True)
    try:
        from kaggle import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(slug, path=str(folder), unzip=True, quiet=True)
        files = list(folder.glob("**/*.csv"))
        if not files:
            files = list(folder.glob("**/*.json")) + list(folder.glob("**/*.parquet"))
        count = sum(1 for _ in files)
        _write_readme(folder, title, f"Kaggle ({slug})", description, entity_col, use_case)
        print(_green(f"✓  {count} file(s) downloaded"))
        return True
    except Exception as exc:
        print(_red(f"✗  {exc}"))
        shutil.rmtree(folder, ignore_errors=True)
        return False


def download_kaggle_datasets() -> list[str]:
    if not _kaggle_available():
        print(_red("  Kaggle package not importable — skipping Kaggle datasets."))
        return []

    results = []

    ok = _kaggle_download(
        slug="vjchoudhary7/customer-segmentation-tutorial-in-python",
        folder_name="mall_customers",
        title="Mall Customers (Customer segmentation)",
        expected_file="Mall_Customers.csv",
        entity_col="CustomerID",
        use_case="Benchmark: small (200 customers), clean, perfect for quick pipeline tests",
        description=(
            "200 mall customers with Age, Annual Income (k$), Spending Score (1-100). "
            "Classic segmentation benchmark — great for sanity-checking cluster quality."
        ),
    )
    results.append("mall_customers" if ok else "mall_customers:FAILED")

    ok = _kaggle_download(
        slug="pavansubhasht/ibm-hr-analytics-attrition-dataset",
        folder_name="ibm_hr",
        title="IBM HR Analytics (Employee attrition)",
        expected_file="WA_Fn-UseC_-HR-Employee-Attrition.csv",
        entity_col="EmployeeNumber",
        use_case="Cluster employee profiles by HR features; detect attrition-risk personas",
        description=(
            "1,470 employee records with 35 features: age, department, job role, "
            "satisfaction scores, years at company, attrition label, etc."
        ),
    )
    results.append("ibm_hr" if ok else "ibm_hr:FAILED")

    ok = _kaggle_download(
        slug="zalando-research/fashionmnist",
        folder_name="fashion_mnist",
        title="Fashion-MNIST (Images as tabular)",
        expected_file="fashion-mnist_train.csv",
        entity_col="row index (one row per 28×28 image)",
        use_case="Cluster image rows directly by their 784 raw pixel intensities; "
                 "expect 10 garment categories (t-shirt, trouser, pullover, dress, "
                 "coat, sandal, shirt, sneaker, bag, ankle boot)",
        description=(
            "70,000 grayscale 28×28 images of fashion items. Each row is one "
            "image flattened: label + 784 pixel intensity columns (0-255). "
            "Train: 60k rows · test: 10k rows. Tests how the pipeline handles "
            "very high-dimensional all-numeric data with strong cluster structure."
        ),
    )
    results.append("fashion_mnist" if ok else "fashion_mnist:FAILED")

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Summary
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(all_results: list[str]) -> None:
    print()
    print(_bold("=" * 60))
    print(_bold("  Dataset Download Summary"))
    print(_bold("=" * 60))

    rows = []
    for r in all_results:
        if r.endswith(":FAILED"):
            name = r[:-7]
            rows.append((_red("FAILED"), name))
        else:
            folder = RAW_DIR / r
            csvs = list(folder.glob("*.csv"))
            size = sum(f.stat().st_size for f in csvs) / 1024 / 1024 if csvs else 0
            rows.append((_green("  OK  "), f"{r:<24} {size:.1f} MB"))

    for status, detail in rows:
        print(f"  {status}  {detail}")

    print()
    print("All datasets saved under:", _bold(str(RAW_DIR.resolve())))
    print()
    print(_bold("Next steps:"))
    print("  • Each folder has a README.md describing the entity column and use case.")
    print("  • To run the pipeline on a different dataset, point config.yaml")
    print("    data_path to the new CSV and adjust entity_id_col.")
    print("  • Online Retail and Bank Marketing need feature-engineering before")
    print("    clustering — the feature_engineer agent handles this automatically.")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download public datasets for pipeline testing")
    parser.add_argument("--list", action="store_true", help="List datasets without downloading")
    parser.add_argument("--only", choices=["uci", "kaggle"], help="Download only one source")
    args = parser.parse_args()

    if args.list:
        print(_bold("\nDatasets available for download:\n"))
        datasets = [
            ("UCI",    "bank_marketing",      "Bank Marketing — 41k customers, campaign responses"),
            ("UCI",    "online_retail",       "Online Retail II — 1M e-commerce transactions"),
            ("UCI",    "air_quality",         "Air Quality — 9k IoT/chemical sensor readings"),
            ("UCI",    "occupancy",           "Occupancy Detection — 20k environmental sensor rows"),
            ("UCI",    "wholesale_customers", "Wholesale Customers — 440 buyers × 6 product categories"),
            ("UCI",    "adult_census",        "Adult Census — 48k humans × demographic features"),
            ("UCI",    "credit_default",      "Credit Card Default — 30k humans × financial history"),
            ("UCI",    "news_popularity",     "Online News Popularity — 40k articles × 60 content features"),
            ("UCI",    "wine_quality",        "Wine Quality — 6.5k wines × 11 chemistry features"),
            ("UCI",    "magic_gamma",         "MAGIC Gamma Telescope — 19k physics signal windows × 10 features"),
            ("UCI",    "breast_cancer_wdbc",  "Breast Cancer WDBC — 569 tumour samples × 30 cell-shape features"),
            ("UCI",    "pendigits",           "Pen Digits — 11k handwritten digit pen-strokes × 16 features"),
            ("Kaggle", "mall_customers",      "Mall Customers — 200 rows, quick sanity check"),
            ("Kaggle", "ibm_hr",              "IBM HR Analytics — 1.5k employee profiles"),
            ("Kaggle", "fashion_mnist",       "Fashion-MNIST — 70k images × 784 pixel features"),
        ]
        for src, folder, desc in datasets:
            status = _green("downloaded") if any((RAW_DIR / folder).glob("*.csv")) else "not downloaded"
            print(f"  [{src:6s}] {folder:<22} {desc}  ({status})")
        print()
        return

    all_results: list[str] = []

    if args.only != "kaggle":
        print(_bold("\n── UCI ML Repository datasets ─────────────────────────────"))
        all_results += download_uci_datasets()

    if args.only != "uci":
        print(_bold("\n── Kaggle datasets ────────────────────────────────────────"))
        all_results += download_kaggle_datasets()

    _print_summary(all_results)


if __name__ == "__main__":
    main()
