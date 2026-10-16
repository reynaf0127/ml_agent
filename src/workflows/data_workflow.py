import json
import os

import pandas as pd

from src.data.bigquery import BigQueryDataSource


CACHE_PATH = "data/raw/dataset.parquet"
DATASET_METADATA_PATH = "data/raw/dataset_metadata.json"


def load_data_project(
    project_id,
    dataset_id,
):

    print("\n" + "=" * 70)
    print("DATA PROJECT")
    print("=" * 70)

    cache_exists = os.path.exists(
        CACHE_PATH
    )

    metadata_exists = os.path.exists(
        DATASET_METADATA_PATH
    )

    dataset_metadata = None

    if metadata_exists:

        with open(
            DATASET_METADATA_PATH,
            "r",
        ) as f:

            dataset_metadata = json.load(f)

    # ========================================================
    # EXISTING CACHE
    # ========================================================

    if cache_exists:

        if dataset_metadata:

            print(
                "\nCurrent cached dataset:"
            )

            print(
                dataset_metadata.get(
                    "full_table_name",
                    "Unknown",
                )
            )

        else:

            print(
                "\nCurrent cached dataset found."
            )

        new_project = input(
            "\nStart a new data project with "
            "a different BigQuery table? (y/N):\n> "
        ).strip().lower()

    else:

        print(
            "\nNo cached dataset found."
        )

        new_project = "y"

    # ========================================================
    # LOAD EXISTING
    # ========================================================

    if new_project not in [
        "y",
        "yes",
    ]:

        print(
            "\n[CACHE] Loading existing dataset..."
        )

        df = pd.read_parquet(
            CACHE_PATH
        )

        if dataset_metadata:

            print(
                "[CACHE] Table: "
                f"{dataset_metadata.get('full_table_name')}"
            )

        print(
            f"[CACHE] Shape: "
            f"{df.shape[0]:,} rows × "
            f"{df.shape[1]:,} columns"
        )

        return (
            df,
            dataset_metadata,
        )

    # ========================================================
    # NEW PROJECT
    # ========================================================

    print("\n" + "=" * 70)
    print("NEW DATA PROJECT")
    print("=" * 70)

    bq = BigQueryDataSource()

    tables = bq.list_tables(
        dataset_id
    )

    print(
        f"\nAvailable tables in "
        f"{project_id}.{dataset_id}:"
    )

    for table in tables:

        print(
            f"  - {table}"
        )

    table_name = input(
        "\nEnter table name:\n> "
    ).strip()

    if not table_name:

        raise ValueError(
            "Table name cannot be empty."
        )

    if table_name not in tables:

        raise ValueError(
            f"Unknown table: {table_name}"
        )

    full_table_name = (
        f"{project_id}."
        f"{dataset_id}."
        f"{table_name}"
    )

    print(
        f"\n[BIGQUERY] Loading:"
        f"\n{full_table_name}"
    )

    use_sample = input(
        "\nLoad a random sample instead of "
        "the full table? (y/N):\n> "
    ).strip().lower()

    # ========================================================
    # SAMPLE
    # ========================================================

    if use_sample in [
        "y",
        "yes",
    ]:

        sample_input = input(
            "\nHow many rows would you like "
            "to sample? "
            "(default 50000):\n> "
        ).strip()

        if sample_input == "":

            sample_size = 50_000

        else:

            try:

                sample_size = int(
                    sample_input
                )

            except ValueError:

                print(
                    "\n[DATA] Invalid sample size. "
                    "Using 50,000."
                )

                sample_size = 50_000

        if sample_size <= 0:

            raise ValueError(
                "Sample size must be greater than 0."
            )

        df = bq.load_table_sample(
            table_name=full_table_name,
            sample_size=sample_size,
        )

        sampling_method = (
            "random_sample"
        )

    # ========================================================
    # FULL TABLE
    # ========================================================

    else:

        print(
            "\n[BIGQUERY] Loading full table..."
        )

        df = bq.load_table(
            full_table_name
        )

        sample_size = None

        sampling_method = (
            "full_table"
        )

    # ========================================================
    # SAVE
    # ========================================================

    bq.save_local(
        df,
        CACHE_PATH,
    )

    dataset_metadata = {

        "project_id":
            project_id,

        "dataset_id":
            dataset_id,

        "table_name":
            table_name,

        "full_table_name":
            full_table_name,

        "rows":
            int(df.shape[0]),

        "columns":
            int(df.shape[1]),

        "sampling_method":
            sampling_method,

        "sample_size":
            sample_size,
    }

    os.makedirs(
        os.path.dirname(
            DATASET_METADATA_PATH
        ),
        exist_ok=True,
    )

    with open(
        DATASET_METADATA_PATH,
        "w",
    ) as f:

        json.dump(
            dataset_metadata,
            f,
            indent=4,
        )

    print(
        "\n[DATA] New dataset cached."
    )

    print(
        f"[DATA] Shape: "
        f"{df.shape[0]:,} rows × "
        f"{df.shape[1]:,} columns"
    )

    return (
        df,
        dataset_metadata,
    )