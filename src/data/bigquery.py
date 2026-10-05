import os
import re
from typing import Optional, Sequence

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery
from google.oauth2 import service_account


class BigQueryDataSource:
    """
    Reusable BigQuery data-access layer.

    Supports:
    - Service-account JSON authentication
    - Running arbitrary SQL
    - Loading BigQuery tables into pandas
    - Random sampling in BigQuery
    - Listing datasets
    - Listing tables
    - Inspecting table schemas
    - Inspecting table metadata
    - Saving local Parquet files
    - Uploading pandas DataFrames to BigQuery
    - Automatically sanitizing BigQuery column names
    """

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        project_id: Optional[str] = None,
        credentials_path: Optional[str] = None,
    ):
        load_dotenv()

        # ----------------------------------------------------
        # Credentials
        # ----------------------------------------------------

        credentials_path = (
            credentials_path
            or os.getenv(
                "GOOGLE_APPLICATION_CREDENTIALS"
            )
        )

        if not credentials_path:
            raise ValueError(
                "BigQuery credentials not found. "
                "Set GOOGLE_APPLICATION_CREDENTIALS "
                "in .env or pass credentials_path."
            )

        credentials_path = os.path.expanduser(
            credentials_path
        )

        if not os.path.exists(
            credentials_path
        ):
            raise FileNotFoundError(
                "Credential file not found: "
                f"{credentials_path}"
            )

        self.credentials = (
            service_account
            .Credentials
            .from_service_account_file(
                credentials_path
            )
        )

        # ----------------------------------------------------
        # Project
        # ----------------------------------------------------

        self.project_id = (
            project_id
            or os.getenv(
                "GCP_PROJECT_ID"
            )
            or self.credentials.project_id
        )

        if not self.project_id:
            raise ValueError(
                "GCP project ID could not be determined."
            )

        # ----------------------------------------------------
        # BigQuery client
        # ----------------------------------------------------

        self.client = bigquery.Client(
            project=self.project_id,
            credentials=self.credentials,
        )

    # ========================================================
    # QUERY
    # ========================================================

    def query(
        self,
        sql: str,
        job_config: Optional[
            bigquery.QueryJobConfig
        ] = None,
    ) -> pd.DataFrame:
        """
        Execute a BigQuery SQL query and return
        a pandas DataFrame.
        """

        query_job = (
            self.client.query(
                sql,
                job_config=job_config,
            )
        )

        return (
            query_job
            .result()
            .to_dataframe()
        )

    # ========================================================
    # LOAD TABLE
    # ========================================================

    def load_table(
        self,
        table: str,
        columns: Optional[
            Sequence[str]
        ] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load a BigQuery table into a pandas DataFrame.

        Example:

            df = bq.load_table(
                "project.dataset.table",
                columns=[
                    "user_id",
                    "age",
                    "target",
                ],
                limit=10000,
            )
        """

        if columns:

            column_sql = ", ".join(
                f"`{column}`"
                for column in columns
            )

        else:

            column_sql = "*"

        sql = f"""
        SELECT
            {column_sql}
        FROM `{table}`
        """

        if limit is not None:

            if (
                not isinstance(
                    limit,
                    int,
                )
                or limit <= 0
            ):
                raise ValueError(
                    "limit must be a "
                    "positive integer."
                )

            sql += (
                f"\nLIMIT {limit}"
            )

        return self.query(
            sql
        )

    # ========================================================
    # RANDOM SAMPLE
    # ========================================================

    def load_table_sample(
        self,
        table_name: str,
        sample_size: int,
        random_seed=None,
    ) -> pd.DataFrame:
        """
        Load a random sample from BigQuery.

        Sampling happens in BigQuery so the entire
        table is not downloaded locally.

        Note:
        ORDER BY RAND() is convenient but can be expensive
        for very large BigQuery tables.
        """

        if (
            not isinstance(
                sample_size,
                int,
            )
            or sample_size <= 0
        ):
            raise ValueError(
                "sample_size must be "
                "a positive integer."
            )

        print(
            f"\n[BIGQUERY] Randomly sampling "
            f"{sample_size:,} rows..."
        )

        query = f"""
        SELECT *
        FROM `{table_name}`
        ORDER BY RAND()
        LIMIT {int(sample_size)}
        """

        df = (
            self.client
            .query(query)
            .result()
            .to_dataframe()
        )

        print(
            f"[BIGQUERY] Loaded "
            f"{len(df):,} sampled rows."
        )

        return df

    # ========================================================
    # DATASETS
    # ========================================================

    def list_datasets(
        self,
    ) -> list[str]:
        """
        Return datasets available to
        the service account.
        """

        return [
            dataset.dataset_id
            for dataset
            in self.client.list_datasets()
        ]

    # ========================================================
    # TABLES
    # ========================================================

    def list_tables(
        self,
        dataset: str,
    ) -> list[str]:
        """
        Return tables in a dataset.

        dataset can be:

            dataset_name

        or:

            project.dataset_name
        """

        if "." not in dataset:

            dataset = (
                f"{self.project_id}."
                f"{dataset}"
            )

        return [
            table.table_id
            for table
            in self.client.list_tables(
                dataset
            )
        ]

    # ========================================================
    # SCHEMA
    # ========================================================

    def get_schema(
        self,
        table: str,
    ) -> pd.DataFrame:
        """
        Return BigQuery table schema
        as a pandas DataFrame.
        """

        table_obj = (
            self.client
            .get_table(
                table
            )
        )

        schema = []

        for field in table_obj.schema:

            schema.append(
                {
                    "column":
                        field.name,

                    "type":
                        field.field_type,

                    "mode":
                        field.mode,

                    "description":
                        field.description,
                }
            )

        return pd.DataFrame(
            schema
        )

    # ========================================================
    # TABLE INFORMATION
    # ========================================================

    def get_table_info(
        self,
        table: str,
    ) -> dict:
        """
        Return basic metadata about
        a BigQuery table.
        """

        table_obj = (
            self.client
            .get_table(
                table
            )
        )

        return {
            "project":
                table_obj.project,

            "dataset":
                table_obj.dataset_id,

            "table":
                table_obj.table_id,

            "num_rows":
                table_obj.num_rows,

            "num_columns":
                len(
                    table_obj.schema
                ),

            "size_bytes":
                table_obj.num_bytes,

            "created":
                table_obj.created,

            "modified":
                table_obj.modified,
        }

    # ========================================================
    # SAVE LOCAL DATA
    # ========================================================

    def save_local(
        self,
        df: pd.DataFrame,
        path: str = (
            "data/raw/dataset.parquet"
        ),
    ) -> None:
        """
        Save DataFrame locally as Parquet.
        """

        directory = os.path.dirname(
            path
        )

        if directory:

            os.makedirs(
                directory,
                exist_ok=True,
            )

        df.to_parquet(
            path,
            index=False,
        )

        print(
            f"Dataset saved to: {path}"
        )

    # ========================================================
    # BIGQUERY COLUMN NAME SANITIZATION
    # ========================================================

    @staticmethod
    def _sanitize_column_name(
        column,
    ) -> str:
        """
        Convert a column name into a safe
        BigQuery-compatible column name.

        Examples:

            probability_0.0
                -> probability_0_0

            customer age
                -> customer_age

            income($)
                -> income

            feature/value
                -> feature_value
        """

        name = str(
            column
        ).strip()

        # Replace invalid characters with "_"
        name = re.sub(
            r"[^A-Za-z0-9_]",
            "_",
            name,
        )

        # Collapse multiple underscores
        name = re.sub(
            r"_+",
            "_",
            name,
        )

        # Remove leading/trailing underscores
        name = name.strip(
            "_"
        )

        # Empty column names are not useful
        if not name:

            name = "column"

        # Keep names within BigQuery limit
        name = name[:300]

        return name

    # ========================================================
    # SANITIZE DATAFRAME COLUMNS
    # ========================================================

    def sanitize_dataframe_columns(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return a copy of the DataFrame with
        BigQuery-safe column names.

        Raises an error if sanitization creates
        duplicate column names.
        """

        df = df.copy()

        original_columns = (
            df.columns
            .astype(str)
            .tolist()
        )

        new_columns = [
            self._sanitize_column_name(
                column
            )
            for column
            in original_columns
        ]

        # ----------------------------------------------------
        # Check duplicate names after sanitization
        #
        # Example:
        #
        # feature-a -> feature_a
        # feature.a -> feature_a
        #
        # These would collide.
        # ----------------------------------------------------

        duplicate_columns = (
            pd.Index(
                new_columns
            )
            [
                pd.Index(
                    new_columns
                ).duplicated()
            ]
            .unique()
            .tolist()
        )

        if duplicate_columns:

            raise ValueError(
                "Column-name sanitization created "
                "duplicate BigQuery columns: "
                f"{duplicate_columns}"
            )

        # ----------------------------------------------------
        # Print renamed columns
        # ----------------------------------------------------

        renamed = [
            (
                old,
                new,
            )
            for old, new
            in zip(
                original_columns,
                new_columns,
            )
            if old != new
        ]

        if renamed:

            print(
                "\n[BIGQUERY] "
                "Sanitized column names:"
            )

            for old, new in renamed:

                print(
                    f"  {old} -> {new}"
                )

        else:

            print(
                "\n[BIGQUERY] "
                "All column names are valid."
            )

        df.columns = (
            new_columns
        )

        return df

    # ========================================================
    # UPLOAD DATAFRAME
    # ========================================================

    def upload_dataframe(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        table_name: str,
        project_id: Optional[
            str
        ] = None,
        write_disposition: str = (
            "WRITE_TRUNCATE"
        ),
    ) -> str:
        """
        Upload a pandas DataFrame to BigQuery.

        Parameters
        ----------
        df:
            DataFrame to upload.

        dataset_id:
            Destination BigQuery dataset.

        table_name:
            Destination BigQuery table.

        project_id:
            Optional project override.

        write_disposition:
            WRITE_TRUNCATE
            WRITE_APPEND
            WRITE_EMPTY

        Returns
        -------
        str
            Full BigQuery table ID.
        """

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if not isinstance(
            df,
            pd.DataFrame,
        ):
            raise TypeError(
                "df must be a pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "Cannot upload an empty DataFrame."
            )

        if not dataset_id:
            raise ValueError(
                "dataset_id is required."
            )

        if not table_name:
            raise ValueError(
                "table_name is required."
            )

        allowed_write_dispositions = {
            "WRITE_TRUNCATE",
            "WRITE_APPEND",
            "WRITE_EMPTY",
        }

        if (
            write_disposition
            not in allowed_write_dispositions
        ):
            raise ValueError(
                "Invalid write_disposition. "
                "Use WRITE_TRUNCATE, "
                "WRITE_APPEND, or WRITE_EMPTY."
            )

        # ----------------------------------------------------
        # Project
        # ----------------------------------------------------

        if project_id is None:

            project_id = (
                self.client.project
            )

        # ----------------------------------------------------
        # Sanitize table name
        # ----------------------------------------------------

        safe_table_name = (
            self._sanitize_column_name(
                table_name
            )
        )

        if (
            safe_table_name
            != table_name
        ):

            print(
                "\n[BIGQUERY] "
                "Sanitized table name:"
            )

            print(
                f"  {table_name} "
                f"-> {safe_table_name}"
            )

        table_name = (
            safe_table_name
        )

        # ----------------------------------------------------
        # Sanitize DataFrame columns
        # ----------------------------------------------------

        df_upload = (
            self
            .sanitize_dataframe_columns(
                df
            )
        )

        # ----------------------------------------------------
        # Build destination
        # ----------------------------------------------------

        table_id = (
            f"{project_id}."
            f"{dataset_id}."
            f"{table_name}"
        )

        # ----------------------------------------------------
        # Upload information
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 70
        )

        print(
            "BIGQUERY UPLOAD"
        )

        print(
            "=" * 70
        )

        print(
            f"\nDestination: "
            f"{table_id}"
        )

        print(
            f"Rows: "
            f"{len(df_upload):,}"
        )

        print(
            f"Columns: "
            f"{len(df_upload.columns):,}"
        )

        print(
            f"Write disposition: "
            f"{write_disposition}"
        )

        # ----------------------------------------------------
        # Job configuration
        # ----------------------------------------------------

        job_config = (
            bigquery.LoadJobConfig(
                write_disposition=(
                    write_disposition
                ),
            )
        )

        # ----------------------------------------------------
        # Upload
        # ----------------------------------------------------

        print(
            "\nUploading table..."
        )

        try:

            job = (
                self.client
                .load_table_from_dataframe(
                    df_upload,
                    table_id,
                    job_config=job_config,
                )
            )

            # Wait for completion
            job.result()

        except Exception as exc:

            print(
                "\n[BIGQUERY] "
                "Upload failed."
            )

            print(
                f"Destination: "
                f"{table_id}"
            )

            print(
                f"Error: {exc}"
            )

            raise

        # ----------------------------------------------------
        # Verify uploaded table
        # ----------------------------------------------------

        table = (
            self.client
            .get_table(
                table_id
            )
        )

        print(
            "\n[BIGQUERY] "
            "Upload complete."
        )

        print(
            f"Table: "
            f"{table_id}"
        )

        print(
            f"Rows: "
            f"{table.num_rows:,}"
        )

        print(
            f"Columns: "
            f"{len(table.schema):,}"
        )

        return table_id