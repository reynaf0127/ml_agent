import os
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
    - Listing datasets
    - Listing tables
    - Inspecting table schemas
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        credentials_path: Optional[str] = None,
    ):
        load_dotenv()

        # Use explicitly supplied path first,
        # otherwise read from .env
        credentials_path = credentials_path or os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )

        if not credentials_path:
            raise ValueError(
                "BigQuery credentials not found. "
                "Set GOOGLE_APPLICATION_CREDENTIALS in .env "
                "or pass credentials_path."
            )

        if not os.path.exists(credentials_path):
            raise FileNotFoundError(
                f"Credential file not found: {credentials_path}"
            )

        self.credentials = (
            service_account.Credentials.from_service_account_file(
                credentials_path
            )
        )

        # If project_id isn't supplied,
        # use the project from the service-account JSON.
        self.project_id = (
            project_id
            or os.getenv("GCP_PROJECT_ID")
            or self.credentials.project_id
        )

        self.client = bigquery.Client(
            project=self.project_id,
            credentials=self.credentials,
        )

    # ---------------------------------------------------------
    # Query
    # ---------------------------------------------------------

    def query(
        self,
        sql: str,
        job_config: Optional[bigquery.QueryJobConfig] = None,
    ) -> pd.DataFrame:
        """
        Execute a BigQuery SQL query and return a pandas DataFrame.
        """

        query_job = self.client.query(
            sql,
            job_config=job_config,
        )

        return query_job.result().to_dataframe()

    # ---------------------------------------------------------
    # Load table
    # ---------------------------------------------------------

    def load_table(
        self,
        table: str,
        columns: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load a BigQuery table into a pandas DataFrame.

        Example:
            df = bq.load_table(
                "project.dataset.table",
                columns=["user_id", "age", "target"],
                limit=10000,
            )
        """

        if columns:
            column_sql = ", ".join(
                f"`{column}`" for column in columns
            )
        else:
            column_sql = "*"

        sql = f"""
        SELECT
            {column_sql}
        FROM `{table}`
        """

        if limit is not None:
            if not isinstance(limit, int) or limit <= 0:
                raise ValueError(
                    "limit must be a positive integer."
                )

            sql += f"\nLIMIT {limit}"

        return self.query(sql)

    # ---------------------------------------------------------
    # Datasets
    # ---------------------------------------------------------

    def list_datasets(self) -> list[str]:
        """
        Return datasets available to the service account.
        """

        return [
            dataset.dataset_id
            for dataset in self.client.list_datasets()
        ]

    # ---------------------------------------------------------
    # Tables
    # ---------------------------------------------------------

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
            dataset = f"{self.project_id}.{dataset}"

        return [
            table.table_id
            for table in self.client.list_tables(dataset)
        ]

    # ---------------------------------------------------------
    # Schema
    # ---------------------------------------------------------

    def get_schema(
        self,
        table: str,
    ) -> pd.DataFrame:
        """
        Return BigQuery table schema as a pandas DataFrame.
        """

        table_obj = self.client.get_table(table)

        schema = []

        for field in table_obj.schema:
            schema.append(
                {
                    "column": field.name,
                    "type": field.field_type,
                    "mode": field.mode,
                    "description": field.description,
                }
            )

        return pd.DataFrame(schema)

    # ---------------------------------------------------------
    # Table information
    # ---------------------------------------------------------

    def get_table_info(
        self,
        table: str,
    ) -> dict:
        """
        Return basic metadata about a BigQuery table.
        """

        table_obj = self.client.get_table(table)

        return {
            "project": table_obj.project,
            "dataset": table_obj.dataset_id,
            "table": table_obj.table_id,
            "num_rows": table_obj.num_rows,
            "num_columns": len(table_obj.schema),
            "size_bytes": table_obj.num_bytes,
            "created": table_obj.created,
            "modified": table_obj.modified,
        }

    def save_local(
        self,
        df: pd.DataFrame,
        path: str = "data/raw/dataset.parquet",
    ) -> None:
        """
        Save DataFrame locally as Parquet.
        """

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True,
        )

        df.to_parquet(
            path,
            index=False,
        )

        print(f"Dataset saved to: {path}")

    def load_table_sample(
        self,
        table_name,
        sample_size,
        random_seed=None,
    ):
        """
        Load a random sample from a BigQuery table.

        Sampling happens in BigQuery so the entire table
        is not downloaded locally.
        """

        print(
            f"\n[BIGQUERY] Randomly sampling "
            f"{sample_size:,} rows..."
        )

        # RAND() gives each row a random value.
        #
        # LIMIT controls how many rows are returned.
        #
        # Note:
        # ORDER BY RAND() can be expensive for extremely
        # large tables, but is simple and appropriate
        # for the current framework.

        query = f"""
            SELECT *
            FROM `{table_name}`
            ORDER BY RAND()
            LIMIT {int(sample_size)}
        """

        df = (
            self.client
            .query(query)
            .to_dataframe()
        )

        print(
            f"[BIGQUERY] Loaded "
            f"{len(df):,} sampled rows."
        )

        return df