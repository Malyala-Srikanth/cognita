import os

import os
import tempfile
from typing import Dict, Iterator, List

from boto3 import client

from backend.logger import logger
from backend.modules.dataloaders.loader import BaseDataLoader
from backend.types import DataIngestionMode, DataPoint, DataSource, LoadedDataPoint

class S3Loader(BaseDataLoader):
    """
    Load data from an S3 bucket
    """

    def load_filtered_data(
        self,
        data_source: DataSource,
        dest_dir: str,
        previous_snapshot: Dict[str, str],
        batch_size: int,
        data_ingestion_mode: DataIngestionMode,
    ) -> Iterator[List[LoadedDataPoint]]:
        """
        Loads data from an S3 bucket
        """

        s3_client = client("s3")
        bucket, *key_parts = data_source.uri.split("/")[2:]
        prefix = "/".join(key_parts)
        
        # List objects in the S3 bucket with the given prefix
        objects = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

        loaded_data_points: List[LoadedDataPoint] = []
        for obj in objects.get('Contents', []):
            key = obj['Key']
            with tempfile.NamedTemporaryFile(dir=dest_dir, delete=False) as temp_file:
                s3_client.download_fileobj(bucket, key, temp_file)
                temp_file_path = temp_file.name

            logger.debug(f"S3Loader: Downloaded {key} and saved to {temp_file_path}")

            with open(temp_file_path, "r") as file:
                content = file.read()
            
            rel_path = os.path.relpath(key, prefix)
            loaded_data_points.append(
                LoadedDataPoint(
                    data_point=DataPoint(uri=rel_path, content=content),
                    snapshot=previous_snapshot.get(rel_path),
                )
            )
            
            os.unlink(temp_file_path)

            if len(loaded_data_points) == batch_size:
                yield loaded_data_points
                loaded_data_points = []

        if loaded_data_points:
            yield loaded_data_points