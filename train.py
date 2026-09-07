"""Обучение RandomForest и регистрация модели в MLflow."""

import argparse
import os
import sys
import time
from pathlib import Path

import boto3
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
from botocore.client import Config
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data.csv")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "sentiment_classification"
REGISTERED_MODEL_NAME = "production_classifier"
DEFAULT_N_ESTIMATORS = 100
DEFAULT_MAX_DEPTH = 10
RANDOM_STATE = 42


def configure_environment() -> None:
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadminpassword")
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"S3 endpoint: {os.environ['MLFLOW_S3_ENDPOINT_URL']}")


def ensure_minio_bucket(bucket_name: str = "mlflow") -> None:
    endpoint = os.environ["MLFLOW_S3_ENDPOINT_URL"]
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )
    existing = {bucket["Name"] for bucket in client.list_buckets().get("Buckets", [])}
    if bucket_name in existing:
        print(f"S3-бакет уже существует: {bucket_name}")
        return
    client.create_bucket(Bucket=bucket_name)
    print(f"Создан S3-бакет: {bucket_name}")


def wait_for_mlflow(retries: int = 15, delay_sec: float = 2.0) -> None:
    for attempt in range(1, retries + 1):
        try:
            mlflow.search_experiments()
            print("MLflow сервер доступен")
            return
        except Exception as exc:  # noqa: BLE001 — ждём готовности сервиса
            print(f"Ожидание MLflow ({attempt}/{retries}): {exc}")
            time.sleep(delay_sec)
    print("MLflow недоступен. Проверьте docker-compose и адрес http://localhost:5000")
    sys.exit(1)


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    if not DATA_PATH.exists():
        print(f"Файл {DATA_PATH} не найден. Сначала запустите: python prepare_data.py")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    if "target" not in df.columns:
        print("В data.csv отсутствует колонка 'target'")
        sys.exit(1)

    print(f"Загружен датасет: {DATA_PATH} ({len(df)} строк)")
    return df.drop(columns=["target"]), df["target"]


def train(n_estimators: int, max_depth: int) -> None:
    configure_environment()
    ensure_minio_bucket()
    wait_for_mlflow()

    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        print(
            f"Обучение RandomForestClassifier "
            f"(n_estimators={n_estimators}, max_depth={max_depth})..."
        )
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score = f1_score(y_test, y_pred)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("f1_score", score)
        # Сигнатура и пример входа: без них REST-эндпоинт не знает схему
        # запроса, а MLflow пишет предупреждение при логировании.
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            signature=infer_signature(X_test, y_pred),
            input_example=X_test.head(3),
        )

        print(f"Run ID: {run.info.run_id}")
        print(f"f1_score: {score:.4f}")
        print(f"Модель зарегистрирована как '{REGISTERED_MODEL_NAME}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обучение RandomForest и регистрация модели в MLflow."
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=DEFAULT_N_ESTIMATORS,
        help=f"Число деревьев (по умолчанию {DEFAULT_N_ESTIMATORS})",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help=f"Максимальная глубина дерева (по умолчанию {DEFAULT_MAX_DEPTH})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        train(args.n_estimators, args.max_depth)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Ошибка обучения: {exc}")
        sys.exit(1)
