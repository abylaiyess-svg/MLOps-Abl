"""Деплой: перевод проверенной версии модели в стадию Production."""

import argparse
import os
import sys

import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
REGISTERED_MODEL_NAME = "production_classifier"
SOURCE_STAGE = "Staging"
TARGET_STAGE = "Production"
SERVE_PORT = 5001


def configure_environment() -> None:
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadminpassword")
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")


def pick_version(client: MlflowClient, version: str | None):
    """Явно указанная версия, иначе самая свежая в Staging."""
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        print(f"Модель '{REGISTERED_MODEL_NAME}' не найдена в Model Registry")
        sys.exit(1)

    if version is not None:
        found = next((v for v in versions if v.version == str(version)), None)
        if found is None:
            print(f"Версия v{version} не найдена")
            sys.exit(1)
        return found

    staged = [v for v in versions if v.current_stage == SOURCE_STAGE]
    if not staged:
        print(
            f"Нет ни одной версии в стадии {SOURCE_STAGE}. "
            "Сначала запустите: python eval_gate.py"
        )
        sys.exit(1)
    return max(staged, key=lambda v: int(v.version))


def deploy(version: str | None) -> None:
    configure_environment()
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    target = pick_version(client, version)
    run = client.get_run(target.run_id)
    score = run.data.metrics.get("f1_score")
    print(
        f"Кандидат: v{target.version} (стадия {target.current_stage}, "
        f"f1_score={score:.4f})"
        if score is not None
        else f"Кандидат: v{target.version} (стадия {target.current_stage})"
    )

    previous = [
        v
        for v in client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        if v.current_stage == TARGET_STAGE
    ]
    for old in previous:
        print(f"Текущая production-версия v{old.version} будет отправлена в Archived")

    # archive_existing_versions=True — в Production остаётся ровно одна версия.
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=target.version,
        stage=TARGET_STAGE,
        archive_existing_versions=True,
    )

    print(f"Задеплоено: '{REGISTERED_MODEL_NAME}' v{target.version} -> {TARGET_STAGE}")
    print()
    print("Поднять REST-сервис с этой моделью:")
    print(
        f"  mlflow models serve -m models:/{REGISTERED_MODEL_NAME}/{TARGET_STAGE}"
        f" -p {SERVE_PORT} --env-manager local"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Перевод версии модели в стадию Production."
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Номер версии. По умолчанию — самая свежая в Staging.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        deploy(args.version)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Ошибка деплоя: {exc}")
        sys.exit(1)
