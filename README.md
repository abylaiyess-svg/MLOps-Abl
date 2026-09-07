# MLOps-практикум (1 час)

Стек: DVC, MLflow, Docker Compose, GitHub Actions.

## Запуск инфраструктуры

```bash
docker-compose up -d
```

- MLflow UI: http://localhost:5000
- MinIO UI: http://localhost:9001 (логин `minioadmin`, пароль `minioadminpassword`)

## Обучение и Quality Gate

```bash
pip install -r requirements.txt
python prepare_data.py
python train.py
python eval_gate.py
```

`eval_gate.py` пропускает модель `production_classifier` в стадию `Staging`, если `f1_score >= 0.80`.

## DVC

```bash
dvc init
dvc remote add -d minio s3://mlflow/dvc
dvc remote modify minio endpointurl http://localhost:9000
dvc remote modify minio access_key_id minioadmin
dvc remote modify minio secret_access_key minioadminpassword
```

После этого данные можно версионировать так:

```bash
dvc add data.csv
dvc push
```

## CI

При push/PR в `main` GitHub Actions поднимает стек, обучает модель и запускает Quality Gate.

## Вторая версия модели и деплой

Гиперпараметры `train.py` задаются флагами, поэтому новая версия обучается той же
командой с другими параметрами:

```bash
python train.py                                  # v1: n_estimators=100, max_depth=10  -> f1 0.8426
python eval_gate.py
python train.py --n-estimators 300 --max-depth 20  # v2: улучшенная              -> f1 0.8638
python eval_gate.py
```

`eval_gate.py` переводит прошедшую версию в `Staging`. Деплой — отдельный шаг:

```bash
python deploy.py              # самая свежая версия из Staging -> Production
python deploy.py --version 2  # либо конкретная версия
```

`deploy.py` использует `archive_existing_versions=True`, поэтому в `Production`
всегда остаётся ровно одна версия, а предыдущая уходит в `Archived`.

### REST-сервис с production-моделью

```bash
mlflow models serve -m models:/production_classifier/Production -p 5001 --env-manager local
```

Проверка:

```bash
curl -X POST http://localhost:5001/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_split": {"columns": ["feature_0", "..."], "data": [[...]]}}'
# {"predictions": [1, 1, 1, 1, 0]}
```

Служебные эндпоинты: `GET /ping`, `GET /health`, `GET /version`.
