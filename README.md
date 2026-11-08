# IMU Forecasting with PyTorch and MLOps

Proyecto de forecasting temporal para las seis señales inerciales del dataset
**EuRoC MAV**: tres velocidades angulares y tres aceleraciones lineales. Se
entrenan modelos RNN y LSTM en PyTorch para predecir la muestra `t+1`, se
exportan a ONNX y se sirven mediante Triton Inference Server.

El proyecto incluye evaluación de métricas, reconstrucción visual de trayectoria
2D, pruebas unitarias y observabilidad con Prometheus y Grafana.

## Estructura del proyecto

```text
IMU_Forecasting/
├── checkpoints/                 # Pesos .pth de los mejores modelos
├── data/
│   ├── raw/                     # Secuencias EuRoC IMU y Ground Truth
│   └── processed/               # Scaler, métricas visuales y trayectorias
├── model_repository/
│   ├── lstm_forecaster/         # Modelo ONNX y configuración Triton
│   └── rnn_forecaster/          # Modelo ONNX y configuración Triton
├── src/
│   ├── data_pipeline.py         # Carga, normalización y ventanas temporales
│   ├── models.py                # Arquitecturas IMURNN e IMULSTM
│   ├── train.py                 # Entrenamiento, checkpoints y exportación ONNX
│   ├── evaluate.py              # Métricas y gráficos de predicción
│   ├── trajectory_integration.py # Integración Strapdown y trayectoria 2D
│   └── triton_client.py         # Cliente HTTP de validación de Triton
├── tests/                       # Pruebas del pipeline y modelos
├── deploy/
│   ├── Dockerfile               # Entorno PyTorch CUDA reproducible
│   ├── docker-compose.yml       # Triton, Prometheus y Grafana
│   └── observability/           # Scrape de métricas y dashboard Grafana
├── requirements.txt
└── .pre-commit-config.yaml
```

`src/` contiene el flujo de machine learning; `model_repository/` sigue la
estructura requerida por Triton (`<modelo>/1/model.onnx`); `deploy/` concentra
la ejecución reproducible y la observabilidad del servicio.

## Instalación y configuración local

Se recomienda usar un entorno Conda con Python 3.11 y PyTorch instalado según
la versión CUDA disponible:

```bash
conda create -n deeplearning python=3.11
conda activate deeplearning

pip install -r requirements.txt
pip install pre-commit
pre-commit install
```

Para una instalación CPU de PyTorch, consulte el selector oficial de PyTorch.
La imagen definida en `deploy/Dockerfile` ya parte de una imagen PyTorch con
CUDA.

## Flujo de ejecución

Todos los comandos se ejecutan desde la raíz del proyecto.

### 1. Pruebas unitarias

```bash
python -m pytest tests/
```

### 2. Entrenamiento

Ejemplo LSTM:

```bash
python src/train.py \
  --model_type lstm \
  --sequence data/raw/V1_01_easy/imu0/data.csv \
  --hidden_dim 128 \
  --num_layers 2
```

Para RNN:

```bash
python src/train.py \
  --model_type rnn \
  --sequence data/raw/V1_01_easy/imu0/data.csv
```

El entrenamiento guarda el mejor checkpoint PyTorch en
`checkpoints/best_<modelo>.pth` y exporta el modelo ONNX a
`model_repository/<modelo>_forecaster/1/model.onnx`.

### 3. Evaluación

```bash
python src/evaluate.py \
  --model_type lstm \
  --sequence data/raw/V1_01_easy/imu0/data.csv \
  --hidden_dim 128 \
  --num_layers 2
```

La evaluación calcula MSE y MAE, y guarda la comparación de señales en
`data/processed/eval_plot_lstm.png`.

### 4. Reconstrucción 2D de trayectoria

```bash
python src/trajectory_integration.py \
  --model_type lstm \
  --sequence data/raw/V1_01_easy/imu0/data.csv
```

El script integra IMU real en Train/Val y predicha en Test, usando el Ground
Truth EuRoC para las condiciones iniciales. Guarda la figura en
`data/processed/trajectory_lstm.png`.

## Despliegue MLOps con Triton

El Compose debe ejecutarse desde la raíz del proyecto. Las rutas del archivo
Compose están definidas relativas a `deploy/docker-compose.yml`.

### 1. Levantar servicios

```bash
docker compose -f deploy/docker-compose.yml up -d
```

Servicios publicados:

- Triton HTTP: `http://localhost:8000`
- Triton gRPC: `localhost:8001`
- Métricas Triton: `http://localhost:8002/metrics`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

La configuración monta `model_repository/` en Triton, provisiona Prometheus
como datasource de Grafana y carga automáticamente el dashboard de Triton.

### 2. Probar inferencia HTTP

Instala `tritonclient[http]` mediante `requirements.txt` y ejecuta:

```bash
python src/triton_client.py --model lstm_forecaster
```

Para probar la RNN:

```bash
python src/triton_client.py --model rnn_forecaster
```

El cliente obtiene un único batch normalizado mediante `get_dataloaders`,
envía una entrada `[1, 50, 6]` al endpoint HTTP de Triton y muestra la salida
`[1, 6]`.

### 3. Dashboard

Accede a [Grafana](http://localhost:3000) para consultar throughput, latencia
de inferencia y utilización de GPU. Las credenciales iniciales configuradas en
Compose son `admin` / `admin`; cámbialas para un entorno real.
