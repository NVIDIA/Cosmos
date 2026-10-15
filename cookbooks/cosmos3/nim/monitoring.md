<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Monitor the Cosmos3 Certified NIM with Prometheus and Grafana

Scrape the NIM's Prometheus metrics and chart them in Grafana. The steps apply
to both the Generator and the Reasoner.

## Prerequisites

A running NIM that serves metrics. Confirm that the endpoint returns
Prometheus text; see [Metrics](operations.md#metrics).

```bash
export NIM_URL=${NIM_URL:-http://localhost:8000}
curl -fsS "$NIM_URL/v1/metrics" | head
```

## Prometheus

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: cosmos3-certified-nim
    metrics_path: /v1/metrics
    scrape_interval: 15s
    static_configs:
      - targets: ["cosmos3-generator:8000"]
```

| Field | Value |
|-------|-------|
| `metrics_path` | `/v1/metrics` (not the Prometheus default `/metrics`) |
| `targets` | `<host>:8000`, the NIM HTTP API port. Use the container name on a shared Docker network, or `localhost:8000` with host networking. |

Run Prometheus with the config mounted read-only. This example uses a shared
Docker network; attach the NIM container to it. The examples use the Generator
container name from [Deployment](deployment.md#launch-generator); for a Reasoner,
use `cosmos3-reasoner` instead.

```bash
docker network create cosmos3-monitoring   # skip if it exists
docker network connect cosmos3-monitoring cosmos3-generator

docker run -d --name prometheus \
  --network cosmos3-monitoring \
  -p 9090:9090 \
  -v "$PWD/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
  prom/prometheus:v2.53.0
```

To scrape a NIM published on the host instead, use `--network host`, drop
`-p 9090:9090`, and set the target to `localhost:8000`.

Open `http://localhost:9090/targets` (**Status > Targets**). The
`cosmos3-certified-nim` job must show `UP`.

| Symptom | Cause |
|---------|-------|
| Target `DOWN`, `404` | `metrics_path` is not `/v1/metrics`. |
| Target `DOWN`, connection refused or no such host | Prometheus cannot reach the NIM. Put both containers on the same Docker network and use the container name, or use host networking. |

## Grafana

```bash
docker run -d --name grafana \
  --network cosmos3-monitoring \
  -p 3000:3000 \
  grafana/grafana:11.1.0
```

Open `http://localhost:3000`, sign in as `admin` / `admin`, and change the
password when prompted. Then add the data source:

1. Go to **Connections > Data sources > Add data source**.
2. Select **Prometheus**.
3. Set **Prometheus server URL** to `http://prometheus:9090` (the container
   name on the shared network).
4. Select **Save & test**.

Optionally, provision the data source instead. Save the file as
`grafana-datasource.yml` and add
`-v "$PWD/grafana-datasource.yml:/etc/grafana/provisioning/datasources/prometheus.yml:ro"`
to the `docker run` command:

```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

Use **Explore** to browse the metrics Prometheus has collected, then build
dashboard panels from them. Metric names differ between the Generator and
Reasoner runtimes and by image, so take them from the `/v1/metrics` output of
your running service rather than from a fixed list.

## Kubernetes

For Helm deployments, set `metrics.serviceMonitor.enabled: true` to have
Prometheus Operator scrape the NIM. See [Helm](helm.md).

## Security

The NIM does not authenticate requests. Do not expose `/v1/metrics`,
Prometheus, or Grafana outside a trusted network. Change Grafana's default
`admin` credentials before use.
