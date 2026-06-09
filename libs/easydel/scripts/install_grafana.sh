#!/usr/bin/env bash
# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Install Prometheus + Grafana for eSurge monitoring.
# Usage: sudo bash scripts/install_grafana.sh [--start]
#
# Options:
#   --start    Start and enable both services after installation.

set -euo pipefail

PROMETHEUS_VERSION="3.3.0"

START_SERVICE=false
for arg in "$@"; do
    case "$arg" in
        --start) START_SERVICE=true ;;
        -h|--help)
            echo "Usage: sudo bash scripts/install_grafana.sh [--start]"
            echo ""
            echo "Installs Prometheus and Grafana OSS for eSurge monitoring."
            echo "Supports Debian/Ubuntu and RHEL/Fedora."
            echo ""
            echo "Options:"
            echo "  --start    Start and enable services after installation"
            echo "  -h,--help  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

if [ "$(id -u)" -ne 0 ]; then
    echo "Error: this script must be run as root (use sudo)."
    exit 1
fi

# ── Detect architecture ──
ARCH=$(uname -m)
case "$ARCH" in
    x86_64)  PROM_ARCH="amd64" ;;
    aarch64) PROM_ARCH="arm64" ;;
    armv7l)  PROM_ARCH="armv7" ;;
    *)
        echo "Error: unsupported architecture: $ARCH"
        exit 1
        ;;
esac

# ────────────────────────────────────────────────────────────────
#  Prometheus (standalone binary — works on any distro)
# ────────────────────────────────────────────────────────────────
install_prometheus() {
    if command -v prometheus &> /dev/null; then
        echo ">> Prometheus already installed: $(command -v prometheus)"
        return
    fi

    local tarball="prometheus-${PROMETHEUS_VERSION}.linux-${PROM_ARCH}.tar.gz"
    local url="https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/${tarball}"
    local tmpdir
    tmpdir=$(mktemp -d)

    echo ">> Downloading Prometheus ${PROMETHEUS_VERSION} (${PROM_ARCH})..."
    wget -q -O "${tmpdir}/${tarball}" "$url"

    echo ">> Extracting..."
    tar -xzf "${tmpdir}/${tarball}" -C "${tmpdir}"

    local extracted="${tmpdir}/prometheus-${PROMETHEUS_VERSION}.linux-${PROM_ARCH}"
    install -m 0755 "${extracted}/prometheus" /usr/local/bin/prometheus
    install -m 0755 "${extracted}/promtool"   /usr/local/bin/promtool

    rm -rf "${tmpdir}"
    echo ">> Prometheus installed to /usr/local/bin/prometheus"
}

# ────────────────────────────────────────────────────────────────
#  Grafana
# ────────────────────────────────────────────────────────────────
install_grafana_debian() {
    echo ">> Installing prerequisites..."
    apt-get update
    apt-get install -y apt-transport-https software-properties-common wget gnupg

    echo ">> Adding Grafana GPG key..."
    mkdir -p /etc/apt/keyrings/
    wget -q -O - https://apt.grafana.com/gpg.key \
        | gpg --dearmor \
        | tee /etc/apt/keyrings/grafana.gpg > /dev/null

    echo ">> Adding Grafana APT repository..."
    echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" \
        | tee /etc/apt/sources.list.d/grafana.list

    echo ">> Installing Grafana OSS..."
    apt-get update
    apt-get install -y grafana
}

install_grafana_rhel() {
    echo ">> Adding Grafana YUM repository..."
    cat > /etc/yum.repos.d/grafana.repo <<'REPO'
[grafana]
name=grafana
baseurl=https://rpm.grafana.com
repo_gpgcheck=1
enabled=1
gpgcheck=1
gpgkey=https://rpm.grafana.com/gpg.key
sslverify=1
sslcacert=/etc/pki/tls/certs/ca-bundle.crt
REPO

    echo ">> Installing Grafana OSS..."
    if command -v dnf &> /dev/null; then
        dnf install -y grafana
    else
        yum install -y grafana
    fi
}

# ── Install both ──
install_prometheus

if [ -f /etc/debian_version ]; then
    install_grafana_debian
elif [ -f /etc/redhat-release ]; then
    install_grafana_rhel
else
    echo "Error: unsupported distribution for Grafana. This script supports Debian/Ubuntu and RHEL/Fedora."
    echo "See https://grafana.com/docs/grafana/latest/setup-grafana/installation/ for manual instructions."
    exit 1
fi

echo ""
echo "Installation complete."
echo "  Prometheus: $(command -v prometheus || echo 'not found on PATH')"
echo "  Grafana:    $(command -v grafana-server || echo 'not found on PATH')"
echo ""
echo "  Grafana UI: http://localhost:3000  (admin / admin)"
echo ""
echo "  Note: eSurge auto-starts Prometheus (port 9090) and provisions Grafana"
echo "  when you call engine.start_monitoring(start_grafana=True)."

if [ "$START_SERVICE" = true ]; then
    echo ""
    echo ">> Starting grafana-server..."
    systemctl daemon-reload
    systemctl start grafana-server
    systemctl enable grafana-server
    echo "grafana-server is running and enabled on boot."
    echo ""
    echo "  Prometheus does NOT need to be started manually — eSurge starts"
    echo "  it automatically with the correct scrape config."
fi
