# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""eSurge Web Dashboard with improved stability and error handling."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict
from typing import Any

from eformer.loggings import get_logger

try:
    import uvicorn
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from .metrics import get_metrics_collector

logger = get_logger("eSurgeDashboard")

DASHBOARD_HTML_FIXED = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>eSurge Monitoring Dashboard</title>
    <script>
        // Simple Canvas Chart Implementation (Chart.js alternative)
        class SimpleChart {
            constructor(ctx, config) {
                this.ctx = ctx;
                this.canvas = ctx.canvas;
                this.config = config;
                this.data = config.data;
                this.options = config.options || {};
                this.type = config.type;
                this.draw();
            }

            // Update chart data without recreating
            update(newData) {
                if (newData && newData.labels) {
                    this.data.labels = newData.labels;
                }
                if (newData && newData.datasets && newData.datasets[0]) {
                    this.data.datasets[0].data = newData.datasets[0].data;
                }
                this.draw();
            }

            destroy() {
                // Simple cleanup
                if (this.canvas) {
                    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                }
            }

            draw() {
                const canvas = this.canvas;
                const ctx = this.ctx;

                // Validate canvas element
                if (!canvas || !ctx) {
                    console.error('Canvas or context not available');
                    return;
                }

                // Check if canvas is properly rendered - but don't retry infinitely
                if (canvas.offsetWidth === 0 || canvas.offsetHeight === 0) {
                    // Try to get dimensions from parent or use defaults
                    const parent = canvas.parentElement;
                    const parentWidth = parent ? parent.offsetWidth : 400;
                    const parentHeight = parent ? parent.offsetHeight : 250;

                    if (parentWidth === 0 || parentHeight === 0) {
                        // Use fixed dimensions as last resort
                        canvas.style.width = '400px';
                        canvas.style.height = '250px';
                    }
                }

                // Set canvas size with fallback dimensions
                const containerWidth = canvas.offsetWidth || canvas.parentElement?.offsetWidth || 400;
                const containerHeight = canvas.offsetHeight || canvas.parentElement?.offsetHeight || 250;

                canvas.width = containerWidth;
                canvas.height = containerHeight;

                const width = canvas.width;
                const height = canvas.height;
                const padding = 40;

                // Clear canvas
                ctx.clearRect(0, 0, width, height);

                // Get data
                const labels = this.data.labels || [];
                const dataset = this.data.datasets[0] || {};
                const data = dataset.data || [];

                if (data.length === 0) {
                    // Draw "No Data" message
                    ctx.fillStyle = '#f0f0f0';
                    ctx.fillRect(0, 0, width, height);
                    ctx.fillStyle = '#666';
                    ctx.font = 'bold 18px Arial';
                    ctx.textAlign = 'center';
                    ctx.fillText('No Data Available', width / 2, height / 2);
                    ctx.font = '14px Arial';
                    ctx.fillText('Waiting for metrics...', width / 2, height / 2 + 25);
                    return;
                }

                // Calculate chart area
                const chartWidth = width - (padding * 2);
                const chartHeight = height - (padding * 2);

                // Find min/max values
                const maxValue = Math.max(...data);
                const minValue = Math.min(...data);
                const valueRange = maxValue - minValue || 1;

                // Draw background
                ctx.fillStyle = '#f9f9f9';
                ctx.fillRect(padding, padding, chartWidth, chartHeight);

                // Draw grid lines
                ctx.strokeStyle = '#e0e0e0';
                ctx.lineWidth = 1;
                for (let i = 0; i <= 5; i++) {
                    const y = padding + (chartHeight / 5) * i;
                    ctx.beginPath();
                    ctx.moveTo(padding, y);
                    ctx.lineTo(padding + chartWidth, y);
                    ctx.stroke();
                }

                // Draw line chart
                if (data.length > 1) {
                    ctx.strokeStyle = dataset.borderColor || '#667eea';
                    ctx.lineWidth = 2;
                    ctx.beginPath();

                    for (let i = 0; i < data.length; i++) {
                        const x = padding + (chartWidth / (data.length - 1)) * i;
                        const normalizedValue = (data[i] - minValue) / valueRange;
                        const y = padding + chartHeight - (normalizedValue * chartHeight);

                        if (i === 0) {
                            ctx.moveTo(x, y);
                        } else {
                            ctx.lineTo(x, y);
                        }
                    }
                    ctx.stroke();

                    // Draw fill area if specified
                    if (dataset.fill) {
                        ctx.fillStyle = dataset.backgroundColor || 'rgba(102, 126, 234, 0.1)';
                        ctx.lineTo(padding + chartWidth, padding + chartHeight);
                        ctx.lineTo(padding, padding + chartHeight);
                        ctx.closePath();
                        ctx.fill();
                    }

                    // Draw points (show fewer points to reduce clutter)
                    ctx.fillStyle = dataset.borderColor || '#667eea';
                    const pointStep = Math.max(1, Math.floor(data.length / 20)); // Show max 20 points
                    for (let i = 0; i < data.length; i += pointStep) {
                        const x = padding + (chartWidth / (data.length - 1)) * i;
                        const normalizedValue = (data[i] - minValue) / valueRange;
                        const y = padding + chartHeight - (normalizedValue * chartHeight);

                        ctx.beginPath();
                        ctx.arc(x, y, 2, 0, 2 * Math.PI); // Smaller radius
                        ctx.fill();
                    }
                }

                // Draw axes
                ctx.strokeStyle = '#333';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(padding, padding);
                ctx.lineTo(padding, padding + chartHeight);
                ctx.lineTo(padding + chartWidth, padding + chartHeight);
                ctx.stroke();

                // Draw labels
                ctx.fillStyle = '#666';
                ctx.font = '12px Arial';
                ctx.textAlign = 'center';

                // Y-axis labels
                for (let i = 0; i <= 5; i++) {
                    const value = minValue + (valueRange / 5) * (5 - i);
                    const y = padding + (chartHeight / 5) * i;
                    ctx.textAlign = 'right';
                    ctx.fillText(value.toFixed(1), padding - 10, y + 4);
                }

                // X-axis labels (show only a few labels to avoid crowding)
                const maxLabels = 6;
                const labelStep = Math.max(1, Math.floor(labels.length / maxLabels));
                for (let i = 0; i < labels.length; i += labelStep) {
                    const x = padding + (chartWidth / (data.length - 1)) * i;
                    ctx.textAlign = 'center';
                    // Only show time part (remove seconds for cleaner look)
                    const timeLabel = labels[i].split(':').slice(0, 2).join(':');
                    ctx.fillText(timeLabel, x, height - 10);
                }

                // Draw title
                if (dataset.label) {
                    ctx.fillStyle = '#333';
                    ctx.font = 'bold 14px Arial';
                    ctx.textAlign = 'center';
                    ctx.fillText(dataset.label, width / 2, 20);
                }
            }
        }

        // Create Chart.js-compatible interface
        window.Chart = SimpleChart;
        console.log('Simple Chart implementation loaded');
    </script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
            min-height: 100vh;
        }

        .dashboard {
            max-width: 1400px;
            margin: 0 auto;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
            color: white;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            background: #4CAF50;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }

        .status-indicator.disconnected {
            background: #f44336;
            animation: none;
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.2s ease;
        }

        .card:hover {
            transform: translateY(-2px);
        }

        .card h3 {
            color: #5a67d8;
            margin-bottom: 15px;
            font-size: 1.2em;
            display: flex;
            align-items: center;
        }

        .card h3::before {
            content: "📊";
            margin-right: 8px;
        }

        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }

        .metric:last-child {
            border-bottom: none;
        }

        .metric-label {
            font-weight: 500;
            color: #666;
        }

        .metric-value {
            font-weight: bold;
            color: #333;
            font-family: 'Courier New', monospace;
        }

        .chart-container {
            grid-column: 1 / -1;
            height: 400px;
            position: relative;
        }

        .chart-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            grid-column: 1 / -1;
        }

        .chart-grid .card {
            min-height: 300px;
        }

        .chart-grid canvas {
            max-height: 250px !important;
            width: 100% !important;
        }

        .requests-table {
            grid-column: 1 / -1;
        }

        .table-container {
            max-height: 300px;
            overflow-y: auto;
            border-radius: 8px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
            font-size: 0.9em;
        }

        th {
            background: #f5f5f5;
            font-weight: 600;
            color: #555;
            position: sticky;
            top: 0;
        }

        .status-success {
            color: #4CAF50;
            font-weight: bold;
        }

        .status-error {
            color: #f44336;
            font-weight: bold;
        }

        .connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 15px;
            border-radius: 20px;
            color: white;
            font-weight: bold;
            font-size: 0.9em;
            z-index: 1000;
            transition: all 0.3s ease;
        }

        .connected {
            background: #4CAF50;
        }

        .disconnected {
            background: #f44336;
        }

        .connecting {
            background: #ff9800;
        }

        .loading {
            text-align: center;
            padding: 50px;
            color: white;
        }

        .error-message {
            background: rgba(244, 67, 54, 0.9);
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            display: none;
        }

        .retry-button {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            margin-left: 10px;
        }

        .retry-button:hover {
            background: #45a049;
        }

        .controls {
            text-align: center;
            margin: 20px 0;
        }

        .controls button {
            background: rgba(255, 255, 255, 0.2);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.3);
            padding: 10px 20px;
            border-radius: 20px;
            cursor: pointer;
            margin: 0 10px;
            transition: all 0.3s ease;
        }

        .controls button:hover {
            background: rgba(255, 255, 255, 0.3);
        }

        .stats-summary {
            grid-column: 1 / -1;
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            text-align: center;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
        }

        .stats-row {
            display: flex;
            justify-content: space-around;
            margin-top: 15px;
        }

        .stat-item {
            text-align: center;
        }

        .stat-value {
            font-size: 2em;
            font-weight: bold;
            display: block;
        }

        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
    </style>
</head>
<body>
    <div class="connection-status" id="connectionStatus">Connecting...</div>

    <div class="dashboard">
        <div class="header">
            <h1><span class="status-indicator" id="statusIndicator"></span>eSurge Real-time Dashboard</h1>
            <p>Live monitoring of inference engine performance</p>
        </div>

        <div class="controls">
            <button onclick="dashboard.toggleAutoRefresh()">⏸️ Pause</button>
            <button onclick="dashboard.clearData()">🗑️ Clear Data</button>
            <button onclick="dashboard.exportData()">💾 Export</button>
            <button onclick="dashboard.reconnect()">🔄 Reconnect</button>
        </div>

        <div class="error-message" id="errorMessage">
            <span id="errorText"></span>
            <button class="retry-button" onclick="dashboard.reconnect()">Retry</button>
        </div>

        <div class="grid" id="metricsGrid">
            <div class="loading">
                <h3>🔄 Connecting to metrics stream...</h3>
                <p>Please wait while we establish connection</p>
            </div>
        </div>
    </div>

    <script>
        class eSurgeDashboard {
            constructor() {
                this.ws = null;
                this.charts = {};
                this.metricsHistory = {
                    throughput: [],
                    latency: [],
                    requests: [],
                    timestamps: []
                };
                this.maxDataPoints = 50; // Reduced for cleaner charts
                this.reconnectAttempts = 0;
                this.maxReconnectAttempts = 10;
                this.reconnectDelay = 1000;
                this.autoRefresh = true;
                this.lastDataUpdate = Date.now();

                this.connect();
                this.startHealthCheck();
            }

            connect() {
                try {
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = `${protocol}//${window.location.host}/ws`;

                    console.log(`Connecting to: ${wsUrl}`);
                    this.updateConnectionStatus('connecting');

                    this.ws = new WebSocket(wsUrl);

                    this.ws.onopen = () => {
                        console.log('Connected to eSurge metrics stream');
                        this.updateConnectionStatus('connected');
                        this.reconnectAttempts = 0;
                        this.hideError();
                    };

                    this.ws.onmessage = (event) => {
                        try {
                            const data = JSON.parse(event.data);
                            this.lastDataUpdate = Date.now();
                            if (this.autoRefresh) {
                                this.updateDashboard(data);
                            }
                        } catch (error) {
                            console.error('Error parsing message:', error);
                            this.showError('Error parsing server data');
                        }
                    };

                    this.ws.onclose = (event) => {
                        console.log('Disconnected from metrics stream', event);
                        this.updateConnectionStatus('disconnected');
                        this.scheduleReconnect();
                    };

                    this.ws.onerror = (error) => {
                        console.error('WebSocket error:', error);
                        this.updateConnectionStatus('disconnected');
                        this.showError('Connection error occurred');
                    };

                } catch (error) {
                    console.error('Error creating WebSocket:', error);
                    this.showError('Failed to create connection');
                    this.scheduleReconnect();
                }
            }

            scheduleReconnect() {
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    const delay = this.reconnectDelay * Math.pow(1.5, this.reconnectAttempts - 1);

                    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
                    setTimeout(() => this.connect(), delay);
                } else {
                    this.showError('Max reconnection attempts reached. Click Retry to try again.');
                }
            }

            reconnect() {
                this.reconnectAttempts = 0;
                this.hideError();
                if (this.ws) {
                    this.ws.close();
                }
                setTimeout(() => this.connect(), 100);
            }

            updateConnectionStatus(status) {
                const statusElement = document.getElementById('connectionStatus');
                const indicatorElement = document.getElementById('statusIndicator');

                statusElement.className = `connection-status ${status}`;

                switch (status) {
                    case 'connected':
                        statusElement.textContent = '🟢 Connected';
                        indicatorElement.className = 'status-indicator';
                        break;
                    case 'connecting':
                        statusElement.textContent = '🟡 Connecting...';
                        indicatorElement.className = 'status-indicator';
                        break;
                    case 'disconnected':
                        statusElement.textContent = '🔴 Disconnected';
                        indicatorElement.className = 'status-indicator disconnected';
                        break;
                }
            }

            showError(message) {
                const errorDiv = document.getElementById('errorMessage');
                const errorText = document.getElementById('errorText');
                errorText.textContent = message;
                errorDiv.style.display = 'block';
            }

            hideError() {
                const errorDiv = document.getElementById('errorMessage');
                errorDiv.style.display = 'none';
            }

            updateDashboard(data) {
                try {
                    const grid = document.getElementById('metricsGrid');

                    if (data && data.system) {
                        this.updateMetricsHistory(data);

                        // Check if this is the first render or if we need to rebuild the DOM
                        const isFirstRender = !document.getElementById('throughputChart');

                        if (isFirstRender) {
                            // Initial render - create the full dashboard
                            grid.innerHTML = this.generateDashboardHTML(data);
                            // Create charts after DOM is ready
                            setTimeout(() => this.updateCharts(), 500);
                        } else {
                            // Update existing dashboard without rebuilding DOM
                            this.updateDashboardData(data);
                            // Update charts with new data
                            this.updateCharts();
                        }
                    }
                } catch (error) {
                    console.error('Error updating dashboard:', error);
                    this.showError('Error updating dashboard display');
                }
            }

            updateMetricsHistory(data) {
                const now = new Date().toLocaleTimeString();
                const throughput = data.system?.average_throughput || 0;
                const latency = (data.system?.average_latency || 0) * 1000; // Convert to ms
                const requests = data.system?.requests_per_second || 0;

                this.metricsHistory.timestamps.push(now);
                this.metricsHistory.throughput.push(throughput);
                this.metricsHistory.latency.push(latency);
                this.metricsHistory.requests.push(requests);


                // Keep only last N data points
                if (this.metricsHistory.timestamps.length > this.maxDataPoints) {
                    Object.keys(this.metricsHistory).forEach(key => {
                        this.metricsHistory[key].shift();
                    });
                }
            }

            updateDashboardData(data) {
                // Update only the dynamic data without rebuilding DOM
                const system = data.system || {};
                const scheduler = data.scheduler || {};
                const runner = data.runner || {};
                const cache = data.cache || {};

                // Update system metrics
                this.updateElementText('.stat-value', [
                    (system.requests_per_second || 0).toFixed(1),
                    (system.average_throughput || 0).toFixed(0),
                    ((system.average_latency || 0) * 1000).toFixed(0) + 'ms',
                    system.total_requests_completed || 0
                ]);

                // Update individual metric cards
                this.updateMetricCard('System Performance', {
                    'Requests/sec': (system.requests_per_second || 0).toFixed(2),
                    'Avg Latency': ((system.average_latency || 0) * 1000).toFixed(1) + 'ms',
                    'Avg TTFT': ((system.average_ttft || 0) * 1000).toFixed(1) + 'ms',
                    'Throughput': (system.average_throughput || 0).toFixed(1) + ' tok/s',
                    'Success Rate': this.calculateSuccessRate(system) + '%'
                });

                this.updateMetricCard('Scheduler Status', {
                    'Waiting Requests': scheduler.num_waiting_requests || 0,
                    'Running Requests': scheduler.num_running_requests || 0,
                    'Scheduled Tokens': scheduler.num_scheduled_tokens || 0,
                    'Batch Size': scheduler.batch_size || 0,
                    'Schedule Time': ((scheduler.schedule_time || 0) * 1000).toFixed(2) + 'ms'
                });

                this.updateMetricCard('Model Runner', {
                    'Execution Time': ((runner.execution_time || 0) * 1000).toFixed(2) + 'ms',
                    'Batch Size': runner.batch_size || 0,
                    'Tokens Processed': runner.num_tokens || 0,
                    'Instant Throughput': (runner.tokens_per_second || 0).toFixed(1) + ' tok/s'
                });

                this.updateMetricCard('Cache Status', {
                    'Total Pages': cache.total_pages || 0,
                    'Used Pages': cache.used_pages || 0,
                    'Utilization': cache.total_pages ? ((cache.used_pages / cache.total_pages) * 100).toFixed(1) + '%' : '0%',
                    'Hit Rate': ((cache.cache_hit_rate || 0) * 100).toFixed(1) + '%'
                });

                // Update recent requests table if needed
                if (data.recent_requests) {
                    this.updateRequestsTable(data.recent_requests);
                }
            }

            updateElementText(selector, values) {
                const elements = document.querySelectorAll(selector);
                values.forEach((value, index) => {
                    if (elements[index]) {
                        elements[index].textContent = value;
                    }
                });
            }

            updateMetricCard(cardTitle, metrics) {
                const cards = document.querySelectorAll('.card');
                for (const card of cards) {
                    const titleElement = card.querySelector('h3');
                    if (titleElement && titleElement.textContent.includes(cardTitle.split(' ')[0])) {
                        const metricElements = card.querySelectorAll('.metric');
                        Object.entries(metrics).forEach(([label, value], index) => {
                            if (metricElements[index]) {
                                const valueElement = metricElements[index].querySelector('.metric-value');
                                if (valueElement) {
                                    valueElement.textContent = value;
                                }
                            }
                        });
                        break;
                    }
                }
            }

            updateRequestsTable(recentRequests) {
                const tbody = document.querySelector('.requests-table tbody');
                if (tbody) {
                    tbody.innerHTML = recentRequests.slice(-10).map(req => `
                        <tr>
                            <td title="${req.request_id}">${req.request_id.substring(0, 12)}...</td>
                            <td class="${req.error ? 'status-error' : 'status-success'}">
                                ${req.error ? '❌' : '✅'}
                            </td>
                            <td>${req.total_latency ? (req.total_latency * 1000).toFixed(1) + 'ms' : 'N/A'}</td>
                            <td>${req.time_to_first_token ? (req.time_to_first_token * 1000).toFixed(1) + 'ms' : 'N/A'}</td>
                            <td>${req.generated_tokens || 0}</td>
                            <td>${req.tokens_per_second ? req.tokens_per_second.toFixed(1) + ' tok/s' : 'N/A'}</td>
                            <td>${req.finish_reason || 'N/A'}</td>
                        </tr>
                    `).join('');
                }
            }

            generateDashboardHTML(data) {
                const system = data.system || {};
                const scheduler = data.scheduler || {};
                const runner = data.runner || {};
                const cache = data.cache || {};

                return `
                    <!-- Summary Stats -->
                    <div class="stats-summary">
                        <h3>📈 Performance Overview</h3>
                        <div class="stats-row">
                            <div class="stat-item">
                                <span class="stat-value">${(system.requests_per_second || 0).toFixed(1)}</span>
                                <span class="stat-label">Requests/sec</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-value">${(system.average_throughput || 0).toFixed(0)}</span>
                                <span class="stat-label">Tokens/sec</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-value">${((system.average_latency || 0) * 1000).toFixed(0)}ms</span>
                                <span class="stat-label">Avg Latency</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-value">${system.total_requests_completed || 0}</span>
                                <span class="stat-label">Completed</span>
                            </div>
                        </div>
                    </div>

                    <!-- System Metrics -->
                    <div class="card">
                        <h3>🖥️ System Performance</h3>
                        <div class="metric">
                            <span class="metric-label">Requests/sec</span>
                            <span class="metric-value">${(system.requests_per_second || 0).toFixed(2)}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Avg Latency</span>
                            <span class="metric-value">${((system.average_latency || 0) * 1000).toFixed(1)}ms</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Avg TTFT</span>
                            <span class="metric-value">${((system.average_ttft || 0) * 1000).toFixed(1)}ms</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Throughput</span>
                            <span class="metric-value">${(system.average_throughput || 0).toFixed(1)} tok/s</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Success Rate</span>
                            <span class="metric-value">${this.calculateSuccessRate(system)}%</span>
                        </div>
                    </div>

                    <!-- Scheduler Metrics -->
                    <div class="card">
                        <h3>⚡ Scheduler Status</h3>
                        <div class="metric">
                            <span class="metric-label">Waiting Requests</span>
                            <span class="metric-value">${scheduler.num_waiting_requests || 0}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Running Requests</span>
                            <span class="metric-value">${scheduler.num_running_requests || 0}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Scheduled Tokens</span>
                            <span class="metric-value">${scheduler.num_scheduled_tokens || 0}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Batch Size</span>
                            <span class="metric-value">${scheduler.batch_size || 0}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Schedule Time</span>
                            <span class="metric-value">${((scheduler.schedule_time || 0) * 1000).toFixed(2)}ms</span>
                        </div>
                    </div>

                    <!-- Runner Metrics -->
                    <div class="card">
                        <h3>🚀 Model Runner</h3>
                        <div class="metric">
                            <span class="metric-label">Execution Time</span>
                            <span class="metric-value">${((runner.execution_time || 0) * 1000).toFixed(2)}ms</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Batch Size</span>
                            <span class="metric-value">${runner.batch_size || 0}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Tokens Processed</span>
                            <span class="metric-value">${runner.num_tokens || 0}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Instant Throughput</span>
                            <span class="metric-value">${(runner.tokens_per_second || 0).toFixed(1)} tok/s</span>
                        </div>
                    </div>

                    <!-- Cache Metrics -->
                    <div class="card">
                        <h3>💾 Cache Status</h3>
                        <div class="metric">
                            <span class="metric-label">Total Pages</span>
                            <span class="metric-value">${cache.total_pages || 0}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Used Pages</span>
                            <span class="metric-value">${cache.used_pages || 0}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Utilization</span>
                            <span class="metric-value">${cache.total_pages ? ((cache.used_pages / cache.total_pages) * 100).toFixed(1) : 0}%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Hit Rate</span>
                            <span class="metric-value">${((cache.cache_hit_rate || 0) * 100).toFixed(1)}%</span>
                        </div>
                    </div>

                    <!-- Charts -->
                    <div class="chart-grid">
                        <div class="card">
                            <h3>📈 Throughput Over Time</h3>
                            <canvas id="throughputChart" width="400" height="200"></canvas>
                        </div>
                        <div class="card">
                            <h3>⏱️ Latency Over Time</h3>
                            <canvas id="latencyChart" width="400" height="200"></canvas>
                        </div>
                    </div>

                    <!-- Recent Requests Table -->
                    <div class="card requests-table">
                        <h3>📋 Recent Requests</h3>
                        <div class="table-container">
                            <table>
                                <thead>
                                    <tr>
                                        <th>Request ID</th>
                                        <th>Status</th>
                                        <th>Latency</th>
                                        <th>TTFT</th>
                                        <th>Tokens</th>
                                        <th>Throughput</th>
                                        <th>Finish Reason</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${(data.recent_requests || []).slice(-10).map(req => `
                                        <tr>
                                            <td title="${req.request_id}">${req.request_id.substring(0, 12)}...</td>
                                            <td class="${req.error ? 'status-error' : 'status-success'}">
                                                ${req.error ? '❌' : '✅'}
                                            </td>
                                            <td>${req.total_latency ? (req.total_latency * 1000).toFixed(1) + 'ms' : 'N/A'}</td>
                                            <td>${req.time_to_first_token ? (req.time_to_first_token * 1000).toFixed(1) + 'ms' : 'N/A'}</td>
                                            <td>${req.generated_tokens || 0}</td>
                                            <td>${req.tokens_per_second ? req.tokens_per_second.toFixed(1) + ' tok/s' : 'N/A'}</td>
                                            <td>${req.finish_reason || 'N/A'}</td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                `;
            }
            calculateSuccessRate(system) {
                const total = (system.total_requests_completed || 0) + (system.total_requests_failed || 0);
                if (total === 0) return 100;
                return ((system.total_requests_completed || 0) / total * 100).toFixed(1);
            }
            updateCharts() {
                // Add a delay to ensure DOM elements are available and have proper dimensions
                setTimeout(() => {
                    this.updateThroughputChart();
                    this.updateLatencyChart();
                }, 100);
            }
            updateThroughputChart() {
                const canvas = document.getElementById('throughputChart');
                if (!canvas) {
                    return;
                }

                // Get canvas context
                const ctx = canvas.getContext('2d');
                if (!ctx) {
                    return;
                }

                // Validate data before creating/updating chart
                if (!this.metricsHistory.timestamps.length || !this.metricsHistory.throughput.length) {
                    return;
                }

                // Check if Chart is available
                if (typeof Chart === 'undefined') {
                    return;
                }

                const chartData = {
                    labels: this.metricsHistory.timestamps,
                    datasets: [{
                        label: 'Tokens/second',
                        data: this.metricsHistory.throughput,
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0, // No dots on line
                        pointHoverRadius: 4,
                        borderWidth: 2
                    }]
                };

                try {
                    // Update existing chart or create new one
                    if (this.charts.throughput && this.charts.throughput.update) {
                        this.charts.throughput.update(chartData);
                    } else {
                        this.charts.throughput = new Chart(ctx, {
                            type: 'line',
                            data: chartData,
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                interaction: {
                                    intersect: false,
                                    mode: 'index'
                                },
                                scales: {
                                    y: {
                                        beginAtZero: true,
                                        title: {
                                            display: true,
                                            text: 'Tokens/second'
                                        }
                                    },
                                    x: {
                                        title: {
                                            display: true,
                                            text: 'Time'
                                        }
                                    }
                                },
                                plugins: {
                                    legend: {
                                        display: false
                                    }
                                }
                            }
                        });
                    }
                } catch (e) {
                    console.error('Error updating throughput chart:', e);
                }
            }
            updateLatencyChart() {
                const canvas = document.getElementById('latencyChart');
                if (!canvas) {
                    return;
                }

                // Get canvas context
                const ctx = canvas.getContext('2d');
                if (!ctx) {
                    return;
                }

                // Validate data before creating/updating chart
                if (!this.metricsHistory.timestamps.length || !this.metricsHistory.latency.length) {
                    return;
                }

                // Check if Chart is available
                if (typeof Chart === 'undefined') {
                    return;
                }

                const chartData = {
                    labels: this.metricsHistory.timestamps,
                    datasets: [{
                        label: 'Latency (ms)',
                        data: this.metricsHistory.latency,
                        borderColor: '#764ba2',
                        backgroundColor: 'rgba(118, 75, 162, 0.1)',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0, // No dots on line
                        pointHoverRadius: 4,
                        borderWidth: 2
                    }]
                };

                try {
                    // Update existing chart or create new one
                    if (this.charts.latency && this.charts.latency.update) {
                        this.charts.latency.update(chartData);
                    } else {
                        this.charts.latency = new Chart(ctx, {
                            type: 'line',
                            data: chartData,
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                interaction: {
                                    intersect: false,
                                    mode: 'index'
                                },
                                scales: {
                                    y: {
                                        beginAtZero: true,
                                        title: {
                                            display: true,
                                            text: 'Latency (ms)'
                                        }
                                    },
                                    x: {
                                        title: {
                                            display: true,
                                            text: 'Time'
                                        }
                                    }
                                },
                                plugins: {
                                    legend: {
                                        display: false
                                    }
                                }
                            }
                        });
                    }
                } catch (e) {
                    console.error('Error updating latency chart:', e);
                }
            }

            toggleAutoRefresh() {
                this.autoRefresh = !this.autoRefresh;
                const btn = event.target;
                btn.textContent = this.autoRefresh ? '⏸️ Pause' : '▶️ Resume';
                btn.style.background = this.autoRefresh ? 'rgba(255, 255, 255, 0.2)' : 'rgba(255, 152, 0, 0.8)';
            }

            clearData() {
                this.metricsHistory = {
                    throughput: [],
                    latency: [],
                    requests: [],
                    timestamps: []
                };
                // Destroy existing charts when clearing data
                if (this.charts.throughput) {
                    try {
                        this.charts.throughput.destroy();
                        this.charts.throughput = null;
                    } catch (e) {
                        console.warn('Error destroying throughput chart:', e);
                    }
                }
                if (this.charts.latency) {
                    try {
                        this.charts.latency.destroy();
                        this.charts.latency = null;
                    } catch (e) {
                        console.warn('Error destroying latency chart:', e);
                    }
                }
            }

            exportData() {
                const data = {
                    exported_at: new Date().toISOString(),
                    metrics_history: this.metricsHistory
                };

                const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `esurge_metrics_${new Date().toISOString().slice(0, 19)}.json`;
                a.click();
                URL.revokeObjectURL(url);
            }

            startHealthCheck() {
                setInterval(() => {
                    const timeSinceLastUpdate = Date.now() - this.lastDataUpdate;

                    // If no data for 10 seconds, show warning
                    if (timeSinceLastUpdate > 10000 && this.ws && this.ws.readyState === WebSocket.OPEN) {
                        console.warn('No data received for 10 seconds');
                        this.showError('No data received recently. Check if eSurge is running.');
                    }
                }, 5000);
            }
        }

        // Global dashboard instance
        let dashboard;

        // Initialize dashboard when page loads
        document.addEventListener('DOMContentLoaded', () => {
            dashboard = new eSurgeDashboard();
            window.dashboard = dashboard; // Make globally accessible


            // Handle page visibility changes
            document.addEventListener('visibilitychange', () => {
                if (document.hidden) {
                    console.log('Page hidden, pausing updates');
                } else {
                    console.log('Page visible, resuming updates');
                    dashboard.lastDataUpdate = Date.now(); // Reset health check
                }
            });
        });

        // Handle page unload
        window.addEventListener('beforeunload', () => {
            if (dashboard && dashboard.ws) {
                dashboard.ws.close();
            }
        });
    </script>
</body>
</html>
"""  # noqa


class eSurgeWebDashboard:
    """Improved FastAPI-based web dashboard with enhanced stability."""

    def __init__(self, host: str = "localhost", port: int = 8080, debug: bool = False):
        """Initialize improved web dashboard."""
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")

        self.host = host
        self.port = port
        self.debug = debug
        self.app = FastAPI(title="eSurge Dashboard", description="Real-time monitoring dashboard", version="2.0.0")
        self.connected_clients: set[WebSocket] = set()
        self.logger = logging.getLogger("esurge.dashboard")

        # Add CORS middleware for cross-origin requests
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._setup_routes()
        self._setup_error_handlers()

    def _setup_error_handlers(self) -> None:
        """Setup global error handlers."""

        @self.app.exception_handler(Exception)
        async def global_exception_handler(request: Request, exc: Exception):
            self.logger.error(f"Global exception: {exc}", exc_info=True)
            return JSONResponse(status_code=500, content={"error": "Internal server error", "detail": str(exc)})

    def _setup_routes(self) -> None:
        """Setup FastAPI routes with improved error handling."""

        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard():
            """Serve the improved dashboard HTML."""
            return DASHBOARD_HTML_FIXED

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            collector = get_metrics_collector()
            return {
                "status": "healthy",
                "metrics_collector": collector is not None,
                "connected_clients": len(self.connected_clients),
                "timestamp": time.time(),
            }

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """Improved WebSocket endpoint with better error handling."""
            await websocket.accept()
            self.connected_clients.add(websocket)
            client_id = id(websocket)

            self.logger.info(f"Client {client_id} connected. Total clients: {len(self.connected_clients)}")

            try:
                while True:
                    # Send metrics every 2 seconds to reduce chart flashing
                    await asyncio.sleep(2.0)

                    try:
                        collector = get_metrics_collector()
                        if collector:
                            metrics_data = self._get_dashboard_data(collector)
                            await websocket.send_text(json.dumps(metrics_data, default=str))
                        else:
                            # Send empty data if no collector
                            await websocket.send_text(
                                json.dumps({"error": "No metrics collector available", "timestamp": time.time()})
                            )

                    except WebSocketDisconnect:
                        break
                    except Exception as e:
                        self.logger.error(f"Error sending data to client {client_id}: {e}")
                        break

            except WebSocketDisconnect:
                pass
            except Exception as e:
                self.logger.error(f"WebSocket error for client {client_id}: {e}")
            finally:
                self.connected_clients.discard(websocket)
                self.logger.info(f"Client {client_id} disconnected. Total clients: {len(self.connected_clients)}")

        @self.app.get("/api/metrics")
        async def get_metrics():
            """API endpoint for current metrics with error handling."""
            try:
                collector = get_metrics_collector()
                if not collector:
                    return JSONResponse(status_code=503, content={"error": "Metrics collector not initialized"})

                return self._get_dashboard_data(collector)

            except Exception as e:
                self.logger.error(f"Error getting metrics: {e}")
                return JSONResponse(status_code=500, content={"error": "Failed to get metrics", "detail": str(e)})

        @self.app.get("/api/export")
        async def export_metrics():
            """Export detailed metrics as JSON."""
            try:
                collector = get_metrics_collector()
                if not collector:
                    return JSONResponse(status_code=503, content={"error": "Metrics collector not initialized"})

                # Export comprehensive metrics
                with collector._lock:
                    export_data = {
                        "exported_at": time.time(),
                        "system_metrics": asdict(collector.get_system_metrics()),
                        "completed_requests": [asdict(req) for req in list(collector.completed_requests)],
                        "scheduler_metrics": [asdict(m) for m in list(collector.scheduler_metrics)],
                        "runner_metrics": [asdict(m) for m in list(collector.runner_metrics)],
                        "cache_metrics": [asdict(m) for m in list(collector.cache_metrics)],
                        "active_requests": len(collector.request_metrics),
                    }

                return export_data

            except Exception as e:
                self.logger.error(f"Error exporting metrics: {e}")
                return JSONResponse(status_code=500, content={"error": "Failed to export metrics", "detail": str(e)})

    def _get_dashboard_data(self, collector) -> dict[str, Any]:
        """Get dashboard data with improved error handling."""
        try:
            system_metrics = collector.get_system_metrics()

            with collector._lock:
                # Get latest metrics from each component
                latest_scheduler = collector.scheduler_metrics[-1] if collector.scheduler_metrics else None
                latest_runner = collector.runner_metrics[-1] if collector.runner_metrics else None
                latest_cache = collector.cache_metrics[-1] if collector.cache_metrics else None

                # Get recent completed requests (last 10)
                recent_requests = [asdict(req) for req in list(collector.completed_requests)[-10:]]

            return {
                "timestamp": time.time(),
                "system": asdict(system_metrics),
                "scheduler": asdict(latest_scheduler) if latest_scheduler else None,
                "runner": asdict(latest_runner) if latest_runner else None,
                "cache": asdict(latest_cache) if latest_cache else None,
                "recent_requests": recent_requests,
                "active_requests": len(collector.request_metrics),
                "connected_clients": len(self.connected_clients),
            }

        except Exception as e:
            self.logger.error(f"Error getting dashboard data: {e}")
            return {"error": f"Error getting dashboard data: {e}", "timestamp": time.time()}

    async def broadcast_metrics(self) -> None:
        """Broadcast metrics to all connected clients with error handling."""
        if not self.connected_clients:
            return

        try:
            collector = get_metrics_collector()
            if not collector:
                return

            metrics_data = self._get_dashboard_data(collector)
            message = json.dumps(metrics_data, default=str)

            # Send to all connected clients
            disconnected = set()
            for client in self.connected_clients.copy():
                try:
                    await client.send_text(message)
                except Exception as e:
                    self.logger.warning(f"Failed to send to client: {e}")
                    disconnected.add(client)

            # Remove disconnected clients
            self.connected_clients -= disconnected

        except Exception as e:
            self.logger.error(f"Error broadcasting metrics: {e}")

    def run(self, **kwargs) -> None:
        """Run the improved dashboard server."""
        logger.info(f" Starting eSurge Dashboard v2.0 at http://{self.host}:{self.port}")
        logger.info(" Enhanced with improved stability and error handling")
        logger.info(f" Health check available at: http://{self.host}:{self.port}/health")
        logger.info(f" API endpoint: http://{self.host}:{self.port}/api/metrics")

        # Configure logging
        log_level = "debug" if self.debug else "info"
        kwargs.pop("log_level", None)
        uvicorn.run(self.app, host=self.host, port=self.port, log_level=log_level, **kwargs)


def create_dashboard(host: str = "localhost", port: int = 8080, debug: bool = False) -> eSurgeWebDashboard:
    """Create and return a new improved web dashboard instance."""
    return eSurgeWebDashboard(host=host, port=port, debug=debug)


def create_dashboard_fixed(host: str = "localhost", port: int = 8080, debug: bool = False) -> eSurgeWebDashboard:
    """Create and return a new improved web dashboard instance."""
    return eSurgeWebDashboard(host=host, port=port, debug=debug)
