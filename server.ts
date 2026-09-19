import express from "express";
import { spawn, ChildProcess } from "child_process";
import { createProxyMiddleware } from "http-proxy-middleware";
import http from "http";

const PORT = 3000;
const FLASK_PORT = 5000;
const FLASK_URL = `http://127.0.0.1:${FLASK_PORT}`;

let flaskProcess: ChildProcess | null = null;
let flaskReady = false;

function startFlaskBackend(): Promise<void> {
  return new Promise((resolve) => {
    console.log(`[Voice Sentinel] Spawning Python ML Backend on port ${FLASK_PORT}...`);
    
    flaskProcess = spawn("python3", ["app.py"], {
      env: {
        ...process.env,
        FLASK_PORT: String(FLASK_PORT),
        PYTHONUNBUFFERED: "1"
      },
      stdio: ["ignore", "pipe", "pipe"]
    });

    flaskProcess.stdout?.on("data", (data) => {
      const msg = data.toString().trim();
      console.log(`[Flask] ${msg}`);
      if (msg.includes("Starting Voice Sentinel") || msg.includes("Running on")) {
        flaskReady = true;
      }
    });

    flaskProcess.stderr?.on("data", (data) => {
      console.error(`[Flask Err] ${data.toString().trim()}`);
    });

    flaskProcess.on("exit", (code) => {
      console.log(`[Voice Sentinel] Flask process exited with code ${code}`);
      flaskReady = false;
    });

    // Check health endpoint repeatedly
    let retries = 0;
    const maxRetries = 25;
    const checkInterval = setInterval(() => {
      retries++;
      const req = http.get(`${FLASK_URL}/health`, (res) => {
        if (res.statusCode === 200) {
          flaskReady = true;
          clearInterval(checkInterval);
          console.log("[Voice Sentinel] Flask ML Backend is fully responsive!");
          resolve();
        }
      });

      req.on("error", () => {
        if (retries >= maxRetries) {
          clearInterval(checkInterval);
          console.warn("[Voice Sentinel] Flask health check timeout, proceeding anyway...");
          resolve();
        }
      });
    }, 600);
  });
}

async function startServer() {
  await startFlaskBackend();

  const app = express();

  // Proxy all requests directly to Python Flask backend
  const proxy = createProxyMiddleware({
    target: FLASK_URL,
    changeOrigin: true,
    ws: true,
    on: {
      error: (err, req, res) => {
        console.error("[Proxy Error]", err.message);
        if (res && "headersSent" in res && !(res as any).headersSent && "writeHead" in res) {
          (res as any).writeHead(502, { "Content-Type": "text/html" });
          (res as any).end(`
            <!DOCTYPE html>
            <html>
            <head><title>Voice Sentinel Starting</title><meta http-equiv="refresh" content="3"></head>
            <body style="background:#090D16;color:#F1F5F9;font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;">
              <div style="text-align:center;padding:2rem;background:#111827;border:1px solid #1E293B;border-radius:12px;">
                <h2 style="color:#06B6D4;">VOICE SENTINEL</h2>
                <p style="color:#94A3B8;margin-top:0.5rem;">Initializing Audio Processing Pipeline & Python Machine Learning Engine...</p>
                <p style="color:#64748B;font-size:0.85rem;margin-top:1rem;">Refreshing in 3 seconds...</p>
              </div>
            </body>
            </html>
          `);
        }
      }
    }
  });

  app.use(proxy);

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`[Voice Sentinel] Production Gateway running on http://0.0.0.0:${PORT}`);
  });
}

// Clean termination handling
process.on("SIGTERM", () => {
  if (flaskProcess) flaskProcess.kill();
  process.exit(0);
});

process.on("SIGINT", () => {
  if (flaskProcess) flaskProcess.kill();
  process.exit(0);
});

startServer();
