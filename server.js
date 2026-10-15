
"use strict";

const express = require("express");
const cors = require("cors");
const axios = require("axios");
const { StringDecoder } = require("string_decoder");
const crypto = require("crypto");

const app = express();

const PORT = Number(process.env.PORT) || 3000;
const NVIDIA_API_BASE = (
  process.env.NVIDIA_API_BASE ||
  "https://integrate.api.nvidia.com/v1"
).replace(/\/+$/, "");

const NVIDIA_API_KEY = process.env.NVIDIA_API_KEY || "";

const DEFAULT_MODEL =
  process.env.NVIDIA_MODEL ||
  "nvidia/nemotron-3-super-120b-a12b";

// Parâmetros recomendados pela NVIDIA para este modelo.
const DEFAULT_TEMPERATURE = 1.0;
const DEFAULT_TOP_P = 0.95;
const DEFAULT_MAX_TOKENS = 8192;

// Defina ENABLE_THINKING=false no Render para desativar o raciocínio.
const ENABLE_THINKING =
  String(process.env.ENABLE_THINKING || "true").toLowerCase() === "true";

const REQUEST_TIMEOUT = Math.max(
  30000,
  Number(process.env.REQUEST_TIMEOUT_MS) || 300000
);

app.disable("x-powered-by");
app.use(cors());
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true, limit: "10mb" }));

// Compatibilidade com nomes de modelos usados pelo Janitor AI.
const MODEL_ALIASES = new Set([
  "gpt-3.5-turbo",
  "gpt-4",
  "gpt-4o",
  "gpt-4o-mini",
  "gpt-4.1",
  "claude-3-opus",
  "claude-3-sonnet",
  "claude-3-haiku",
  "gemini-pro",
]);

function resolveModel(requestedModel) {
  if (!requestedModel || typeof requestedModel !== "string") {
    return DEFAULT_MODEL;
  }

  const model = requestedModel.trim();

  if (
    model === DEFAULT_MODEL ||
    model === "nvidia/nemotron-3-super"
  ) {
    return DEFAULT_MODEL;
  }

  // Os nomes de compatibilidade do Janitor apontam para o modelo escolhido.
  if (MODEL_ALIASES.has(model)) {
    return DEFAULT_MODEL;
  }

  // Permite selecionar explicitamente outro modelo NVIDIA.
  if (model.startsWith("nvidia/")) {
    return model;
  }

  return DEFAULT_MODEL;
}

function clampNumber(value, fallback, min, max) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return fallback;
  }

  return Math.min(max, Math.max(min, number));
}

function makeId(prefix = "chatcmpl") {
  return `${prefix}-${crypto.randomBytes(12).toString("hex")}`;
}

function log(message, details) {
  if (details !== undefined) {
    console.log(`[Proxy] ${message}`, details);
  } else {
    console.log(`[Proxy] ${message}`);
  }
}

function getNvidiaHeaders() {
  return {
    Authorization: `Bearer ${NVIDIA_API_KEY}`,
    "Content-Type": "application/json",
    Accept: "application/json",
  };
}

function getErrorDetails(error) {
  const status = error.response?.status;
  const data = error.response?.data;

  let detail = "";

  if (typeof data === "string") {
    detail = data;
  } else if (data && typeof data === "object") {
    detail = JSON.stringify(data);
  } else {
    detail = error.message || "Unknown upstream error";
  }

  // Evita colocar respostas enormes no log do Render.
  if (detail.length > 3000) {
    detail = detail.slice(0, 3000) + "...";
  }

  return { status, detail };
}

// Rotas básicas de verificação.
app.get("/", (req, res) => {
  res.json({
    status: "ok",
    service: "OpenAI to NVIDIA NIM Proxy",
    model: DEFAULT_MODEL,
  });
});

app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    service: "OpenAI to NVIDIA NIM Proxy",
    model: DEFAULT_MODEL,
    nvidia_api_configured: Boolean(NVIDIA_API_KEY),
    thinking_enabled: ENABLE_THINKING,
  });
});

app.get("/health/", (req, res) => {
  res.redirect(307, "/health");
});

// Lista de modelos no formato esperado pela API OpenAI.
app.get("/v1/models", (req, res) => {
  const models = [
    DEFAULT_MODEL,
    "nvidia/nemotron-3-super",
    ...MODEL_ALIASES,
  ];

  res.json({
    object: "list",
    data: [...new Set(models)].map((id) => ({
      id,
      object: "model",
      created: 0,
      owned_by: "nvidia",
    })),
  });
});

app.get("/v1/models/:model", (req, res) => {
  res.json({
    id: req.params.model,
    object: "model",
    created: 0,
    owned_by: "nvidia",
  });
});

// Processa um evento SSE completo recebido da NVIDIA.
function parseSseEvent(eventText) {
  const dataLines = eventText
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trim());

  if (dataLines.length === 0) {
    return null;
  }

  const data = dataLines.join("\n");

  if (!data || data === "[DONE]") {
    return { done: true };
  }

  try {
    return { json: JSON.parse(data) };
  } catch {
    log("Evento SSE inválido recebido; evento ignorado.");
    return null;
  }
}

function getRequestParameters(body) {
  const temperature = clampNumber(
    body.temperature,
    DEFAULT_TEMPERATURE,
    0,
    2
  );

  const topP = clampNumber(
    body.top_p,
    DEFAULT_TOP_P,
    0.01,
    1
  );

  const requestedTokens =
    body.max_completion_tokens ?? body.max_tokens;

  const maxTokens = Math.round(
    clampNumber(requestedTokens, DEFAULT_MAX_TOKENS, 1, 16384)
  );

  return { temperature, topP, maxTokens };
}

app.post("/v1/chat/completions", async (req, res) => {
  const body = req.body || {};
  const messages = body.messages;

  if (!NVIDIA_API_KEY) {
    return res.status(500).json({
      error: {
        message: "NVIDIA_API_KEY is not configured in Render.",
        type: "configuration_error",
      },
    });
  }

  if (!Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({
      error: {
        message: "The messages field must be a non-empty array.",
        type: "invalid_request_error",
      },
    });
  }

  const model = resolveModel(body.model);
  const stream = body.stream === true;
  const { temperature, topP, maxTokens } =
    getRequestParameters(body);

  const requestId = makeId();
  const created = Math.floor(Date.now() / 1000);

  const payload = {
    model,
    messages,
    temperature,
    top_p: topP,
    max_tokens: maxTokens,
    stream,
    // NVIDIA expects this configuration at the top level,
    // not nested inside an "extra_body" JSON property.
    chat_template_kwargs: {
      enable_thinking: ENABLE_THINKING,
    },
  };

  // Preserve supported optional parameters supplied by the client.
  if (typeof body.stop === "string" || Array.isArray(body.stop)) {
    payload.stop = body.stop;
  }

  if (typeof body.frequency_penalty === "number") {
    payload.frequency_penalty = clampNumber(
      body.frequency_penalty,
      0,
      -2,
      2
    );
  }

  if (typeof body.presence_penalty === "number") {
    payload.presence_penalty = clampNumber(
      body.presence_penalty,
      0,
      -2,
      2
    );
  }

  log("Chat completion request.", {
    model,
    stream,
    temperature,
    top_p: topP,
    max_tokens: maxTokens,
    thinking: ENABLE_THINKING,
    message_count: messages.length,
  });

  let upstream;

  try {
    upstream = await axios.post(
      `${NVIDIA_API_BASE}/chat/completions`,
      payload,
      {
        headers: {
          ...getNvidiaHeaders(),
          ...(stream ? { Accept: "text/event-stream" } : {}),
        },
        responseType: stream ? "stream" : "json",
        timeout: REQUEST_TIMEOUT,
        maxBodyLength: Infinity,
        maxContentLength: Infinity,
        validateStatus: (status) => status >= 200 && status < 300,
      }
    );
  } catch (error) {
    const { status, detail } = getErrorDetails(error);

    log("NVIDIA request failed.", { status, detail });

    const clientStatus =
      status && status >= 400 && status < 500 ? status : 502;

    return res.status(clientStatus).json({
      error: {
        message: "NVIDIA NIM request failed.",
        type: "upstream_error",
        upstream_status: status || null,
        details: detail,
      },
    });
  }

  // Non-streaming response.
  if (!stream) {
    const data = upstream.data;
    const choice = data?.choices?.[0];

    if (!choice) {
      log("NVIDIA returned no choices.", data);

      return res.status(502).json({
        error: {
          message: "NVIDIA returned an empty completion.",
          type: "upstream_error",
        },
      });
    }

    // Do not expose internal reasoning to Janitor.
    if (choice.message) {
      delete choice.message.reasoning_content;
    }

    if (!choice.message?.content) {
      log("Warning: completion content is empty.", {
        finish_reason: choice.finish_reason || null,
      });
    }

    data.model = model;

    return res.status(200).json(data);
  }

  // Streaming response: parse complete SSE events and forward only
  // user-facing content, not reasoning_content.
  res.status(200);
  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");
  res.flushHeaders?.();

  let sseBuffer = "";
  let streamEnded = false;
  let upstreamDestroyed = false;

  const decoder = new StringDecoder("utf8");
  const upstreamStream = upstream.data;

  function sendChunk(delta, finishReason = null) {
    if (res.writableEnded || res.destroyed) {
      return;
    }

    const chunk = {
      id: requestId,
      object: "chat.completion.chunk",
      created,
      model,
      choices: [
        {
          index: 0,
          delta,
          finish_reason: finishReason,
        },
      ],
    };

    res.write(`data: ${JSON.stringify(chunk)}\n\n`);
  }

  function finishStream() {
    if (streamEnded) {
      return;
    }

    streamEnded = true;

    if (!res.writableEnded && !res.destroyed) {
      res.write("data: [DONE]\n\n");
      res.end();
    }
  }

  function processSseEvent(eventText) {
    const parsed = parseSseEvent(eventText);

    if (!parsed) {
      return;
    }

    if (parsed.done) {
      finishStream();
      return;
    }

    const data = parsed.json;
    const choice = data?.choices?.[0];

    if (!choice) {
      return;
    }

    const delta = choice.delta || {};

    // Never forward model reasoning to the user-facing conversation.
    if (typeof delta.content === "string" && delta.content.length > 0) {
      sendChunk({ content: delta.content });
    }

    if (choice.finish_reason !== null &&
        choice.finish_reason !== undefined) {
      sendChunk({}, choice.finish_reason);
    }
  }

  function processBuffer(final = false) {
    let delimiterMatch;

    while ((delimiterMatch = /\r?\n\r?\n/.exec(sseBuffer))) {
      const eventText = sseBuffer.slice(0, delimiterMatch.index);
      sseBuffer = sseBuffer.slice(
        delimiterMatch.index + delimiterMatch[0].length
      );

      processSseEvent(eventText);

      if (streamEnded) {
        sseBuffer = "";
        return;
      }
    }

    if (final && sseBuffer.trim()) {
      processSseEvent(sseBuffer);
      sseBuffer = "";
    }
  }

  function destroyUpstream() {
    if (upstreamDestroyed) {
      return;
    }

    upstreamDestroyed = true;

    if (upstreamStream && !upstreamStream.destroyed) {
      upstreamStream.destroy();
    }
  }

  res.on("close", () => {
    // Cancel upstream only if the client disconnected prematurely.
    if (!res.writableEnded && !streamEnded) {
      log("Client disconnected during streaming.");
      destroyUpstream();
    }
  });

  upstreamStream.on("data", (chunk) => {
    if (streamEnded || res.destroyed) {
      return;
    }

    sseBuffer += decoder.write(chunk);
    processBuffer();
  });

  upstreamStream.on("end", () => {
    if (streamEnded) {
      return;
    }

    sseBuffer += decoder.end();
    processBuffer(true);

    if (!streamEnded) {
      finishStream();
    }

    log("NVIDIA stream ended.");
  });

  upstreamStream.on("error", (error) => {
    if (streamEnded || res.destroyed) {
      return;
    }

    log("NVIDIA streaming error.", error.message);

    // Once SSE headers are sent, a JSON HTTP error is no longer possible.
    // Send an SSE error event and close the connection.
    res.write(
      `event: error\ndata: ${JSON.stringify({
        error: {
          message: "NVIDIA streaming connection failed.",
          type: "upstream_error",
        },
      })}\n\n`
    );

    streamEnded = true;
    res.end();
  });
});

// Unknown routes.
app.use((req, res) => {
  res.status(404).json({
    error: {
      message: `Route not found: ${req.method} ${req.path}`,
      type: "not_found",
    },
  });
});

// Final error handler.
app.use((error, req, res, next) => {
  log("Unhandled server error.", error.message);

  if (res.headersSent) {
    return res.end();
  }

  res.status(500).json({
    error: {
      message: "Internal proxy error.",
      type: "internal_error",
    },
  });
});

app.listen(PORT, "0.0.0.0", () => {
  log(`Server listening on port ${PORT}.`);
  log(`Default model: ${DEFAULT_MODEL}`);
  log(`Thinking enabled: ${ENABLE_THINKING}`);
  log(`NVIDIA API key configured: ${Boolean(NVIDIA_API_KEY)}`);
});
