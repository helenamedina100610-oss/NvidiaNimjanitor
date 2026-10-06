// server.js - OpenAI-compatible proxy for NVIDIA NIM
// Designed for Render + Janitor AI

const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();

const PORT = process.env.PORT || 3000;

const NVIDIA_API_BASE = (
  process.env.NVIDIA_API_BASE ||
  "https://integrate.api.nvidia.com/v1"
).replace(/\/+$/, "");

const NVIDIA_API_KEY = process.env.NVIDIA_API_KEY || "";

const DEFAULT_MODEL =
  process.env.NVIDIA_MODEL || "moonshotai/kimi-k3";

const SHOW_REASONING =
  String(process.env.SHOW_REASONING || "false").toLowerCase() === "true";

const REASONING_EFFORT =
  process.env.REASONING_EFFORT || "max";

const MAX_RESPONSE_CHARS = 19000;

// --------------------------------------------------
// Middleware
// --------------------------------------------------

app.use(cors());

app.use(
  express.json({
    limit: "5mb"
  })
);

app.use(
  express.urlencoded({
    extended: true,
    limit: "5mb"
  })
);

// --------------------------------------------------
// Request logger
// --------------------------------------------------

app.use((req, res, next) => {
  console.log(
    `${new Date().toISOString()} ${req.method} ${req.path}`
  );
  next();
});

// --------------------------------------------------
// Model mapping
// --------------------------------------------------

const MODEL_MAPPING = {
  "gpt-3.5-turbo": DEFAULT_MODEL,
  "gpt-3.5-turbo-16k": DEFAULT_MODEL,
  "gpt-4": DEFAULT_MODEL,
  "gpt-4-turbo": DEFAULT_MODEL,
  "gpt-4-turbo-preview": DEFAULT_MODEL,
  "gpt-4o": DEFAULT_MODEL,
  "gpt-4o-mini": DEFAULT_MODEL,
  "gpt-4.1": DEFAULT_MODEL,
  "gpt-4.1-mini": DEFAULT_MODEL,

  "claude-3-opus": DEFAULT_MODEL,
  "claude-3-sonnet": DEFAULT_MODEL,
  "claude-3.5-sonnet": DEFAULT_MODEL,
  "claude-3.7-sonnet": DEFAULT_MODEL,

  "gemini-pro": DEFAULT_MODEL,

  "nemotron-3-ultra": DEFAULT_MODEL,

  [DEFAULT_MODEL]: DEFAULT_MODEL
};

function resolveModel(requestedModel) {
  if (!requestedModel) {
    return DEFAULT_MODEL;
  }

  if (requestedModel.startsWith("nvidia/")) {
    return requestedModel;
  }

  return MODEL_MAPPING[requestedModel] || DEFAULT_MODEL;
}

// --------------------------------------------------
// Health
// --------------------------------------------------

app.get("/", (req, res) => {
  res.json({
    status: "ok",
    service: "OpenAI to NVIDIA NIM Proxy",
    model: DEFAULT_MODEL
  });
});

app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    service: "OpenAI to NVIDIA NIM Proxy",
    model: DEFAULT_MODEL,
    reasoning_display: SHOW_REASONING,
    reasoning_effort: REASONING_EFFORT,
    nim_api_configured: Boolean(NVIDIA_API_KEY)
  });
});

app.get("/health/", (req, res) => {
  res.json({
    status: "ok",
    service: "OpenAI to NVIDIA NIM Proxy",
    model: DEFAULT_MODEL,
    reasoning_display: SHOW_REASONING,
    reasoning_effort: REASONING_EFFORT,
    nim_api_configured: Boolean(NVIDIA_API_KEY)
  });
});

// --------------------------------------------------
// Models
// --------------------------------------------------

app.get("/v1/models", (req, res) => {
  const aliases = Object.keys(MODEL_MAPPING);

  const models = aliases.map((id) => ({
    id,
    object: "model",
    owned_by: "nvidia-nim"
  }));

  if (!models.some((m) => m.id === DEFAULT_MODEL)) {
    models.push({
      id: DEFAULT_MODEL,
      object: "model",
      owned_by: "nvidia-nim"
    });
  }

  res.json({
    object: "list",
    data: models
  });
});

// --------------------------------------------------
// Chat completions
// --------------------------------------------------

app.post("/v1/chat/completions", async (req, res) => {
  const requestStart = Date.now();

  console.log("==========================================");
  console.log("NVIDIA REQUEST START");

  try {
    if (!NVIDIA_API_KEY) {
      return res.status(500).json({
        error: {
          message: "NVIDIA_API_KEY is not configured.",
          type: "configuration_error"
        }
      });
    }

    const body = req.body || {};

    if (!Array.isArray(body.messages) || body.messages.length === 0) {
      return res.status(400).json({
        error: {
          message: "messages is required and must be a non-empty array.",
          type: "invalid_request_error"
        }
      });
    }

    const requestedModel = body.model || "gpt-4o";
    const model = resolveModel(requestedModel);

    const temperature =
      body.temperature !== undefined
        ? body.temperature
        : 1.0;

    const maxTokens =
      body.max_tokens ??
      body.max_completion_tokens ??
      16384;

    const janitorRequestedStream = Boolean(body.stream);

    console.log(
      "Model requested:",
      requestedModel,
      "-> NVIDIA:",
      model
    );

    console.log("Temperature:", temperature);
    console.log("Max tokens:", maxTokens);
    console.log("Stream requested:", janitorRequestedStream);

    // ------------------------------------------------
    // IMPORTANT:
    // NVIDIA receives stream:false because this is the
    // stable configuration that works with Janitor.
    // We generate the complete response first and then
    // send it to Janitor as compatible SSE chunks.
    // ------------------------------------------------

    const nimRequest = {
      model,
      messages: body.messages,
      temperature,
      max_tokens: maxTokens,
      stream: false
    };

    if (body.top_p !== undefined) {
      nimRequest.top_p = body.top_p;
    }

    if (body.presence_penalty !== undefined) {
      nimRequest.presence_penalty = body.presence_penalty;
    }

    if (body.frequency_penalty !== undefined) {
      nimRequest.frequency_penalty = body.frequency_penalty;
    }

    if (body.stop !== undefined) {
      nimRequest.stop = body.stop;
    }

    console.log(
      "Sending request to NVIDIA:",
      `${NVIDIA_API_BASE}/chat/completions`
    );

    const nvidiaStart = Date.now();

    console.log("NVIDIA timer started.");

    let nvidiaResponse;

    try {
      nvidiaResponse = await axios.post(
        `${NVIDIA_API_BASE}/chat/completions`,
        nimRequest,
        {
          headers: {
            Authorization: `Bearer ${NVIDIA_API_KEY}`,
            "Content-Type": "application/json",
            Accept: "application/json"
          },
          timeout: 300000
        }
      );
    } catch (error) {
      const nvidiaTime = Date.now() - nvidiaStart;

      console.error(
        "NVIDIA request failed after:",
        nvidiaTime,
        "ms"
      );

      if (error.response) {
        console.error(
          "NVIDIA error status:",
          error.response.status
        );

        console.error(
          "NVIDIA error data:",
          JSON.stringify(error.response.data)
        );

        return res.status(error.response.status).json({
          error: {
            message:
              typeof error.response.data === "string"
                ? error.response.data
                : JSON.stringify(error.response.data),
            type: "upstream_error"
          }
        });
      }

      console.error(
        "NVIDIA connection error:",
        error.message
      );

      return res.status(502).json({
        error: {
          message: error.message,
          type: "upstream_error"
        }
      });
    }

    const nvidiaTime = Date.now() - nvidiaStart;

    console.log(
      "TIME NVIDIA:",
      nvidiaTime,
      "ms",
      `(${(nvidiaTime / 1000).toFixed(2)} seconds)`
    );

    console.log(
      "NVIDIA status:",
      nvidiaResponse.status
    );

    console.log("NVIDIA response received.");

    const data = nvidiaResponse.data;

    let message = data?.choices?.[0]?.message || {};

    let content = message.content || "";

    if (typeof content !== "string") {
      content = String(content);
    }

    // ------------------------------------------------
    // Limit response to 19,000 characters.
    // Array.from() avoids cutting Unicode characters
    // incorrectly in the middle of a surrogate pair.
    // ------------------------------------------------

    const originalLength = Array.from(content).length;

    if (originalLength > MAX_RESPONSE_CHARS) {
      content = Array.from(content)
        .slice(0, MAX_RESPONSE_CHARS)
        .join("");

      console.log(
        `Response limited from ${originalLength} to ${MAX_RESPONSE_CHARS} characters.`
      );
    }

    console.log(
      "NVIDIA content length:",
      Array.from(content).length
    );

    // ------------------------------------------------
    // Optional reasoning content
    // ------------------------------------------------

    const reasoningContent =
      message.reasoning_content ||
      message.reasoning ||
      "";

    // ------------------------------------------------
    // If Janitor requested streaming, send the complete
    // NVIDIA response as several small SSE chunks.
    //
    // This is NOT upstream streaming.
    // It is only a compatibility layer for Janitor.
    // ------------------------------------------------

    if (janitorRequestedStream) {
      res.status(200);

      res.setHeader(
        "Content-Type",
        "text/event-stream; charset=utf-8"
      );

      res.setHeader(
        "Cache-Control",
        "no-cache, no-transform"
      );

      res.setHeader(
        "Connection",
        "keep-alive"
      );

      res.setHeader(
        "X-Accel-Buffering",
        "no"
      );

      // First chunk: role
      const roleChunk = {
        id:
          data.id ||
          `chatcmpl-${Date.now()}`,
        object: "chat.completion.chunk",
        created:
          data.created ||
          Math.floor(Date.now() / 1000),
        model,
        choices: [
          {
            index: 0,
            delta: {
              role: "assistant"
            },
            finish_reason: null
          }
        ]
      };

      res.write(
        `data: ${JSON.stringify(roleChunk)}\n\n`
      );

      // ------------------------------------------------
      // Send content in smaller chunks.
      // This avoids putting the entire response into
      // one giant SSE event.
      // ------------------------------------------------

      const characters = Array.from(content);

      const CHUNK_SIZE = 500;

      for (
        let i = 0;
        i < characters.length;
        i += CHUNK_SIZE
      ) {
        const chunkText = characters
          .slice(i, i + CHUNK_SIZE)
          .join("");

        const contentChunk = {
          id:
            data.id ||
            `chatcmpl-${Date.now()}`,
          object: "chat.completion.chunk",
          created:
            data.created ||
            Math.floor(Date.now() / 1000),
          model,
          choices: [
            {
              index: 0,
              delta: {
                content: chunkText
              },
              finish_reason: null
            }
          ]
        };

        res.write(
          `data: ${JSON.stringify(contentChunk)}\n\n`
        );
      }

      // ------------------------------------------------
      // Final chunk
      // ------------------------------------------------

      const finishChunk = {
        id:
          data.id ||
          `chatcmpl-${Date.now()}`,
        object: "chat.completion.chunk",
        created:
          data.created ||
          Math.floor(Date.now() / 1000),
        model,
        choices: [
          {
            index: 0,
            delta: {},
            finish_reason: "stop"
          }
        ]
      };

      res.write(
        `data: ${JSON.stringify(finishChunk)}\n\n`
      );

      res.write("data: [DONE]\n\n");

      res.end();

      const totalTime = Date.now() - requestStart;

      console.log(
        "TIME FINAL:",
        totalTime,
        "ms",
        `(${(totalTime / 1000).toFixed(2)} seconds)`
      );

      console.log(
        "Response ready for Janitor."
      );

      console.log(
        "TIME TOTAL:",
        totalTime,
        "ms",
        `(${(totalTime / 1000).toFixed(2)} seconds)`
      );

      console.log("==========================================");

      return;
    }

    // ------------------------------------------------
    // Normal non-streaming response
    // ------------------------------------------------

    const responseMessage = {
      role: "assistant",
      content
    };

    if (SHOW_REASONING && reasoningContent) {
      responseMessage.reasoning_content =
        reasoningContent;
    }

    const responseData = {
      id:
        data.id ||
        `chatcmpl-${Date.now()}`,
      object: "chat.completion",
      created:
        data.created ||
        Math.floor(Date.now() / 1000),
      model,
      choices: [
        {
          index: 0,
          message: responseMessage,
          finish_reason: "stop"
        }
      ]
    };

    if (data.usage) {
      responseData.usage = data.usage;
    }

    const totalTime = Date.now() - requestStart;

    console.log(
      "TIME FINAL:",
      totalTime,
      "ms",
      `(${(totalTime / 1000).toFixed(2)} seconds)`
    );

    console.log(
      "Response ready for Janitor."
    );

    console.log(
      "TIME TOTAL:",
      totalTime,
      "ms",
      `(${(totalTime / 1000).toFixed(2)} seconds)`
    );

    console.log("==========================================");

    return res.status(200).json(responseData);
  } catch (error) {
    const totalTime = Date.now() - requestStart;

    console.error(
      "Proxy error after:",
      totalTime,
      "ms"
    );

    console.error(error);

    if (!res.headersSent) {
      return res.status(500).json({
        error: {
          message: error.message || "Internal proxy error",
          type: "proxy_error"
        }
      });
    }

    try {
      res.end();
    } catch (_) {}

    return;
  }
});

// --------------------------------------------------
// Start server
// --------------------------------------------------

app.listen(PORT, () => {
  console.log("==========================================");
  console.log("OpenAI to NVIDIA NIM Proxy");
  console.log("Server listening on port", PORT);
  console.log(
    "NVIDIA API configured:",
    Boolean(NVIDIA_API_KEY)
  );
  console.log("Default model:", DEFAULT_MODEL);
  console.log("Reasoning effort:", REASONING_EFFORT);
  console.log("Max tokens default:", 16384);
  console.log("Temperature default:", 1.0);
  console.log(
    "Max response characters:",
    MAX_RESPONSE_CHARS
  );
  console.log("==========================================");
});
