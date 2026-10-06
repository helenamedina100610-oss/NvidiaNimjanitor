// server.js - OpenAI-compatible proxy for NVIDIA NIM
// Designed for Render + Janitor AI
// Real NVIDIA streaming -> Janitor SSE

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
    console.log("Reasoning effort:", REASONING_EFFORT);
    console.log("Max response characters:", MAX_RESPONSE_CHARS);

    // ------------------------------------------------
    // Build NVIDIA request
    // ------------------------------------------------

    const nimRequest = {
      model,
      messages: body.messages,
      temperature,
      max_tokens: maxTokens,
      reasoning_effort: REASONING_EFFORT,
      stream: janitorRequestedStream
    };

    if (body.top_p !== undefined) {
      nimRequest.top_p = body.top_p;
    }

    if (body.stop !== undefined) {
      nimRequest.stop = body.stop;
    }

    if (body.tools !== undefined) {
      nimRequest.tools = body.tools;
    }

    if (body.tool_choice !== undefined) {
      nimRequest.tool_choice = body.tool_choice;
    }

    // ------------------------------------------------
    // REAL STREAMING
    // ------------------------------------------------

    if (janitorRequestedStream) {
      console.log("Real NVIDIA streaming enabled.");

      let upstream;

      try {
        upstream = await axios.post(
          `${NVIDIA_API_BASE}/chat/completions`,
          nimRequest,
          {
            headers: {
              Authorization: `Bearer ${NVIDIA_API_KEY}`,
              "Content-Type": "application/json",
              Accept: "text/event-stream"
            },
            responseType: "stream",
            timeout: 300000
          }
        );
      } catch (error) {
        console.error(
          "NVIDIA streaming connection failed:",
          error.message
        );

        if (error.response) {
          console.error(
            "NVIDIA status:",
            error.response.status
          );
        }

        if (!res.headersSent) {
          return res.status(
            error.response?.status || 502
          ).json({
            error: {
              message:
                error.response?.data ||
                error.message ||
                "NVIDIA streaming error",
              type: "upstream_error"
            }
          });
        }

        return;
      }

      console.log(
        "NVIDIA streaming connection opened."
      );

      console.log(
        "NVIDIA status:",
        upstream.status
      );

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

      if (typeof res.flushHeaders === "function") {
        res.flushHeaders();
      }

      let buffer = "";
      let totalCharacters = 0;
      let responseStarted = false;
      let stopped = false;

      const streamId =
        `chatcmpl-${Date.now()}`;

      const created =
        Math.floor(Date.now() / 1000);

      function sendChunk(chunk) {
        if (stopped) {
          return;
        }

        res.write(
          `data: ${JSON.stringify(chunk)}\n\n`
        );

        if (typeof res.flush === "function") {
          res.flush();
        }
      }

      function sendRoleChunk() {
        if (responseStarted) {
          return;
        }

        responseStarted = true;

        sendChunk({
          id: streamId,
          object: "chat.completion.chunk",
          created,
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
        });
      }

      function processSSELine(line) {
        if (stopped) {
          return;
        }

        line = line.trim();

        if (!line) {
          return;
        }

        if (!line.startsWith("data:")) {
          return;
        }

        const dataText =
          line.slice(5).trim();

        if (!dataText) {
          return;
        }

        if (dataText === "[DONE]") {
          stopped = true;

          sendChunk({
            id: streamId,
            object: "chat.completion.chunk",
            created,
            model,
            choices: [
              {
                index: 0,
                delta: {},
                finish_reason: "stop"
              }
            ]
          });

          res.write("data: [DONE]\n\n");
          res.end();

          const totalTime =
            Date.now() - requestStart;

          console.log(
            "=========================================="
          );

          console.log(
            "STREAM COMPLETE"
          );

          console.log(
            "Characters sent:",
            totalCharacters
          );

          console.log(
            "TIME TOTAL:",
            totalTime,
            "ms",
            `(${(totalTime / 1000).toFixed(2)} seconds)`
          );

          console.log(
            "=========================================="
          );

          return;
        }

        let parsed;

        try {
          parsed = JSON.parse(dataText);
        } catch (error) {
          console.log(
            "Skipping invalid SSE JSON chunk."
          );
          return;
        }

        const choice =
          parsed?.choices?.[0];

        if (!choice) {
          return;
        }

        const delta =
          choice.delta || {};

        sendRoleChunk();

        // --------------------------------------------
        // Reasoning
        //
        // SHOW_REASONING remains false by default.
        // Therefore reasoning_content is NOT sent to
        // Janitor as visible text.
        // --------------------------------------------

        if (
          SHOW_REASONING &&
          typeof delta.reasoning_content === "string"
        ) {
          const reasoningText =
            delta.reasoning_content;

          sendChunk({
            id:
              parsed.id || streamId,
            object: "chat.completion.chunk",
            created:
              parsed.created || created,
            model:
              parsed.model || model,
            choices: [
              {
                index: 0,
                delta: {
                  content: reasoningText
                },
                finish_reason: null
              }
            ]
          });
        }

        // --------------------------------------------
        // Visible content
        // --------------------------------------------

        if (
          typeof delta.content === "string" &&
          delta.content.length > 0
        ) {
          const remaining =
            MAX_RESPONSE_CHARS -
            totalCharacters;

          if (remaining <= 0) {
            console.log(
              "19,000 character limit reached."
            );
            return;
          }

          const contentText =
            delta.content.slice(
              0,
              remaining
            );

          if (contentText.length > 0) {
            totalCharacters +=
              contentText.length;

            sendChunk({
              id:
                parsed.id || streamId,
              object: "chat.completion.chunk",
              created:
                parsed.created || created,
              model:
                parsed.model || model,
              choices: [
                {
                  index: 0,
                  delta: {
                    content: contentText
                  },
                  finish_reason: null
                }
              ]
            });
          }
        }
      }

      upstream.data.on(
        "data",
        (chunk) => {
          if (stopped) {
            return;
          }

          buffer += chunk.toString("utf8");

          const lines =
            buffer.split(/\r?\n/);

          buffer =
            lines.pop() || "";

          for (const line of lines) {
            processSSELine(line);
          }
        }
      );

      upstream.data.on(
        "end",
        () => {
          if (stopped) {
            return;
          }

          if (buffer.trim()) {
            processSSELine(buffer);
          }

          if (!stopped) {
            stopped = true;

            sendChunk({
              id: streamId,
              object: "chat.completion.chunk",
              created,
              model,
              choices: [
                {
                  index: 0,
                  delta: {},
                  finish_reason: "stop"
                }
              ]
            });

            res.write("data: [DONE]\n\n");
            res.end();
          }

          const totalTime =
            Date.now() - requestStart;

          console.log(
            "Streaming connection ended."
          );

          console.log(
            "Characters sent:",
            totalCharacters
          );

          console.log(
            "TIME TOTAL:",
            totalTime,
            "ms",
            `(${(totalTime / 1000).toFixed(2)} seconds)`
          );
        }
      );

      upstream.data.on(
        "error",
        (error) => {
          console.error(
            "NVIDIA streaming error:",
            error.message
          );

          if (!res.writableEnded) {
            try {
              res.end();
            } catch (_) {}
          }
        }
      );

      req.on(
        "close",
        () => {
          if (!stopped) {
            console.log(
              "Janitor closed the connection."
            );

            try {
              upstream.data.destroy();
            } catch (_) {}
          }
        }
      );

      return;
    }

    // ------------------------------------------------
    // NON-STREAMING REQUEST
    // ------------------------------------------------

    console.log(
      "Non-streaming NVIDIA request."
    );

    nimRequest.stream = false;

    const nvidiaStart = Date.now();

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
      console.error(
        "NVIDIA request failed:",
        error.message
      );

      if (error.response) {
        console.error(
          "NVIDIA status:",
          error.response.status
        );

        console.error(
          "NVIDIA error:",
          JSON.stringify(error.response.data)
        );
      }

      return res.status(
        error.response?.status || 502
      ).json({
        error: {
          message:
            error.message ||
            "NVIDIA request failed",
          type: "upstream_error"
        }
      });
    }

    const nvidiaTime =
      Date.now() - nvidiaStart;

    console.log(
      "TIME NVIDIA:",
      nvidiaTime,
      "ms",
      `(${(nvidiaTime / 1000).toFixed(2)} seconds)`
    );

    const data =
      nvidiaResponse.data;

    const message =
      data?.choices?.[0]?.message || {};

    let content =
      typeof message.content === "string"
        ? message.content
        : "";

    const characters =
      Array.from(content);

    if (characters.length > MAX_RESPONSE_CHARS) {
      content = characters
        .slice(0, MAX_RESPONSE_CHARS)
        .join("");
    }

    const responseMessage = {
      role: "assistant",
      content
    };

    if (
      SHOW_REASONING &&
      message.reasoning_content
    ) {
      responseMessage.reasoning_content =
        message.reasoning_content;
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
          finish_reason:
            data?.choices?.[0]?.finish_reason ||
            "stop"
        }
      ]
    };

    if (data.usage) {
      responseData.usage =
        data.usage;
    }

    const totalTime =
      Date.now() - requestStart;

    console.log(
      "TIME TOTAL:",
      totalTime,
      "ms",
      `(${(totalTime / 1000).toFixed(2)} seconds)`
    );

    return res.status(200).json(
      responseData
    );
  } catch (error) {
    console.error(
      "Proxy error:",
      error
    );

    if (!res.headersSent) {
      return res.status(500).json({
        error: {
          message:
            error.message ||
            "Internal proxy error",
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
  console.log(
    "OpenAI to NVIDIA NIM Proxy"
  );
  console.log(
    "Server listening on port",
    PORT
  );
  console.log(
    "NVIDIA API configured:",
    Boolean(NVIDIA_API_KEY)
  );
  console.log(
    "Default model:",
    DEFAULT_MODEL
  );
  console.log(
    "Reasoning effort:",
    REASONING_EFFORT
  );
  console.log(
    "Max tokens default:",
    16384
  );
  console.log(
    "Temperature default:",
    1.0
  );
  console.log(
    "Max response characters:",
    MAX_RESPONSE_CHARS
  );
  console.log(
    "Real streaming: ENABLED"
  );
  console.log("==========================================");
});
