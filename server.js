// server.js
// OpenAI-compatible proxy for NVIDIA NIM
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
  process.env.NVIDIA_MODEL ||
  "nvidia/nemotron-3.5-lightning-30b-a3b";

const DEFAULT_TEMPERATURE = 1.0;
const DEFAULT_MAX_TOKENS = 16384;

const REASONING_EFFORT =
  process.env.REASONING_EFFORT || "high";

const SHOW_REASONING =
  String(process.env.SHOW_REASONING || "false").toLowerCase() === "true";

const ENABLE_THINKING =
  String(process.env.ENABLE_THINKING || "true").toLowerCase() === "true";

const MAX_RESPONSE_CHARS = 19000;

// ============================================================
// ROLEPLAY INSTRUCTION
// ============================================================

const ROLEPLAY_INSTRUCTION = `
ROLEPLAY CONTROL RULES:

You control only the characters, entities, and world elements that belong to you.

The user controls their own character completely.

NEVER speak for the user's character.
NEVER write dialogue for the user's character.
NEVER decide what the user's character says.
NEVER decide what the user's character does.
NEVER describe actions performed by the user's character.
NEVER describe thoughts, feelings, emotions, intentions, reactions, decisions, or sensations belonging to the user's character.
NEVER assume how the user's character reacts to something.
NEVER move, position, or control the user's character.
NEVER finish the user's character's action or sentence for them.

If the user's character needs to respond, stop and leave that response entirely to the user.

You may describe the environment and the actions, dialogue, thoughts, feelings, and reactions of characters that you control.

Do not narrate the user's character as if you are the user.

Always preserve the user's agency and wait for the user to decide their character's next action, dialogue, thoughts, feelings, or reaction.
`;

// ============================================================
// MIDDLEWARE
// ============================================================

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

// ============================================================
// REQUEST LOGGER
// ============================================================

app.use((req, res, next) => {
  console.log(
    `[${new Date().toISOString()}] ${req.method} ${req.path}`
  );

  next();
});

// ============================================================
// MODEL ALIASES
// ============================================================

const MODEL_ALIASES = {
  "gpt-3.5-turbo": DEFAULT_MODEL,
  "gpt-4": DEFAULT_MODEL,
  "gpt-4o": DEFAULT_MODEL,
  "gpt-4.1": DEFAULT_MODEL,

  "claude-3-opus": DEFAULT_MODEL,
  "claude-3-sonnet": DEFAULT_MODEL,
  "claude-3-haiku": DEFAULT_MODEL,

  "claude-3.5-sonnet": DEFAULT_MODEL,
  "claude-3.7-sonnet": DEFAULT_MODEL,

  "gemini-pro": DEFAULT_MODEL,
  "gemini-1.5-pro": DEFAULT_MODEL,
  "gemini-2.0-flash": DEFAULT_MODEL,

  "nemotron": DEFAULT_MODEL,
  "nemotron-3.5": DEFAULT_MODEL
};

// ============================================================
// MODEL RESOLUTION
// ============================================================

function resolveModel(requestedModel) {
  if (!requestedModel) {
    return DEFAULT_MODEL;
  }

  if (MODEL_ALIASES[requestedModel]) {
    return MODEL_ALIASES[requestedModel];
  }

  if (requestedModel.startsWith("nvidia/")) {
    return requestedModel;
  }

  if (requestedModel.includes("/")) {
    return requestedModel;
  }

  return DEFAULT_MODEL;
}

// ============================================================
// HEALTH
// ============================================================

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

// ============================================================
// ROOT
// ============================================================

app.get("/", (req, res) => {
  res.json({
    status: "ok",
    service: "OpenAI to NVIDIA NIM Proxy",
    model: DEFAULT_MODEL
  });
});

// ============================================================
// MODELS
// ============================================================

app.get("/v1/models", (req, res) => {
  const aliases = Object.keys(MODEL_ALIASES);

  const data = aliases.map((id) => ({
    id,
    object: "model",
    created: Math.floor(Date.now() / 1000),
    owned_by: "nvidia-nim"
  }));

  data.push({
    id: DEFAULT_MODEL,
    object: "model",
    created: Math.floor(Date.now() / 1000),
    owned_by: "nvidia"
  });

  res.json({
    object: "list",
    data
  });
});

// ============================================================
// DEBUG NVIDIA
// ============================================================

app.get("/debug-nvidia", async (req, res) => {
  if (!NVIDIA_API_KEY) {
    return res.status(500).json({
      error: "NVIDIA_API_KEY is not configured"
    });
  }

  const model = DEFAULT_MODEL;

  const results = {};

  async function runTest(name, body) {
    const start = Date.now();

    try {
      const response = await axios.post(
        `${NVIDIA_API_BASE}/chat/completions`,
        body,
        {
          headers: {
            Authorization: `Bearer ${NVIDIA_API_KEY}`,
            "Content-Type": "application/json"
          },
          timeout: 300000
        }
      );

      const elapsed = Date.now() - start;

      const text =
        response.data?.choices?.[0]?.message?.content || "";

      results[name] = {
        status: response.status,
        time_ms: elapsed,
        success: response.status === 200,
        text_received: Boolean(text)
      };
    } catch (error) {
      const elapsed = Date.now() - start;

      results[name] = {
        status: error.response?.status || 0,
        time_ms: elapsed,
        success: false,
        error:
          error.response?.data ||
          error.message ||
          "Unknown error"
      };
    }
  }

  await runTest("basic", {
    model,
    messages: [
      {
        role: "user",
        content: "Say hello."
      }
    ],
    temperature: 0.7,
    max_tokens: 50,
    stream: false
  });

  await runTest("thinking", {
    model,
    messages: [
      {
        role: "user",
        content: "Say hello."
      }
    ],
    temperature: 1.0,
    max_tokens: 50,
    stream: false,
    chat_template_kwargs: {
      enable_thinking: true
    }
  });

  await runTest("reasoning_high", {
    model,
    messages: [
      {
        role: "user",
        content: "Say hello."
      }
    ],
    temperature: 1.0,
    max_tokens: 50,
    stream: false,
    reasoning_effort: "high",
    chat_template_kwargs: {
      enable_thinking: true
    }
  });

  res.json({
    model,
    api_base: NVIDIA_API_BASE,
    tests: results
  });
});

// ============================================================
// CHAT COMPLETIONS
// ============================================================

app.post("/v1/chat/completions", async (req, res) => {
  const requestStart = Date.now();

  try {
    if (!NVIDIA_API_KEY) {
      return res.status(500).json({
        error: {
          message: "NVIDIA_API_KEY is not configured",
          type: "configuration_error"
        }
      });
    }

    const requestedModel = req.body?.model;

    const model = resolveModel(requestedModel);

    let messages = Array.isArray(req.body?.messages)
      ? [...req.body.messages]
      : [];

    if (messages.length === 0) {
      return res.status(400).json({
        error: {
          message: "messages is required",
          type: "invalid_request_error"
        }
      });
    }

    console.log(
      `Model requested: ${requestedModel || "none"} -> NVIDIA: ${model}`
    );

    // ========================================================
    // ADD ROLEPLAY INSTRUCTION
    // ========================================================

    const systemIndex = messages.findIndex(
      (message) => message?.role === "system"
    );

    if (systemIndex >= 0) {
      messages[systemIndex] = {
        ...messages[systemIndex],
        content:
          ROLEPLAY_INSTRUCTION +
          "\n\n" +
          String(messages[systemIndex].content || "")
      };
    } else {
      messages.unshift({
        role: "system",
        content: ROLEPLAY_INSTRUCTION
      });
    }

    const temperature =
      typeof req.body?.temperature === "number"
        ? req.body.temperature
        : DEFAULT_TEMPERATURE;

    const maxTokens =
      typeof req.body?.max_tokens === "number"
        ? req.body.max_tokens
        : DEFAULT_MAX_TOKENS;

    const stream = Boolean(req.body?.stream);

    const nimRequest = {
      model,
      messages,
      temperature,
      max_tokens: maxTokens,
      stream,
      reasoning_effort: REASONING_EFFORT,
      chat_template_kwargs: {
        enable_thinking: ENABLE_THINKING
      }
    };

    console.log("Sending request to NVIDIA now...");
    console.log("Model:", model);
    console.log("Temperature:", temperature);
    console.log("Max tokens:", maxTokens);
    console.log("Stream:", stream);
    console.log("Reasoning effort:", REASONING_EFFORT);
    console.log("Thinking enabled:", ENABLE_THINKING);

    const nvidiaStart = Date.now();

    // ========================================================
    // STREAMING
    // ========================================================

    if (stream) {
      const response = await axios.post(
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

      console.log(
        "NVIDIA status:",
        response.status
      );

      console.log(
        "NVIDIA HTTP response received."
      );

      res.status(200);

      res.setHeader(
        "Content-Type",
        "text/event-stream"
      );

      res.setHeader(
        "Cache-Control",
        "no-cache"
      );

      res.setHeader(
        "Connection",
        "keep-alive"
      );

      res.flushHeaders();

      let firstTokenReceived = false;
      let outputChars = 0;
      let finished = false;

      function sendChunk(content, finishReason = null) {
        if (finished) {
          return;
        }

        const chunk = {
          id: `chatcmpl-${Date.now()}`,
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model,
          choices: [
            {
              index: 0,
              delta: content
                ? {
                    content
                  }
                : {},
              finish_reason: finishReason
            }
          ]
        };

        res.write(
          `data: ${JSON.stringify(chunk)}\n\n`
        );
      }

      function sendFinish() {
        if (finished) {
          return;
        }

        const chunk = {
          id: `chatcmpl-${Date.now()}`,
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
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
          `data: ${JSON.stringify(chunk)}\n\n`
        );

        res.write("data: [DONE]\n\n");

        finished = true;
      }

      response.data.on("data", (chunk) => {
        if (finished) {
          return;
        }

        const text = chunk.toString("utf8");

        const lines = text.split(/\r?\n/);

        for (const line of lines) {
          if (!line.startsWith("data:")) {
            continue;
          }

          const data = line.slice(5).trim();

          if (!data || data === "[DONE]") {
            continue;
          }

          let parsed;

          try {
            parsed = JSON.parse(data);
          } catch {
            continue;
          }

          const choice = parsed?.choices?.[0];

          if (!choice) {
            continue;
          }

          const delta = choice.delta || {};

          let content = "";

          if (typeof delta.content === "string") {
            content = delta.content;
          }

          if (!content) {
            continue;
          }

          if (!firstTokenReceived) {
            firstTokenReceived = true;

            console.log(
              `FIRST TOKEN RECEIVED after ${
                Date.now() - nvidiaStart
              } ms`
            );
          }

          const remaining =
            MAX_RESPONSE_CHARS - outputChars;

          if (remaining <= 0) {
            sendFinish();

            if (!res.writableEnded) {
              res.end();
            }

            return;
          }

          if (content.length > remaining) {
            content = content.slice(0, remaining);
          }

          outputChars += content.length;

          sendChunk(content);

          if (
            outputChars >= MAX_RESPONSE_CHARS
          ) {
            sendFinish();

            if (!res.writableEnded) {
              res.end();
            }

            return;
          }
        }
      });

      response.data.on("end", () => {
        console.log("NVIDIA stream ended.");

        sendFinish();

        if (!res.writableEnded) {
          res.end();
        }

        console.log(
          `TIME TOTAL: ${
            Date.now() - requestStart
          } ms`
        );
      });

      response.data.on("error", (error) => {
        console.error(
          "NVIDIA streaming error:",
          error.message
        );

        if (!res.writableEnded) {
          res.end();
        }
      });

      req.on("close", () => {
        if (!response.data.destroyed) {
          response.data.destroy();
        }
      });

      return;
    }

    // ========================================================
    // NON-STREAMING
    // ========================================================

    const response = await axios.post(
      `${NVIDIA_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          Authorization: `Bearer ${NVIDIA_API_KEY}`,
          "Content-Type": "application/json"
        },
        timeout: 300000
      }
    );

    console.log(
      "NVIDIA status:",
      response.status
    );

    const data = response.data;

    const choice = data?.choices?.[0];

    if (
      choice?.message &&
      typeof choice.message.content === "string"
    ) {
      if (
        choice.message.content.length >
        MAX_RESPONSE_CHARS
      ) {
        choice.message.content =
          choice.message.content.slice(
            0,
            MAX_RESPONSE_CHARS
          );
      }
    }

    console.log(
      `TIME NVIDIA: ${
        Date.now() - nvidiaStart
      } ms`
    );

    console.log(
      `TIME TOTAL: ${
        Date.now() - requestStart
      } ms`
    );

    return res.status(200).json(data);
  } catch (error) {
    console.error(
      "Proxy error:",
      error.response?.data ||
        error.message ||
        error
    );

    if (res.headersSent) {
      if (!res.writableEnded) {
        res.end();
      }

      return;
    }

    const status =
      error.response?.status || 500;

    return res.status(status).json({
      error: {
        message:
          error.response?.data?.error?.message ||
          error.response?.data ||
          error.message ||
          "Upstream NVIDIA error",
        type: "upstream_error",
        status
      }
    });
  }
});

// ============================================================
// 404
// ============================================================

app.use((req, res) => {
  res.status(404).json({
    error: {
      message: "Route not found",
      type: "not_found",
      path: req.path
    }
  });
});

// ============================================================
// START SERVER
// ============================================================

app.listen(PORT, () => {
  console.log(
    `Proxy listening on port ${PORT}`
  );

  console.log(
    `NVIDIA API: ${NVIDIA_API_BASE}`
  );

  console.log(
    `Default model: ${DEFAULT_MODEL}`
  );

  console.log(
    `Reasoning effort: ${REASONING_EFFORT}`
  );

  console.log(
    `Thinking enabled: ${ENABLE_THINKING}`
  );

  console.log(
    `Max response chars: ${MAX_RESPONSE_CHARS}`
  );
});
