// server.js
// OpenAI-compatible proxy for NVIDIA NIM
// Diagnostic version for Render + Janitor AI

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
  process.env.NVIDIA_MODEL || "nvidia/nemotron-3.5-lightning-30b-a3b";

const DEFAULT_TEMPERATURE = 1.0;
const DEFAULT_MAX_TOKENS = 16384;

const REASONING_EFFORT =
  process.env.REASONING_EFFORT || "high";

const SHOW_REASONING =
  String(process.env.SHOW_REASONING || "false").toLowerCase() === "true";

const ENABLE_THINKING =
  String(process.env.ENABLE_THINKING || "true").toLowerCase() === "true";

const MAX_RESPONSE_CHARS = 19000;

app.use(cors());

app.use(
  express.json({
    limit: "5mb"
  })
);

// ==================================================
// REQUEST LOG
// ==================================================

app.use((req, res, next) => {
  console.log(
    `[${new Date().toISOString()}] ${req.method} ${req.path}`
  );
  next();
});

// ==================================================
// HEALTH
// ==================================================

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

// ==================================================
// MODEL ALIASES
// ==================================================

const MODEL_ALIASES = {
  "gpt-3.5-turbo": DEFAULT_MODEL,
  "gpt-4": DEFAULT_MODEL,
  "gpt-4o": DEFAULT_MODEL,
  "gpt-4o-mini": DEFAULT_MODEL,
  "gpt-4.1": DEFAULT_MODEL,
  "gpt-4.1-mini": DEFAULT_MODEL,
  "claude-3": DEFAULT_MODEL,
  "claude-3.5-sonnet": DEFAULT_MODEL,
  "claude-3.7-sonnet": DEFAULT_MODEL,
  "gemini-pro": DEFAULT_MODEL
};

function resolveModel(requestedModel) {
  if (!requestedModel) {
    return DEFAULT_MODEL;
  }

  if (requestedModel.startsWith("nvidia/")) {
    return requestedModel;
  }

  if (requestedModel.includes("/")) {
    return requestedModel;
  }

  return MODEL_ALIASES[requestedModel] || DEFAULT_MODEL;
}

// ==================================================
// /v1/models
// ==================================================

app.get("/v1/models", (req, res) => {
  res.json({
    object: "list",
    data: [
      {
        id: "gpt-4o",
        object: "model",
        created: 0,
        owned_by: "nvidia-nim"
      },
      {
        id: DEFAULT_MODEL,
        object: "model",
        created: 0,
        owned_by: "nvidia-nim"
      }
    ]
  });
});

// ==================================================
// DEBUG - THREE KIMI TESTS
// ==================================================

app.get("/debug-nvidia", async (req, res) => {
  console.log("");
  console.log("==========================================");
  console.log("KIMI K3 DIAGNOSTIC TEST");
  console.log("==========================================");

  if (!NVIDIA_API_KEY) {
    return res.status(500).json({
      error: "NVIDIA_API_KEY is not configured"
    });
  }

  const headers = {
    Authorization: `Bearer ${NVIDIA_API_KEY}`,
    "Content-Type": "application/json"
  };

  const results = {
    model: DEFAULT_MODEL,
    api_base: NVIDIA_API_BASE,
    tests: {}
  };

  const baseRequest = {
    model: DEFAULT_MODEL,
    messages: [
      {
        role: "user",
        content: "Say hello in one short sentence."
      }
    ],
    temperature: 1.0,
    max_tokens: 50,
    stream: false
  };

  // ------------------------------------------------
  // TEST 1 - BASIC
  // ------------------------------------------------

  console.log("");
  console.log("TEST 1 - BASIC KIMI");
  console.log("Thinking: not specified");
  console.log("Reasoning: not specified");

  const test1Start = Date.now();

  try {
    const response = await axios.post(
      `${NVIDIA_API_BASE}/chat/completions`,
      baseRequest,
      {
        headers,
        timeout: 300000
      }
    );

    const elapsed = Date.now() - test1Start;

    console.log("TEST 1 STATUS:", response.status);
    console.log("TEST 1 TIME:", elapsed, "ms");

    results.tests.basic = {
      status: response.status,
      time_ms: elapsed,
      success: true,
      text_received: Boolean(
        response.data?.choices?.[0]?.message?.content
      )
    };
  } catch (error) {
    const elapsed = Date.now() - test1Start;

    console.log("TEST 1 ERROR");
    console.log("TEST 1 TIME:", elapsed, "ms");
    console.log(
      error.response?.data || error.message
    );

    results.tests.basic = {
      status: error.response?.status || null,
      time_ms: elapsed,
      success: false,
      error:
        error.response?.data ||
        error.message
    };
  }

  // ------------------------------------------------
  // TEST 2 - THINKING
  // ------------------------------------------------

  console.log("");
  console.log("TEST 2 - KIMI WITH THINKING");
  console.log("Thinking: true");
  console.log("Reasoning: not specified");

  const test2Start = Date.now();

  try {
    const response = await axios.post(
      `${NVIDIA_API_BASE}/chat/completions`,
      {
        ...baseRequest,
        chat_template_kwargs: {
          enable_thinking: true
        }
      },
      {
        headers,
        timeout: 300000
      }
    );

    const elapsed = Date.now() - test2Start;

    console.log("TEST 2 STATUS:", response.status);
    console.log("TEST 2 TIME:", elapsed, "ms");

    results.tests.thinking = {
      status: response.status,
      time_ms: elapsed,
      success: true,
      text_received: Boolean(
        response.data?.choices?.[0]?.message?.content
      )
    };
  } catch (error) {
    const elapsed = Date.now() - test2Start;

    console.log("TEST 2 ERROR");
    console.log("TEST 2 TIME:", elapsed, "ms");
    console.log(
      error.response?.data || error.message
    );

    results.tests.thinking = {
      status: error.response?.status || null,
      time_ms: elapsed,
      success: false,
      error:
        error.response?.data ||
        error.message
    };
  }

  // ------------------------------------------------
  // TEST 3 - THINKING + REASONING HIGH
  // ------------------------------------------------

  console.log("");
  console.log("TEST 3 - KIMI FULL REASONING");
  console.log("Thinking: true");
  console.log("Reasoning: high");

  const test3Start = Date.now();

  try {
    const response = await axios.post(
      `${NVIDIA_API_BASE}/chat/completions`,
      {
        ...baseRequest,
        reasoning_effort: "high",
        chat_template_kwargs: {
          enable_thinking: true
        }
      },
      {
        headers,
        timeout: 300000
      }
    );

    const elapsed = Date.now() - test3Start;

    console.log("TEST 3 STATUS:", response.status);
    console.log("TEST 3 TIME:", elapsed, "ms");

    results.tests.reasoning_high = {
      status: response.status,
      time_ms: elapsed,
      success: true,
      text_received: Boolean(
        response.data?.choices?.[0]?.message?.content
      )
    };
  } catch (error) {
    const elapsed = Date.now() - test3Start;

    console.log("TEST 3 ERROR");
    console.log("TEST 3 TIME:", elapsed, "ms");
    console.log(
      error.response?.data || error.message
    );

    results.tests.reasoning_high = {
      status: error.response?.status || null,
      time_ms: elapsed,
      success: false,
      error:
        error.response?.data ||
        error.message
    };
  }

  console.log("");
  console.log("==========================================");
  console.log("KIMI K3 DIAGNOSTIC FINISHED");
  console.log("==========================================");

  res.json(results);
});

// ==================================================
// CHAT COMPLETIONS
// ==================================================

app.post("/v1/chat/completions", async (req, res) => {
  const requestStart = Date.now();

  if (!NVIDIA_API_KEY) {
    return res.status(500).json({
      error: {
        message: "NVIDIA_API_KEY is not configured",
        type: "configuration_error"
      }
    });
  }

  const body = req.body || {};

  const requestedModel = body.model;
  const model = resolveModel(requestedModel);

  const messages = Array.isArray(body.messages)
    ? [...body.messages]
    : [];

  const temperature =
    typeof body.temperature === "number"
      ? body.temperature
      : DEFAULT_TEMPERATURE;

  const maxTokens =
    typeof body.max_tokens === "number"
      ? body.max_tokens
      : DEFAULT_MAX_TOKENS;

  const stream = body.stream === true;

  console.log("Received chat completion request.");
  console.log("Model requested:", requestedModel);
  console.log("Model sent to NVIDIA:", model);
  console.log("Temperature:", temperature);
  console.log("Max tokens:", maxTokens);
  console.log("Stream:", stream);
  console.log("Reasoning effort:", REASONING_EFFORT);

  // ==================================================
  // ROLEPLAY INSTRUCTION
  // ==================================================

  messages.unshift({
    role: "system",
    content:
      "Stay in character and respond naturally to the roleplay. " +
      "Do not speak, act, think, feel, decide, or narrate actions " +
      "for the user's character. Only control the characters and " +
      "world elements that belong to you."
  });

  // ==================================================
  // NVIDIA REQUEST
  // ==================================================

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

  const headers = {
    Authorization: `Bearer ${NVIDIA_API_KEY}`,
    "Content-Type": "application/json"
  };

  console.log(
    "Sending request to NVIDIA:",
    `${NVIDIA_API_BASE}/chat/completions`
  );

  // ==================================================
  // NON-STREAM
  // ==================================================

  if (!stream) {
    try {
      const response = await axios.post(
        `${NVIDIA_API_BASE}/chat/completions`,
        nimRequest,
        {
          headers,
          timeout: 300000
        }
      );

      const totalTime = Date.now() - requestStart;

      console.log("NVIDIA status:", response.status);
      console.log("TIME TOTAL:", totalTime, "ms");

      let output =
        response.data?.choices?.[0]?.message?.content || "";

      if (typeof output !== "string") {
        output = String(output);
      }

      if (output.length > MAX_RESPONSE_CHARS) {
        output = output.slice(0, MAX_RESPONSE_CHARS);
      }

      if (response.data?.choices?.[0]?.message) {
        response.data.choices[0].message.content = output;
      }

      return res.status(response.status).json(response.data);
    } catch (error) {
      const totalTime = Date.now() - requestStart;

      console.log(
        "NVIDIA error after",
        totalTime,
        "ms"
      );

      console.log(
        error.response?.data || error.message
      );

      return res.status(
        error.response?.status || 500
      ).json({
        error: {
          message:
            error.response?.data ||
            error.message,
          type: "upstream_error"
        }
      });
    }
  }

  // ==================================================
  // STREAM
  // ==================================================

  try {
    const response = await axios.post(
      `${NVIDIA_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers,
        responseType: "stream",
        timeout: 300000
      }
    );

    console.log("NVIDIA status:", response.status);
    console.log("NVIDIA HTTP response received.");

    res.status(200);

    res.setHeader(
      "Content-Type",
      "text/event-stream; charset=utf-8"
    );
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Accel-Buffering", "no");

    if (res.flushHeaders) {
      res.flushHeaders();
    }

    let buffer = "";
    let sentChars = 0;
    let finished = false;
    let firstTokenReceived = false;

    const sendFinish = () => {
      if (finished) {
        return;
      }

      res.write(
        "data: " +
          JSON.stringify({
            id: "chatcmpl-nvidia",
            object: "chat.completion.chunk",
            choices: [
              {
                index: 0,
                delta: {},
                finish_reason: "stop"
              }
            ]
          }) +
          "\n\n"
      );

      res.write("data: [DONE]\n\n");

      finished = true;
      res.end();
    };

    response.data.on("data", (chunk) => {
      const text = chunk.toString();

      if (!firstTokenReceived && text.trim()) {
        firstTokenReceived = true;

        console.log(
          "FIRST TOKEN RECEIVED:",
          Date.now() - requestStart,
          "ms"
        );
      }

      buffer += text;

      const parts = buffer.split("\n");
      buffer = parts.pop() || "";

      for (const line of parts) {
        const trimmed = line.trim();

        if (!trimmed) {
          continue;
        }

        if (!trimmed.startsWith("data:")) {
          continue;
        }

        const data = trimmed.slice(5).trim();

        if (data === "[DONE]") {
          sendFinish();
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

        let content = delta.content || "";

        if (typeof content !== "string") {
          content = String(content);
        }

        if (!content) {
          continue;
        }

        if (
          sentChars + content.length >
          MAX_RESPONSE_CHARS
        ) {
          content = content.slice(
            0,
            MAX_RESPONSE_CHARS - sentChars
          );
        }

        if (!content) {
          continue;
        }

        sentChars += content.length;

        res.write(
          "data: " +
            JSON.stringify({
              id:
                parsed.id ||
                "chatcmpl-nvidia",
              object:
                "chat.completion.chunk",
              created:
                parsed.created ||
                Math.floor(Date.now() / 1000),
              model: "gpt-4o",
              choices: [
                {
                  index: 0,
                  delta: {
                    content
                  },
                  finish_reason: null
                }
              ]
            }) +
            "\n\n"
        );

        if (
          sentChars >= MAX_RESPONSE_CHARS
        ) {
          console.log(
            "Maximum response character limit reached."
          );

          sendFinish();
          return;
        }
      }
    });

    response.data.on("end", () => {
      console.log("NVIDIA stream ended.");
      console.log(
        "Characters sent:",
        sentChars
      );
      console.log(
        "TIME TOTAL:",
        Date.now() - requestStart,
        "ms"
      );

      sendFinish();
    });

    response.data.on("error", (error) => {
      console.log(
        "NVIDIA streaming error:",
        error.message
      );

      if (!finished) {
        res.write(
          "data: " +
            JSON.stringify({
              error: {
                message: error.message,
                type: "upstream_error"
              }
            }) +
            "\n\n"
        );

        res.end();
        finished = true;
      }
    });
  } catch (error) {
    const totalTime = Date.now() - requestStart;

    console.log(
      "NVIDIA streaming request failed after",
      totalTime,
      "ms"
    );

    console.log(
      error.response?.data ||
        error.message
    );

    if (!res.headersSent) {
      return res.status(
        error.response?.status || 500
      ).json({
        error: {
          message:
            error.response?.data ||
            error.message,
          type: "upstream_error"
        }
      });
    }

    res.end();
  }
});

// ==================================================
// ROOT
// ==================================================

app.get("/", (req, res) => {
  res.json({
    status: "ok",
    service: "OpenAI to NVIDIA NIM Proxy",
    model: DEFAULT_MODEL,
    debug_endpoint: "/debug-nvidia"
  });
});

// ==================================================
// 404
// ==================================================

app.use((req, res) => {
  res.status(404).json({
    error: "Not found"
  });
});

// ==================================================
// START
// ==================================================

app.listen(PORT, () => {
  console.log("");
  console.log("==========================================");
  console.log("OpenAI -> NVIDIA NIM Proxy");
  console.log("==========================================");
  console.log("Port:", PORT);
  console.log("Model:", DEFAULT_MODEL);
  console.log("Temperature:", DEFAULT_TEMPERATURE);
  console.log("Max tokens:", DEFAULT_MAX_TOKENS);
  console.log("Reasoning effort:", REASONING_EFFORT);
  console.log("Thinking enabled:", ENABLE_THINKING);
  console.log("Reasoning display:", SHOW_REASONING);
  console.log(
    "NVIDIA API configured:",
    Boolean(NVIDIA_API_KEY)
  );
  console.log("==========================================");
});
