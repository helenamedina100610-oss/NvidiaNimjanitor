
const express = require("express");
const cors = require("cors");
const axios = require("axios");
const { StringDecoder } = require("string_decoder");

const app = express();
const PORT = process.env.PORT || 3000;

const NVIDIA_API_BASE = (
  process.env.NVIDIA_API_BASE ||
  "https://integrate.api.nvidia.com/v1"
).replace(/\/+$/, "");

const NVIDIA_API_KEY = process.env.NVIDIA_API_KEY || "";

const DEFAULT_MODEL =
  process.env.NVIDIA_MODEL || "moonshotai/kimi-k3";

const DEFAULT_TEMPERATURE = 1.0;
const DEFAULT_MAX_TOKENS = 16384;
const MAX_RESPONSE_CHARS = 19000;

const REASONING_EFFORT =
  process.env.REASONING_EFFORT || "high";

const SHOW_REASONING =
  String(process.env.SHOW_REASONING || "false").toLowerCase() === "true";

const ROLEPLAY_INSTRUCTION = `
ROLEPLAY CONTROL RULES:

You control only the characters, entities, and world elements that belong to you.

The user controls their own character completely.

NEVER speak for the user's character.
NEVER write dialogue for the user's character.
NEVER decide what the user's character says or does.
NEVER describe actions performed by the user's character.
NEVER describe thoughts, feelings, intentions, reactions, decisions, or sensations belonging to the user's character.
NEVER assume how the user's character reacts.
NEVER move, position, or control the user's character.
NEVER finish the user's character's action or sentence.

If the user's character needs to respond, stop and leave that response entirely to the user.

You may describe the environment and the actions, dialogue, thoughts, feelings, and reactions of characters you control.

Write clear, natural, well-organized prose.
Avoid repetitive descriptions, redundant dialogue, repeated information, and unnecessary restatements.
Keep character voices distinct and consistent.
Maintain continuity with the established story and previous messages.
Use coherent paragraphs, natural dialogue, and appropriate pacing.
Do not force the story to advance too quickly.
Do not narrate the user's character as if you were the user.
Always preserve the user's agency.
`;

app.use(cors());
app.use(express.json({ limit: "5mb" }));
app.use(express.urlencoded({ extended: true, limit: "5mb" }));

app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
  next();
});

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

function resolveModel(requestedModel) {
  if (!requestedModel) return DEFAULT_MODEL;

  if (MODEL_ALIASES[requestedModel]) {
    return MODEL_ALIASES[requestedModel];
  }

  if (requestedModel.includes("/")) {
    return requestedModel;
  }

  return DEFAULT_MODEL;
}

function healthResponse() {
  return {
    status: "ok",
    service: "OpenAI to NVIDIA NIM Proxy",
    model: DEFAULT_MODEL,
    reasoning_display: SHOW_REASONING,
    reasoning_effort: REASONING_EFFORT,
    nim_api_configured: Boolean(NVIDIA_API_KEY)
  };
}

app.get("/health", (req, res) => {
  res.json(healthResponse());
});

app.get("/health/", (req, res) => {
  res.json(healthResponse());
});

app.get("/", (req, res) => {
  res.json({
    status: "ok",
    service: "OpenAI to NVIDIA NIM Proxy",
    model: DEFAULT_MODEL
  });
});

app.get("/v1/models", (req, res) => {
  const data = Object.keys(MODEL_ALIASES).map((id) => ({
    id,
    object: "model",
    created: Math.floor(Date.now() / 1000),
    owned_by: "nvidia-nim"
  }));

  if (!data.some((item) => item.id === DEFAULT_MODEL)) {
    data.push({
      id: DEFAULT_MODEL,
      object: "model",
      created: Math.floor(Date.now() / 1000),
      owned_by: "nvidia"
    });
  }

  res.json({ object: "list", data });
});

// Diagnostic endpoint: test the NVIDIA API without streaming.
app.get("/debug-nvidia", async (req, res) => {
  if (!NVIDIA_API_KEY) {
    return res.status(500).json({
      error: "NVIDIA_API_KEY is not configured"
    });
  }

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

      const choice = response.data?.choices?.[0];

      results[name] = {
        status: response.status,
        time_ms: Date.now() - start,
        success: response.status === 200,
        text_received:
          typeof choice?.message?.content === "string" &&
          choice.message.content.length > 0,
        finish_reason: choice?.finish_reason || null
      };
    } catch (error) {
      results[name] = {
        status: error.response?.status || 0,
        time_ms: Date.now() - start,
        success: false,
        error:
          error.response?.data ||
          error.message ||
          "Unknown error"
      };
    }
  }

  await runTest("basic", {
    model: DEFAULT_MODEL,
    messages: [
      { role: "user", content: "Reply with a short, natural greeting." }
    ],
    temperature: 1.0,
    max_tokens: 100,
    stream: false
  });

  await runTest("reasoning_max", {
    model: DEFAULT_MODEL,
    messages: [
      { role: "user", content: "Reply with a short, natural greeting." }
    ],
    temperature: 1.0,
    max_tokens: 100,
    stream: false,
    reasoning_effort: REASONING_EFFORT
  });

  res.json({
    model: DEFAULT_MODEL,
    api_base: NVIDIA_API_BASE,
    tests: results
  });
});

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

    // Preserve existing system instructions and add roleplay guidance.
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

    // Keep the requested temperature within the supported range.
    const requestedTemperature =
      typeof req.body?.temperature === "number"
        ? req.body.temperature
        : DEFAULT_TEMPERATURE;

    const temperature = Math.min(
      1.0,
      Math.max(0, requestedTemperature)
    );

    // Respect the model's token limit even if Janitor requests more.
    const requestedMaxTokens =
      typeof req.body?.max_completion_tokens === "number"
        ? req.body.max_completion_tokens
        : typeof req.body?.max_tokens === "number"
          ? req.body.max_tokens
          : DEFAULT_MAX_TOKENS;

    const maxTokens = Math.min(
      DEFAULT_MAX_TOKENS,
      Math.max(1, Math.floor(requestedMaxTokens))
    );

    const stream = req.body?.stream === true;

    const nimRequest = {
      model,
      messages,
      temperature,
      max_tokens: maxTokens,
      stream,
      reasoning_effort: REASONING_EFFORT
    };

    console.log("Sending request to NVIDIA...");
    console.log("Model:", model);
    console.log("Temperature:", temperature);
    console.log("Max tokens:", maxTokens);
    console.log("Stream:", stream);
    console.log("Reasoning effort:", REASONING_EFFORT);

    const nvidiaStart = Date.now();

    // STREAMING RESPONSE
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

      console.log("NVIDIA status:", response.status);

      res.status(200);
      res.setHeader(
        "Content-Type",
        "text/event-stream; charset=utf-8"
      );
      res.setHeader(
        "Cache-Control",
        "no-cache, no-transform"
      );
      res.setHeader("Connection", "keep-alive");
      res.flushHeaders();

      const decoder = new StringDecoder("utf8");

      let sseBuffer = "";
      let outputChars = 0;
      let firstTokenReceived = false;
      let finished = false;

      function sendChunk(content) {
        if (finished || res.writableEnded) return;

        const chunk = {
          id: `chatcmpl-${Date.now()}`,
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model,
          choices: [
            {
              index: 0,
              delta: { content },
              finish_reason: null
            }
          ]
        };

        res.write(`data: ${JSON.stringify(chunk)}\n\n`);
      }

      function sendFinish() {
        if (finished) return;

        finished = true;

        if (res.writableEnded || res.destroyed) return;

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

        res.write(`data: ${JSON.stringify(chunk)}\n\n`);
        res.write("data: [DONE]\n\n");
      }

      function processSseEvent(eventText) {
        if (finished || !eventText) return;

        const dataLines = eventText
          .split(/\r?\n/)
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).replace(/^ /, ""));

        if (dataLines.length === 0) return;

        const data = dataLines.join("\n").trim();

        if (!data) return;

        if (data === "[DONE]") {
          sendFinish();

          if (!res.writableEnded) res.end();

          if (!response.data.destroyed) {
            response.data.destroy();
          }

          return;
        }

        let parsed;

        try {
          parsed = JSON.parse(data);
        } catch {
          console.warn("Ignoring invalid NVIDIA SSE event.");
          return;
        }

        const choice = parsed?.choices?.[0];

        if (!choice) return;

        const delta = choice.delta || {};

        // Do not expose the model's internal reasoning.
        let content =
          typeof delta.content === "string"
            ? delta.content
            : "";

        if (!content) return;

        if (!firstTokenReceived) {
          firstTokenReceived = true;

          console.log(
            `First token received after ${Date.now() - nvidiaStart} ms`
          );
        }

        const remaining = MAX_RESPONSE_CHARS - outputChars;

        if (remaining <= 0) {
          sendFinish();

          if (!res.writableEnded) res.end();

          if (!response.data.destroyed) {
            response.data.destroy();
          }

          return;
        }

        if (content.length > remaining) {
          content = content.slice(0, remaining);
        }

        outputChars += content.length;
        sendChunk(content);

        if (outputChars >= MAX_RESPONSE_CHARS) {
          sendFinish();

          if (!res.writableEnded) res.end();

          if (!response.data.destroyed) {
            response.data.destroy();
          }
        }
      }

      function consumeSseBuffer(final = false) {
        let separatorMatch;

        while (
          !finished &&
          (separatorMatch = /\r?\n\r?\n/.exec(sseBuffer))
        ) {
          const eventText = sseBuffer.slice(
            0,
            separatorMatch.index
          );

          sseBuffer = sseBuffer.slice(
            separatorMatch.index + separatorMatch[0].length
          );

          processSseEvent(eventText);
        }

        if (final && sseBuffer.trim() && !finished) {
          processSseEvent(sseBuffer);
          sseBuffer = "";
        }
      }

      response.data.on("data", (chunk) => {
        if (finished) return;

        sseBuffer += decoder.write(chunk);
        consumeSseBuffer();
      });

      response.data.on("end", () => {
        if (!finished) {
          sseBuffer += decoder.end();
          consumeSseBuffer(true);

          sendFinish();
        }

        if (!res.writableEnded) res.end();

        console.log("NVIDIA stream ended.");
        console.log(
          `Total request time: ${Date.now() - requestStart} ms`
        );
      });

      response.data.on("error", (error) => {
        if (finished) return;

        console.error("NVIDIA streaming error:", error.message);

        if (!res.writableEnded) res.end();
      });

      // Cancel upstream only if the client disconnects prematurely.
      res.on("close", () => {
        if (!res.writableEnded && !response.data.destroyed) {
          response.data.destroy();
        }
      });

      return;
    }

    // NON-STREAMING RESPONSE
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

    console.log("NVIDIA status:", response.status);

    const data = response.data;
    const choice = data?.choices?.[0];

    if (
      choice?.message &&
      typeof choice.message.content === "string" &&
      choice.message.content.length > MAX_RESPONSE_CHARS
    ) {
      choice.message.content =
        choice.message.content.slice(0, MAX_RESPONSE_CHARS);
    }

    console.log(
      `NVIDIA response time: ${Date.now() - nvidiaStart} ms`
    );
    console.log(
      `Total request time: ${Date.now() - requestStart} ms`
    );

    return res.status(200).json(data);
  } catch (error) {
    console.error(
      "Proxy error:",
      error.response?.data || error.message || error
    );

    if (res.headersSent) {
      if (!res.writableEnded) res.end();
      return;
    }

    const status = error.response?.status || 500;

    return res.status(status).json({
      error: {
        message:
          error.response?.data?.error?.message ||
          (typeof error.response?.data === "string"
            ? error.response.data
            : JSON.stringify(error.response?.data)) ||
          error.message ||
          "Upstream NVIDIA error",
        type: "upstream_error",
        status
      }
    });
  }
});

app.use((req, res) => {
  res.status(404).json({
    error: {
      message: "Route not found",
      type: "not_found",
      path: req.path
    }
  });
});

app.listen(PORT, () => {
  console.log(`Proxy listening on port ${PORT}`);
  console.log(`NVIDIA API: ${NVIDIA_API_BASE}`);
  console.log(`Default model: ${DEFAULT_MODEL}`);
  console.log(`Reasoning effort: ${REASONING_EFFORT}`);
  console.log(`Max tokens: ${DEFAULT_MAX_TOKENS}`);
  console.log(`Max response chars: ${MAX_RESPONSE_CHARS}`);
});
