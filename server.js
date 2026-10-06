
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
const DEFAULT_MODEL = process.env.NVIDIA_MODEL || "moonshotai/kimi-k3";

const SHOW_REASONING =
  String(process.env.SHOW_REASONING || "false").toLowerCase() === "true";

const REASONING_EFFORT = process.env.REASONING_EFFORT || "high";

const MAX_RESPONSE_CHARS = 19000;
const DEFAULT_MAX_TOKENS = 8192;
const DEFAULT_TEMPERATURE = 1.0;

const ROLEPLAY_BOUNDARY_INSTRUCTION = `
IMPORTANT ROLEPLAY RULES:
- You control only your own character and NPCs.
- Never write dialogue, actions, thoughts, feelings, intentions, decisions, or reactions for the user's character.
- Never assume the user's character agrees, moves, responds, or feels anything unless the user explicitly says so.
- Describe your own character, NPCs, and the environment.
- You may describe observable consequences of your character's actions, but never decide how the user's character responds.
- Leave the user's character entirely under the user's control.
- End at a natural point where the user can respond.
- Follow these boundaries throughout the roleplay.
`.trim();

// --------------------------------------------------
// Middleware
// --------------------------------------------------

app.use(cors());

app.use(express.json({ limit: "5mb" }));

app.use(express.urlencoded({
  extended: true,
  limit: "5mb"
}));

app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} ${req.method} ${req.path}`);
  next();
});

// --------------------------------------------------
// Model aliases
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
  if (!requestedModel) return DEFAULT_MODEL;

  if (requestedModel.startsWith("nvidia/")) {
    return requestedModel;
  }

  return MODEL_MAPPING[requestedModel] || DEFAULT_MODEL;
}

// --------------------------------------------------
// Add roleplay instruction without deleting the
// system/developer instructions supplied by Janitor.
// --------------------------------------------------

function addRoleplayBoundary(messages) {
  const result = messages.map((message) => ({ ...message }));

  let insertIndex = 0;

  while (
    insertIndex < result.length &&
    ["system", "developer"].includes(result[insertIndex].role)
  ) {
    insertIndex++;
  }

  result.splice(insertIndex, 0, {
    role: "system",
    content: ROLEPLAY_BOUNDARY_INSTRUCTION
  });

  return result;
}

// --------------------------------------------------
// Health endpoints
// --------------------------------------------------

function healthResponse(req, res) {
  res.json({
    status: "ok",
    service: "OpenAI to NVIDIA NIM Proxy",
    model: DEFAULT_MODEL,
    reasoning_display: SHOW_REASONING,
    reasoning_effort: REASONING_EFFORT,
    max_response_characters: MAX_RESPONSE_CHARS,
    streaming: "enabled",
    nim_api_configured: Boolean(NVIDIA_API_KEY)
  });
}

app.get("/", healthResponse);
app.get("/health", healthResponse);
app.get("/health/", healthResponse);

// --------------------------------------------------
// Models endpoint
// --------------------------------------------------

app.get("/v1/models", (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map((id) => ({
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

  let upstream = null;
  let clientDisconnected = false;

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
        : DEFAULT_TEMPERATURE;

    const maxTokens =
      body.max_tokens ??
      body.max_completion_tokens ??
      DEFAULT_MAX_TOKENS;

    const wantsStream = body.stream === true;

    const nimRequest = {
      model,
      messages: addRoleplayBoundary(body.messages),
      temperature,
      max_tokens: maxTokens,
      reasoning_effort: REASONING_EFFORT,
      stream: wantsStream
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

    console.log("Model:", model);
    console.log("Temperature:", temperature);
    console.log("Max tokens:", maxTokens);
    console.log("Reasoning effort:", REASONING_EFFORT);
    console.log("Stream requested:", wantsStream);
    console.log("Character limit:", MAX_RESPONSE_CHARS);

    // ------------------------------------------------
    // REAL STREAMING
    // ------------------------------------------------

    if (wantsStream) {
      const nvidiaStart = Date.now();

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

      console.log("NVIDIA status:", upstream.status);
      console.log("NVIDIA streaming connection opened.");

      res.status(200);
      res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
      res.setHeader("Cache-Control", "no-cache, no-transform");
      res.setHeader("Connection", "keep-alive");
      res.setHeader("X-Accel-Buffering", "no");

      if (typeof res.flushHeaders === "function") {
        res.flushHeaders();
      }

      const streamId = `chatcmpl-${Date.now()}`;
      const created = Math.floor(Date.now() / 1000);

      let buffer = "";
      let totalCharacters = 0;
      let roleSent = false;
      let finished = false;
      let finishReason = "stop";

      function sendEvent(payload) {
        if (finished || clientDisconnected || res.writableEnded) {
          return;
        }

        res.write(`data: ${JSON.stringify(payload)}\n\n`);

        if (typeof res.flush === "function") {
          res.flush();
        }
      }

      function sendRole() {
        if (roleSent) return;
        roleSent = true;

        sendEvent({
          id: streamId,
          object: "chat.completion.chunk",
          created,
          model,
          choices: [{
            index: 0,
            delta: { role: "assistant" },
            finish_reason: null
          }]
        });
      }

      function finishStream(reason = "stop") {
        if (finished || clientDisconnected || res.writableEnded) {
          return;
        }

        finished = true;

        sendEvent({
          id: streamId,
          object: "chat.completion.chunk",
          created,
          model,
          choices: [{
            index: 0,
            delta: {},
            finish_reason: reason
          }]
        });

        res.write("data: [DONE]\n\n");
        res.end();

        console.log("Stream finished:", reason);
        console.log("Characters sent:", totalCharacters);
        console.log(
          "TIME NVIDIA:",
          Date.now() - nvidiaStart,
          "ms"
        );
        console.log(
          "TIME TOTAL:",
          Date.now() - requestStart,
          "ms"
        );
        console.log("==========================================");
      }

      function processSSELine(line) {
        if (finished || clientDisconnected) return;

        line = line.trim();

        if (!line || line.startsWith(":")) return;
        if (!line.startsWith("data:")) return;

        const raw = line.slice(5).trim();

        if (!raw) return;

        if (raw === "[DONE]") {
          finishStream(finishReason);
          return;
        }

        let parsed;

        try {
          parsed = JSON.parse(raw);
        } catch (_) {
          console.log("Skipping malformed SSE data line.");
          return;
        }

        const choice = parsed?.choices?.[0];

        if (!choice) return;

        const delta = choice.delta || {};

        sendRole();

        // Do not expose internal reasoning by default.
        if (
          SHOW_REASONING &&
          typeof delta.reasoning_content === "string" &&
          delta.reasoning_content.length > 0
        ) {
          const reasoningCharacters =
            Array.from(delta.reasoning_content);

          const remaining =
            MAX_RESPONSE_CHARS - totalCharacters;

          if (remaining > 0) {
            const textToSend = reasoningCharacters
              .slice(0, remaining)
              .join("");

            if (textToSend) {
              totalCharacters += Array.from(textToSend).length;

              sendEvent({
                id: parsed.id || streamId,
                object: "chat.completion.chunk",
                created: parsed.created || created,
                model: parsed.model || model,
                choices: [{
                  index: 0,
                  delta: { content: textToSend },
                  finish_reason: null
                }]
              });
            }
          }
        }

        if (
          typeof delta.content === "string" &&
          delta.content.length > 0
        ) {
          const chars = Array.from(delta.content);
          const remaining = MAX_RESPONSE_CHARS - totalCharacters;

          if (remaining <= 0) {
            finishReason = "length";
            finishStream("length");

            if (upstream && upstream.data) {
              upstream.data.destroy();
            }

            return;
          }

          const textToSend = chars.slice(0, remaining).join("");
          const sentCharacters = Array.from(textToSend).length;

          if (sentCharacters > 0) {
            totalCharacters += sentCharacters;

            sendEvent({
              id: parsed.id || streamId,
              object: "chat.completion.chunk",
              created: parsed.created || created,
              model: parsed.model || model,
              choices: [{
                index: 0,
                delta: { content: textToSend },
                finish_reason: null
              }]
            });
          }

          if (sentCharacters < chars.length ||
              totalCharacters >= MAX_RESPONSE_CHARS) {
            finishReason = "length";
            finishStream("length");

            if (upstream && upstream.data) {
              upstream.data.destroy();
            }
          }
        }

        if (choice.finish_reason && !finished) {
          finishReason = choice.finish_reason;
        }
      }

      upstream.data.on("data", (chunk) => {
        if (finished || clientDisconnected) return;

        buffer += chunk.toString("utf8");

        const lines = buffer.split(/\r?\n/);
        buffer = lines.pop() || "";

        for (const line of lines) {
          processSSELine(line);

          if (finished || clientDisconnected) break;
        }
      });

      upstream.data.on("end", () => {
        if (finished || clientDisconnected) return;

        if (buffer.trim()) {
          processSSELine(buffer);
        }

        if (!finished && !clientDisconnected) {
          finishStream(finishReason);
        }
      });

      upstream.data.on("error", (error) => {
        console.error("NVIDIA streaming error:", error.message);

        if (!finished && !clientDisconnected) {
          if (!res.headersSent) {
            res.status(502).json({
              error: {
                message: "NVIDIA streaming failed.",
                type: "upstream_error"
              }
            });
          } else {
            res.end();
          }
        }
      });

      // Only cancel upstream when the client actually
      // disconnects before the response has finished.
      res.on("close", () => {
        if (!res.writableEnded && !finished) {
          clientDisconnected = true;

          console.log("Janitor disconnected before stream completion.");

          if (upstream && upstream.data) {
            upstream.data.destroy();
          }
        }
      });

      return;
    }

    // ------------------------------------------------
    // NON-STREAMING REQUEST
    // ------------------------------------------------

    const nvidiaStart = Date.now();

    const response = await axios.post(
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

    console.log(
      "TIME NVIDIA:",
      Date.now() - nvidiaStart,
      "ms"
    );

    const data = response.data;
    const choice = data?.choices?.[0] || {};
    const message = choice.message || {};

    let content =
      typeof message.content === "string"
        ? message.content
        : "";

    const chars = Array.from(content);

    if (chars.length > MAX_RESPONSE_CHARS) {
      content = chars.slice(0, MAX_RESPONSE_CHARS).join("");
      console.log("Non-streaming response truncated to character limit.");
    }

    const responseMessage = {
      role: "assistant",
      content
    };

    if (SHOW_REASONING && message.reasoning_content) {
      responseMessage.reasoning_content = message.reasoning_content;
    }

    const output = {
      id: data.id || `chatcmpl-${Date.now()}`,
      object: "chat.completion",
      created: data.created || Math.floor(Date.now() / 1000),
      model,
      choices: [{
        index: 0,
        message: responseMessage,
        finish_reason: choice.finish_reason || "stop"
      }]
    };

    if (data.usage) {
      output.usage = data.usage;
    }

    console.log("TIME TOTAL:", Date.now() - requestStart, "ms");
    console.log("==========================================");

    return res.status(200).json(output);
  } catch (error) {
    console.error("Proxy request failed:", error.message);

    if (error.response) {
      console.error("NVIDIA status:", error.response.status);

      if (error.response.data && !error.response.headers?.["content-type"]?.includes("text/event-stream")) {
        console.error(
          "NVIDIA error data:",
          JSON.stringify(error.response.data)
        );
      }
    }

    if (!res.headersSent) {
      return res.status(error.response?.status || 502).json({
        error: {
          message: error.message || "NVIDIA request failed.",
          type: "upstream_error"
        }
      });
    }

    if (!res.writableEnded) {
      res.end();
    }
  }
});

// --------------------------------------------------
// Start server
// --------------------------------------------------

app.listen(PORT, () => {
  console.log("==========================================");
  console.log("OpenAI to NVIDIA NIM Proxy");
  console.log("Server listening on port", PORT);
  console.log("NVIDIA API configured:", Boolean(NVIDIA_API_KEY));
  console.log("Default model:", DEFAULT_MODEL);
  console.log("Reasoning effort:", REASONING_EFFORT);
  console.log("Temperature default:", DEFAULT_TEMPERATURE);
  console.log("Max tokens default:", DEFAULT_MAX_TOKENS);
  console.log("Max response characters:", MAX_RESPONSE_CHARS);
  console.log("Real streaming: enabled");
  console.log("Roleplay boundary: enabled");
  console.log("==========================================");
});
