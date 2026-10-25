
"use strict";

const express = require("express");
const cors = require("cors");
const axios = require("axios");
const { StringDecoder } = require("string_decoder");

const app = express();

const PORT = Number(process.env.PORT) || 3000;

const NVIDIA_API_BASE = (
  process.env.NVIDIA_API_BASE ||
  "https://integrate.api.nvidia.com/v1"
).replace(/\/+$/, "");

const NVIDIA_API_KEY = process.env.NVIDIA_API_KEY || "";

const DEFAULT_MODEL =
  process.env.NVIDIA_MODEL || "z-ai/glm-5.3-flash";

const DEFAULT_TEMPERATURE = Number(
  process.env.DEFAULT_TEMPERATURE || 1.0
);

const DEFAULT_TOP_P = Number(
  process.env.DEFAULT_TOP_P || 0.95
);

const DEFAULT_MAX_TOKENS = Number(
  process.env.DEFAULT_MAX_TOKENS || 16384
);

const REASONING_EFFORT =
  process.env.REASONING_EFFORT || "max";

const REQUEST_TIMEOUT = Number(
  process.env.REQUEST_TIMEOUT || 300000
);

const ROLEPLAY_INSTRUCTIONS = `
You are an immersive, coherent, multi-character roleplay writer.

WRITING QUALITY:
- Write polished, natural, vivid, immersive prose.
- Balance dialogue, action, atmosphere, and meaningful reactions.
- Give every character a distinct voice, vocabulary, personality, and rhythm.
- Make dialogue sound natural rather than repetitive or overly formal.
- Vary sentence structure and descriptions.
- Avoid repeating phrases, information, gestures, and emotional reactions.
- Do not pad responses with unnecessary words.
- Develop scenes with appropriate pacing and substantial detail.
- Continue the current scene instead of summarizing previous events.
- Prefer meaningful development over unnecessary length.

USER CHARACTER AUTONOMY:
- Never control the user-controlled character.
- Never invent their dialogue, thoughts, feelings, decisions, or actions.
- Do not assume they agree, move, speak, or react unless the user establishes it.
- Describe other characters and the environment while leaving room for the
  user to decide what their character does.

CHARACTER CONSISTENCY:
- Preserve established personalities, motivations, relationships, and voices.
- Keep character knowledge consistent with events they have witnessed.
- Respect the established chronology and previous messages.
- Let relationships and emotional changes develop gradually and believably.
- Do not make every character sound alike.
- Avoid abrupt plot changes without narrative grounding.

IMMERSION:
- Stay in character and write the scene directly.
- Do not add OOC commentary, meta explanations, or writing-process notes.
- Do not repeat the user's message just to make the response longer.
- Avoid ending every response with the same question or dramatic pattern.
- Let the scene progress naturally and leave meaningful opportunities for
  the user to respond.
`.trim();

app.disable("x-powered-by");

app.use(cors());
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true, limit: "10mb" }));

// --------------------------------------------------
// HEALTH CHECKS
// --------------------------------------------------

function healthResponse(req, res) {
  res.json({
    status: "ok",
    service: "Janitor AI - NVIDIA NIM Proxy",
    model: DEFAULT_MODEL,
    default_max_tokens: DEFAULT_MAX_TOKENS,
    temperature: DEFAULT_TEMPERATURE,
    top_p: DEFAULT_TOP_P,
    reasoning_effort: REASONING_EFFORT,
    api_key_configured: Boolean(NVIDIA_API_KEY)
  });
}

app.get("/", healthResponse);
app.get("/health", healthResponse);
app.get("/health/", healthResponse);

// --------------------------------------------------
// MODEL LIST
// --------------------------------------------------

app.get("/v1/models", (req, res) => {
  res.json({
    object: "list",
    data: [
      {
        id: DEFAULT_MODEL,
        object: "model",
        created: 0,
        owned_by: "z-ai"
      },
      {
        id: "gpt-4o",
        object: "model",
        created: 0,
        owned_by: "proxy"
      }
    ]
  });
});

// --------------------------------------------------
// HELPERS
// --------------------------------------------------

function clamp(value, min, max, fallback) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return fallback;
  }

  return Math.min(max, Math.max(min, number));
}

function normalizeMessages(inputMessages) {
  return inputMessages
    .filter(
      (message) =>
        message &&
        typeof message.role === "string" &&
        message.content !== undefined
    )
    .map((message) => {
      const normalized = {
        role: message.role,
        content: message.content
      };

      if (message.name !== undefined) {
        normalized.name = message.name;
      }

      if (message.tool_calls !== undefined) {
        normalized.tool_calls = message.tool_calls;
      }

      if (message.tool_call_id !== undefined) {
        normalized.tool_call_id = message.tool_call_id;
      }

      return normalized;
    });
}

function prepareMessages(inputMessages) {
  const messages = normalizeMessages(inputMessages);

  const systemIndex = messages.findIndex(
    (message) => message.role === "system"
  );

  if (systemIndex === -1) {
    messages.unshift({
      role: "system",
      content: ROLEPLAY_INSTRUCTIONS
    });
  } else if (typeof messages[systemIndex].content === "string") {
    messages[systemIndex] = {
      ...messages[systemIndex],
      content:
        messages[systemIndex].content +
        "\n\n" +
        ROLEPLAY_INSTRUCTIONS
    };
  } else {
    messages.unshift({
      role: "system",
      content: ROLEPLAY_INSTRUCTIONS
    });
  }

  return messages;
}

// --------------------------------------------------
// UPSTREAM ERROR DETAILS
// --------------------------------------------------

async function readStreamText(stream, maxLength = 8000) {
  let result = "";

  try {
    for await (const chunk of stream) {
      if (result.length < maxLength) {
        result += Buffer.from(chunk)
          .toString("utf8")
          .slice(0, maxLength - result.length);
      }
    }
  } catch (error) {
    result += "\nCould not read the complete NVIDIA error body.";
  }

  return result || "NVIDIA returned an empty error response.";
}

async function getUpstreamErrorDetails(error) {
  const data = error.response?.data;

  if (data && typeof data[Symbol.asyncIterator] === "function") {
    return await readStreamText(data);
  }

  if (typeof data === "string") {
    return data.slice(0, 8000);
  }

  if (data !== undefined && data !== null) {
    try {
      return JSON.stringify(data).slice(0, 8000);
    } catch {
      return String(data);
    }
  }

  return error.message || "Unknown NVIDIA error.";
}

// --------------------------------------------------
// STREAMING HELPERS
// --------------------------------------------------

// Remove private reasoning fields while preserving the normal
// OpenAI-compatible content chunks and SSE event structure.

function sanitizeSSEEvent(eventText) {
  const lines = eventText.split(/\r?\n/);

  const outputLines = lines.map((line) => {
    if (!line.startsWith("data:")) {
      return line;
    }

    const raw = line.slice(5).trimStart();

    if (!raw || raw === "[DONE]") {
      return line;
    }

    try {
      const event = JSON.parse(raw);

      if (Array.isArray(event.choices)) {
        for (const choice of event.choices) {
          if (choice.delta && typeof choice.delta === "object") {
            delete choice.delta.reasoning_content;
            delete choice.delta.reasoning;
          }

          if (
            choice.message &&
            typeof choice.message === "object"
          ) {
            delete choice.message.reasoning_content;
            delete choice.message.reasoning;
          }
        }
      }

      return "data: " + JSON.stringify(event);
    } catch {
      return line;
    }
  });

  return outputLines.join("\n") + "\n\n";
}

function buildNvidiaPayload(body, stream) {
  const temperature = clamp(
    body.temperature ?? DEFAULT_TEMPERATURE,
    0,
    2,
    DEFAULT_TEMPERATURE
  );

  const topP = clamp(
    body.top_p ?? DEFAULT_TOP_P,
    0,
    1,
    DEFAULT_TOP_P
  );

  const requestedTokens = Number(
    body.max_tokens ??
    body.max_completion_tokens ??
    DEFAULT_MAX_TOKENS
  );

  // Maintain the configured high output budget even if Janitor
  // sends a smaller limit. Absolute configured cap: 32768.
  const maxTokens = Math.round(
    clamp(
      Math.max(
        DEFAULT_MAX_TOKENS,
        Number.isFinite(requestedTokens) ? requestedTokens : 0
      ),
      1,
      32768,
      DEFAULT_MAX_TOKENS
    )
  );

  const payload = {
    model: DEFAULT_MODEL,
    messages: prepareMessages(body.messages),
    temperature,
    top_p: topP,
    max_tokens: maxTokens,
    stream,
    reasoning_effort: REASONING_EFFORT,
    chat_template_kwargs: {
      clear_thinking: true
    }
  };

  if (body.stop !== undefined) {
    payload.stop = body.stop;
  }

  if (body.frequency_penalty !== undefined) {
    payload.frequency_penalty = clamp(
      body.frequency_penalty,
      -2,
      2,
      0
    );
  }

  if (body.presence_penalty !== undefined) {
    payload.presence_penalty = clamp(
      body.presence_penalty,
      -2,
      2,
      0
    );
  }

  if (
    body.seed !== undefined &&
    Number.isFinite(Number(body.seed))
  ) {
    payload.seed = Number(body.seed);
  }

  return payload;
}

// --------------------------------------------------
// CHAT COMPLETIONS
// --------------------------------------------------

app.post("/v1/chat/completions", async (req, res) => {
  const body = req.body || {};

  const stream =
    body.stream === true ||
    String(body.stream).toLowerCase() === "true";

  console.log("[Proxy] Chat completion request.", {
    model: DEFAULT_MODEL,
    stream,
    max_tokens: body.max_tokens ?? DEFAULT_MAX_TOKENS,
    reasoning_effort: REASONING_EFFORT,
    message_count: Array.isArray(body.messages)
      ? body.messages.length
      : 0
  });

  if (!NVIDIA_API_KEY) {
    return res.status(500).json({
      error: {
        message:
          "NVIDIA_API_KEY is missing. Configure it in Render Environment.",
        type: "configuration_error"
      }
    });
  }

  if (
    !Array.isArray(body.messages) ||
    body.messages.length === 0
  ) {
    return res.status(400).json({
      error: {
        message: "The request must contain a non-empty messages array.",
        type: "invalid_request_error"
      }
    });
  }

  const payload = buildNvidiaPayload(body, stream);

  try {
    const upstream = await axios.post(
      `${NVIDIA_API_BASE}/chat/completions`,
      payload,
      {
        headers: {
          Authorization: `Bearer ${NVIDIA_API_KEY}`,
          "Content-Type": "application/json",
          Accept: stream
            ? "text/event-stream"
            : "application/json"
        },
        timeout: REQUEST_TIMEOUT,
        responseType: stream ? "stream" : "json"
      }
    );

    console.log("[Proxy] NVIDIA response status:", upstream.status);

    if (!stream) {
      return res.status(upstream.status).json(upstream.data);
    }

    res.status(upstream.status);
    res.setHeader(
      "Content-Type",
      "text/event-stream; charset=utf-8"
    );
    res.setHeader("Cache-Control", "no-cache, no-transform");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Accel-Buffering", "no");

    if (typeof res.flushHeaders === "function") {
      res.flushHeaders();
    }

    const decoder = new StringDecoder("utf8");

    let buffer = "";
    let paused = false;
    let streamEnded = false;

    function processBuffer() {
      if (paused || res.destroyed || res.writableEnded) {
        return;
      }

      while (true) {
        const match = /\r?\n\r?\n/.exec(buffer);

        if (!match) {
          break;
        }

        const eventText = buffer.slice(0, match.index);
        buffer = buffer.slice(match.index + match[0].length);

        const output = sanitizeSSEEvent(eventText);

        if (!res.write(output)) {
          paused = true;
          upstream.data.pause();
          return;
        }
      }

      if (streamEnded && buffer.length > 0) {
        const output = sanitizeSSEEvent(buffer.trimEnd());
        buffer = "";

        if (!res.write(output)) {
          paused = true;
          return;
        }
      }

      if (streamEnded && !res.writableEnded) {
        res.end();
        console.log("[Proxy] NVIDIA stream completed.");
      }
    }

    upstream.data.on("data", (chunk) => {
      buffer += decoder.write(chunk);
      processBuffer();
    });

    res.on("drain", () => {
      paused = false;

      if (!streamEnded && !upstream.data.destroyed) {
        upstream.data.resume();
      }

      processBuffer();
    });

    upstream.data.on("end", () => {
      buffer += decoder.end();
      streamEnded = true;
      processBuffer();
    });

    upstream.data.on("error", (error) => {
      console.error("[Proxy] NVIDIA stream error:", error.message);

      if (!res.destroyed && !res.writableEnded) {
        res.end();
      }
    });

    res.on("close", () => {
      if (!res.writableEnded && !upstream.data.destroyed) {
        upstream.data.destroy();
      }
    });

    return;
  } catch (error) {
    const upstreamStatus = error.response?.status || null;
    const details = await getUpstreamErrorDetails(error);

    console.error("[Proxy] NVIDIA request failed.", {
      upstream_status: upstreamStatus,
      message: error.message,
      details
    });

    if (res.headersSent) {
      if (!res.destroyed && !res.writableEnded) {
        res.end();
      }

      return;
    }

    const clientStatus =
      upstreamStatus >= 400 && upstreamStatus < 500
        ? upstreamStatus
        : 502;

    return res.status(clientStatus).json({
      error: {
        message: "NVIDIA NIM request failed.",
        type: "upstream_error",
        upstream_status: upstreamStatus,
        details
      }
    });
  }
});

// --------------------------------------------------
// NOT FOUND
// --------------------------------------------------

app.use((req, res) => {
  res.status(404).json({
    error: {
      message: `Route not found: ${req.method} ${req.path}`,
      type: "not_found"
    }
  });
});

// --------------------------------------------------
// GENERAL ERROR HANDLER
// --------------------------------------------------

app.use((error, req, res, next) => {
  console.error("[Proxy] Server error:", error.message);

  if (res.headersSent) {
    return next(error);
  }

  return res.status(500).json({
    error: {
      message: "Internal proxy server error.",
      type: "internal_error"
    }
  });
});

// --------------------------------------------------
// START SERVER
// --------------------------------------------------

app.listen(PORT, "0.0.0.0", () => {
  console.log("----------------------------------------");
  console.log("Janitor AI - NVIDIA NIM Proxy");
  console.log("Port:", PORT);
  console.log("Model:", DEFAULT_MODEL);
  console.log("Default max tokens:", DEFAULT_MAX_TOKENS);
  console.log("Reasoning effort:", REASONING_EFFORT);
  console.log("NVIDIA API configured:", Boolean(NVIDIA_API_KEY));
  console.log("----------------------------------------");
});
