
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
  process.env.NVIDIA_MODEL || "moonshotai/kimi-k3";

const DEFAULT_TEMPERATURE = 1.0;
const DEFAULT_MAX_TOKENS = 18000;
const DEFAULT_REASONING_EFFORT = "high";
const REQUEST_TIMEOUT_MS = 300000;
const MAX_RESPONSE_CHARS = 19000;

const reasoningSetting = String(
  process.env.REASONING_EFFORT || DEFAULT_REASONING_EFFORT
).toLowerCase();

const REASONING_EFFORT = ["low", "high", "max"].includes(reasoningSetting)
  ? reasoningSetting
  : DEFAULT_REASONING_EFFORT;

// Roleplay instructions.
const ROLEPLAY_INSTRUCTION = `
ROLEPLAY WRITING INSTRUCTIONS:

Write immersive, coherent, natural roleplay with consistent characterization.
Respect established personalities, memories, relationships, setting,
and events from the conversation.

USER CHARACTER AUTONOMY:
Never write dialogue, actions, thoughts, emotions, decisions, or reactions
for the user's character. Never assume what the user's character does or feels.
Describe other characters, their dialogue, actions, perceptions, and the
surrounding environment. Leave the user's character's response open to them.

MULTI-CHARACTER ROLEPLAY:
Portray multiple characters naturally in the same scene when appropriate.
Keep each character's dialogue, personality, motivations, reactions, and
relationships distinct and consistent.

Allow characters to speak to one another, interrupt, disagree, react to
events, and interact independently. Track who is speaking and whom each
character is addressing.

Do not reduce a scene to a single character's perspective when multiple
characters are present. Do not make every character speak in every reply;
include those who would naturally participate.

Do not make different characters sound like the same person.
Keep dialogue attribution clear through context, actions, and natural prose.

STRICT IN-CHARACTER ROLEPLAY:
Remain fully in character and inside the fictional scene.
Do not write OOC comments, out-of-character notes, author commentary,
meta explanations, or messages addressed to the user outside the story.

Never use "OOC:", "(OOC)", "Out of Character", or similar labels.
Do not discuss roleplay rules, writing quality, plot direction, or instructions
outside the story unless the user explicitly asks you to.

Do not explain what you are going to write before writing the scene.
Do not add commentary about the roleplay after the scene.
Do not end with out-of-character questions such as "What would you like
to happen next?" or "How should the story continue?"

If clarification is needed, let characters ask naturally within the scene
whenever possible.

WRITING QUALITY:
Use natural, fluent English when the conversation is in English.
Keep dialogue distinct and believable for each character.
Use descriptive prose when it improves the scene, but avoid repetitive
phrases, unnecessary recaps, filler, and excessive repetition.

Do not repeat the same sentence or paragraph within a response.
Advance the scene naturally without rushing major events or resolving
conflicts before the user can participate.

CONTINUITY:
Treat previous messages as established story context.
Do not contradict known facts or suddenly change character personalities.
Let characters interact with one another independently when appropriate.
Never control the user's character or decide their next move.

End at a natural point inside the scene where the user can respond.
Do not append OOC notes before or after the roleplay.
`.trim();

app.disable("x-powered-by");
app.use(cors());
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true, limit: "10mb" }));

function log(message, details) {
  if (details !== undefined) {
    console.log(`[Proxy] ${message}`, details);
  } else {
    console.log(`[Proxy] ${message}`);
  }
}

function makeId(prefix = "chatcmpl") {
  return `${prefix}-${crypto.randomBytes(12).toString("hex")}`;
}

function clampNumber(value, fallback, min, max) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

function resolveModel(requestedModel) {
  if (typeof requestedModel !== "string" || !requestedModel.trim()) {
    return DEFAULT_MODEL;
  }

  const model = requestedModel.trim();

  const aliases = new Set([
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

  if (aliases.has(model)) return DEFAULT_MODEL;

  if (model.startsWith("moonshotai/")) return model;
  if (model.startsWith("nvidia/")) return model;

  return DEFAULT_MODEL;
}

function getHeaders(stream = false) {
  return {
    Authorization: `Bearer ${NVIDIA_API_KEY}`,
    "Content-Type": "application/json",
    Accept: stream ? "text/event-stream" : "application/json",
  };
}

function prepareMessages(inputMessages) {
  const messages = inputMessages.map((message) => ({ ...message }));
  const systemIndex = messages.findIndex(
    (message) => message.role === "system"
  );

  if (
    systemIndex >= 0 &&
    typeof messages[systemIndex].content === "string"
  ) {
    messages[systemIndex].content +=
      "\n\n" + ROLEPLAY_INSTRUCTION;
  } else {
    messages.unshift({
      role: "system",
      content: ROLEPLAY_INSTRUCTION,
    });
  }

  return messages;
}

function getErrorDetails(error) {
  const status = error.response?.status;
  const data = error.response?.data;

  let details;

  if (typeof data === "string") {
    details = data;
  } else if (data && typeof data === "object") {
    details = JSON.stringify(data);
  } else {
    details = error.message || "Unknown NVIDIA error";
  }

  if (details.length > 2500) {
    details = details.slice(0, 2500) + "...";
  }

  return { status, details };
}

app.get("/", (req, res) => {
  res.json({
    status: "ok",
    service: "Janitor AI - NVIDIA NIM Proxy",
    model: DEFAULT_MODEL,
  });
});

app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    service: "Janitor AI - NVIDIA NIM Proxy",
    model: DEFAULT_MODEL,
    reasoning_effort: REASONING_EFFORT,
    reasoning_display: false,
    nvidia_api_configured: Boolean(NVIDIA_API_KEY),
  });
});

app.get("/health/", (req, res) => res.redirect(307, "/health"));

app.get("/v1/models", (req, res) => {
  const ids = [
    DEFAULT_MODEL,
    "gpt-4o",
    "gpt-4",
    "gpt-3.5-turbo",
  ];

  res.json({
    object: "list",
    data: ids.map((id) => ({
      id,
      object: "model",
      created: 0,
      owned_by: "moonshotai",
    })),
  });
});

app.get("/v1/models/:model", (req, res) => {
  res.json({
    id: req.params.model,
    object: "model",
    created: 0,
    owned_by: "moonshotai",
  });
});

function parseSSE(eventText) {
  const data = eventText
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trim())
    .join("\n");

  if (!data) return null;
  if (data === "[DONE]") return { done: true };

  try {
    return { json: JSON.parse(data) };
  } catch {
    log("Ignoring malformed SSE event.");
    return null;
  }
}

app.post("/v1/chat/completions", async (req, res) => {
  if (!NVIDIA_API_KEY) {
    return res.status(500).json({
      error: {
        message: "NVIDIA_API_KEY is missing from Render Environment.",
        type: "configuration_error",
      },
    });
  }

  const body = req.body || {};

  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    return res.status(400).json({
      error: {
        message: "messages must be a non-empty array.",
        type: "invalid_request_error",
      },
    });
  }

  const model = resolveModel(body.model);
  const stream = body.stream === true;

  const temperature = clampNumber(
    body.temperature,
    DEFAULT_TEMPERATURE,
    0,
    1
  );

  const requestedTokens =
    body.max_completion_tokens ?? body.max_tokens;

  const maxTokens = Math.round(
    clampNumber(requestedTokens, DEFAULT_MAX_TOKENS, 1, 65536)
  );

  const payload = {
    model,
    messages: prepareMessages(body.messages),
    temperature,
    max_tokens: maxTokens,
    stream,
    reasoning_effort: REASONING_EFFORT,
  };

  if (Number.isSafeInteger(body.seed)) {
    payload.seed = body.seed;
  }

  if (typeof body.stop === "string" || Array.isArray(body.stop)) {
    payload.stop = body.stop;
  }

  if (body.stream_options && stream) {
    payload.stream_options = body.stream_options;
  }

  log("Sending NVIDIA request.", {
    model,
    stream,
    temperature,
    max_tokens: maxTokens,
    reasoning_effort: REASONING_EFFORT,
    message_count: body.messages.length,
  });

  let upstream;

  try {
    upstream = await axios.post(
      `${NVIDIA_API_BASE}/chat/completions`,
      payload,
      {
        headers: getHeaders(stream),
        responseType: stream ? "stream" : "json",
        timeout: REQUEST_TIMEOUT_MS,
        maxBodyLength: Infinity,
        maxContentLength: Infinity,
        validateStatus: (status) => status >= 200 && status < 300,
      }
    );
  } catch (error) {
    const { status, details } = getErrorDetails(error);

    log("NVIDIA request failed.", { status, details });

    const clientStatus =
      status && status >= 400 && status < 500 ? status : 502;

    return res.status(clientStatus).json({
      error: {
        message: "NVIDIA NIM request failed.",
        type: "upstream_error",
        upstream_status: status || null,
        details,
      },
    });
  }

  if (!stream) {
    const data = upstream.data;
    const choice = data?.choices?.[0];

    if (!choice || !choice.message) {
      log("NVIDIA returned no completion choices.");

      return res.status(502).json({
        error: {
          message: "NVIDIA returned an empty completion.",
          type: "upstream_error",
        },
      });
    }

    delete choice.message.reasoning_content;
    data.model = model;

    if (
      typeof choice.message.content === "string" &&
      choice.message.content.length > MAX_RESPONSE_CHARS
    ) {
      choice.message.content =
        choice.message.content.slice(0, MAX_RESPONSE_CHARS);
    }

    return res.status(200).json(data);
  }

  res.status(200);
  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");
  res.flushHeaders?.();

  const upstreamStream = upstream.data;
  const decoder = new StringDecoder("utf8");
  const completionId = makeId();
  const created = Math.floor(Date.now() / 1000);

  let buffer = "";
  let ended = false;
  let upstreamCancelled = false;
  let visibleChars = 0;

  function sendChunk(delta, finishReason = null) {
    if (res.destroyed || res.writableEnded) return;

    const chunk = {
      id: completionId,
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

  function finish() {
    if (ended) return;
    ended = true;

    if (!res.destroyed && !res.writableEnded) {
      res.write("data: [DONE]\n\n");
      res.end();
    }
  }

  function cancelUpstream() {
    if (upstreamCancelled) return;
    upstreamCancelled = true;

    if (upstreamStream && !upstreamStream.destroyed) {
      upstreamStream.destroy();
    }
  }

  function processEvent(eventText) {
    if (ended) return;

    const parsed = parseSSE(eventText);
    if (!parsed) return;

    if (parsed.done) {
      finish();
      return;
    }

    const choice = parsed.json?.choices?.[0];
    if (!choice) return;

    const delta = choice.delta || {};

    if (typeof delta.content === "string" && delta.content.length > 0) {
      const remaining = MAX_RESPONSE_CHARS - visibleChars;

      if (remaining > 0) {
        const content = delta.content.slice(0, remaining);
        visibleChars += content.length;
        sendChunk({ content });
      }

      if (visibleChars >= MAX_RESPONSE_CHARS) {
        log("Reached response character limit.");
        finish();
        cancelUpstream();
        return;
      }
    }

    if (choice.finish_reason != null) {
      sendChunk({}, choice.finish_reason);
    }
  }

  function processBuffer(final = false) {
    let match;

    while ((match = /\r?\n\r?\n/.exec(buffer)) !== null) {
      const eventText = buffer.slice(0, match.index);
      buffer = buffer.slice(match.index + match[0].length);

      processEvent(eventText);

      if (ended) {
        buffer = "";
        return;
      }
    }

    if (final && buffer.trim()) {
      processEvent(buffer);
      buffer = "";
    }
  }

  res.on("close", () => {
    if (!res.writableEnded && !ended) {
      log("Client disconnected; cancelling NVIDIA stream.");
      cancelUpstream();
    }
  });

  upstreamStream.on("data", (chunk) => {
    if (ended || res.destroyed) return;

    buffer += decoder.write(chunk);
    processBuffer();
  });

  upstreamStream.on("end", () => {
    if (ended) return;

    buffer += decoder.end();
    processBuffer(true);

    if (!ended) finish();

    log("NVIDIA stream ended.");
  });

  upstreamStream.on("error", (error) => {
    if (ended || res.destroyed) return;

    log("NVIDIA stream error.", error.message);

    res.write(
      `event: error\ndata: ${JSON.stringify({
        error: {
          message: "NVIDIA streaming connection failed.",
          type: "upstream_error",
        },
      })}\n\n`
    );

    ended = true;
    res.end();
  });
});

app.use((req, res) => {
  res.status(404).json({
    error: {
      message: `Route not found: ${req.method} ${req.path}`,
      type: "not_found",
    },
  });
});

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
  log(`Model: ${DEFAULT_MODEL}`);
  log(`Reasoning effort: ${REASONING_EFFORT}`);
  log(`NVIDIA API key configured: ${Boolean(NVIDIA_API_KEY)}`);
});
