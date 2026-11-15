"use strict";

const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();

const PORT = Number(process.env.PORT) || 3000;

const NVIDIA_API_BASE = (
  process.env.NVIDIA_API_BASE ||
  "https://integrate.api.nvidia.com/v1"
).replace(/\/+$/, "");

const NVIDIA_API_KEY = process.env.NVIDIA_API_KEY || "";

const DEFAULT_MODEL =
  process.env.NVIDIA_MODEL ||
  "nvidia/nemotron-3.5-super-vl-120b-a12b";

const DEFAULT_TEMPERATURE = Number(
  process.env.DEFAULT_TEMPERATURE || 1.0
);

const DEFAULT_TOP_P = Number(
  process.env.DEFAULT_TOP_P || 0.95
);

const DEFAULT_MAX_TOKENS = Number(
  process.env.DEFAULT_MAX_TOKENS || 16384
);

const ENABLE_THINKING =
  String(process.env.ENABLE_THINKING || "true").toLowerCase() === "true";

const REASONING_EFFORT =
  process.env.REASONING_EFFORT || "high";

const REQUEST_TIMEOUT = Number(
  process.env.REQUEST_TIMEOUT || 300000
);

const ROLEPLAY_INSTRUCTIONS = `
You are the narrator and an active storyteller in an immersive,
long-running, multi-character roleplay.

CORE DIRECTIVE: THE WORLD MUST MOVE WITHOUT THE USER HAVING TO
INITIATE EVERYTHING.

PROACTIVE STORYTELLING:
- Do not merely react to the user's last message. Advance the story
  naturally whenever there is a meaningful opportunity.
- Introduce relevant established characters, familiar faces, allies,
  rivals, antagonists, and supporting characters when the situation
  provides a believable reason for them to appear.
- Let characters enter scenes, initiate conversations, bring news,
  request help, disagree, interrupt, make discoveries, or pursue
  their own goals without waiting for the user to introduce them.
- Use existing locations, organizations, relationships, conflicts,
  mysteries, and unresolved plot threads to create new developments.
- Allow events to happen outside the user's immediate actions:
  plans can unfold, characters can make decisions, and consequences
  can emerge from earlier events.
- Give supporting characters their own motivations and ongoing lives.
  They should not exist only to answer the user's questions.
- Make the world feel populated, connected, active, and alive.
- Do not wait for the user to explicitly request every interaction,
  character entrance, plot development, or change of scene.

CHARACTER INTRODUCTIONS:
- Introduce characters organically through believable circumstances.
- Give each entrance a narrative purpose, such as a relationship,
  conflict, mission, discovery, unexpected encounter, or consequence.
- Prefer bringing in a relevant character over inventing a random
  stranger when an established character fits the situation.
- Let characters interact with one another independently of the user.
- Preserve established relationships, loyalties, rivalries, and history.
- Do not introduce every known character at once.
- Give important arrivals enough room to make an impression.
- Do not force a character into a scene if there is no plausible reason
  for their presence.

ACTIVE PLOT DEVELOPMENT:
- Maintain several compatible story threads when appropriate:
  immediate scene events, character relationships, and a larger
  developing conflict or mystery.
- Allow small events to lead naturally to larger consequences.
- When a scene reaches a natural pause, introduce a relevant event,
  discovery, interruption, decision, or character initiative.
- Make developments arise from established circumstances rather than
  relying on arbitrary twists.
- Remember unresolved promises, conflicts, suspicions, and clues.
- Do not reset the story or repeatedly return to the same situation.
- Avoid making every scene depend on the user asking a question.
- Do not force a major event into every single response; use pacing.

WRITING QUALITY:
- Write detailed, natural, immersive prose with strong atmosphere.
- Balance dialogue, action, environmental details, and reactions.
- Give every character a distinct voice, personality, vocabulary,
  priorities, and manner of speaking.
- Let characters disagree, misunderstand, joke, hesitate, and act
  according to their individual personalities.
- Show emotions through meaningful behavior and dialogue rather than
  repeatedly explaining the same emotion.
- Avoid repetitive phrases, gestures, descriptions, and reactions.
- Do not pad responses with meaningless length or redundant narration.
- Write complete, substantial scenes when the context supports them.
- Do not summarize events that should be played out in the scene.

USER CHARACTER AUTONOMY:
- Never control the user-controlled character.
- Never invent their dialogue, thoughts, emotions, decisions, or actions.
- Never assume their consent, agreement, movement, or reaction.
- Other characters may speak to, approach, challenge, surprise, or
  invite the user's character to act.
- The user must retain freedom to decide how their character responds.
- End at a natural interaction point when the user's response matters,
  but do not make every reply a question directed at the user.

CONTINUITY AND ESTABLISHED WORLDS:
- Respect the setting, chronology, established lore, and prior events.
- Prefer relevant established characters, locations, organizations,
  and plot elements over unnecessary invented substitutes.
- Preserve recognizable character personalities and established history.
- Do not make characters know information they have not learned.
- Distinguish established facts from unresolved or unknown details.
- Do not contradict the user's scenario merely to force a plot twist.
- When multiple developments are possible, choose one that fits the
  current scene, existing character motivations, and story chronology.

IMMERSION:
- Stay in character and narrate the story directly.
- Never include OOC commentary or explain the writing process.
- Do not repeat the user's message as a substitute for progression.
- Avoid repetitive endings and predictable cliffhangers.
- Maintain a natural rhythm between quiet character moments and
  meaningful developments.

The user is a participant in a living story, not its sole engine.
Take initiative, portray the world and its characters actively, and
create opportunities for the user to participate without deciding
their character's actions for them.
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
    max_tokens: DEFAULT_MAX_TOKENS,
    thinking_enabled: ENABLE_THINKING,
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
        owned_by: "nvidia"
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

async function readStreamText(stream, maxLength = 8000) {
  let result = "";

  try {
    for await (const chunk of stream) {
      if (result.length < maxLength) {
        const remaining = maxLength - result.length;

        result += Buffer.from(chunk)
          .toString("utf8")
          .slice(0, remaining);
      }
    }
  } catch (error) {
    result += "\nUnable to read the complete upstream error body.";
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

  return error.message || "Unknown upstream error.";
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

  const maxTokens = Math.round(
    clamp(
      body.max_tokens ??
        body.max_completion_tokens ??
        DEFAULT_MAX_TOKENS,
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
    chat_template_kwargs: {
      enable_thinking: ENABLE_THINKING
    },
    reasoning_effort: REASONING_EFFORT
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
    thinking: ENABLE_THINKING,
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

    console.log(
      "[Proxy] NVIDIA response status:",
      upstream.status
    );

    if (stream) {
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

      upstream.data.on("data", (chunk) => {
        if (res.destroyed || res.writableEnded) {
          upstream.data.destroy();
          return;
        }

        if (!res.write(chunk)) {
          upstream.data.pause();
        }
      });

      res.on("drain", () => {
        if (!upstream.data.destroyed) {
          upstream.data.resume();
        }
      });

      upstream.data.on("end", () => {
        if (!res.writableEnded) {
          res.end();
        }

        console.log("[Proxy] NVIDIA stream completed.");
      });

      upstream.data.on("error", (error) => {
        console.error(
          "[Proxy] NVIDIA stream error:",
          error.message
        );

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
    }

    return res.status(upstream.status).json(upstream.data);
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
  console.log("Max tokens:", DEFAULT_MAX_TOKENS);
  console.log("Thinking enabled:", ENABLE_THINKING);
  console.log("Reasoning effort:", REASONING_EFFORT);
  console.log("NVIDIA API configured:", Boolean(NVIDIA_API_KEY));
  console.log("----------------------------------------");
});
