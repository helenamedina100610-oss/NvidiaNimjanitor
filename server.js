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

const DEFAULT_TEMPERATURE = 1.0;
const DEFAULT_MAX_TOKENS = 8192;
const MAX_RESPONSE_CHARS = 19000;

const REASONING_EFFORT =
  process.env.REASONING_EFFORT || "low";

const SHOW_REASONING =
  String(process.env.SHOW_REASONING || "false").toLowerCase() === "true";

const ENABLE_THINKING =
  String(process.env.ENABLE_THINKING || "true").toLowerCase() === "true";

const ROLEPLAY_BOUNDARY_INSTRUCTION = `
IMPORTANT ROLEPLAY RULE:
The user controls their own character completely.
Never speak, act, think, feel, decide, or choose actions for the user's character.
Never write dialogue, thoughts, emotions, intentions, movements, or decisions on behalf of the user's character.
Only control the characters and entities that you are responsible for.
Leave the user's character's actions, thoughts, feelings, dialogue, and decisions entirely to the user.
`.trim();

app.use(cors());

app.use(
  express.json({
    limit: "5mb"
  })
);

app.use((req, res, next) => {
  console.log(
    `${new Date().toISOString()} ${req.method} ${req.path}`
  );
  next();
});

function resolveModel(model) {
  if (!model) {
    return DEFAULT_MODEL;
  }

  if (model.startsWith("nvidia/")) {
    return model;
  }

  if (model === "gpt-3.5-turbo") {
    return DEFAULT_MODEL;
  }

  if (model === "gpt-4") {
    return DEFAULT_MODEL;
  }

  if (model === "gpt-4o") {
    return DEFAULT_MODEL;
  }

  if (model === "gpt-4.1") {
    return DEFAULT_MODEL;
  }

  if (model === "claude-3") {
    return DEFAULT_MODEL;
  }

  if (model === "claude-3-opus") {
    return DEFAULT_MODEL;
  }

  if (model === "claude-3-sonnet") {
    return DEFAULT_MODEL;
  }

  if (model === "gemini-pro") {
    return DEFAULT_MODEL;
  }

  return DEFAULT_MODEL;
}

function prepareMessages(messages) {
  const originalMessages = Array.isArray(messages)
    ? messages
    : [];

  const result = [];
  let systemInserted = false;

  for (const message of originalMessages) {
    if (
      message &&
      (message.role === "system" ||
        message.role === "developer")
    ) {
      result.push(message);
      continue;
    }

    if (!systemInserted) {
      result.push({
        role: "system",
        content: ROLEPLAY_BOUNDARY_INSTRUCTION
      });

      systemInserted = true;
    }

    result.push(message);
  }

  if (!systemInserted) {
    result.push({
      role: "system",
      content: ROLEPLAY_BOUNDARY_INSTRUCTION
    });
  }

  return result;
}

function extractTextFromDelta(delta) {
  if (!delta) {
    return "";
  }

  if (typeof delta.content === "string") {
    return delta.content;
  }

  if (Array.isArray(delta.content)) {
    return delta.content
      .map((item) => {
        if (typeof item === "string") {
          return item;
        }

        if (item && typeof item.text === "string") {
          return item.text;
        }

        return "";
      })
      .join("");
  }

  return "";
}

function sendSSE(res, data) {
  res.write(`data: ${JSON.stringify(data)}\n\n`);
}

function sendDone(res) {
  res.write("data: [DONE]\n\n");
}

function createOpenAIChunk(model, content, finishReason = null) {
  return {
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
}

function truncateText(text, maxChars) {
  if (typeof text !== "string") {
    return "";
  }

  if (text.length <= maxChars) {
    return text;
  }

  return text.slice(0, maxChars);
}

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

app.get("/v1/models", (req, res) => {
  res.json({
    object: "list",
    data: [
      {
        id: "gpt-4o",
        object: "model",
        owned_by: "nvidia-nim"
      },
      {
        id: DEFAULT_MODEL,
        object: "model",
        owned_by: "nvidia-nim"
      }
    ]
  });
});

app.get("/", (req, res) => {
  res.json({
    status: "ok",
    service: "OpenAI to NVIDIA NIM Proxy",
    model: DEFAULT_MODEL
  });
});

app.post("/v1/chat/completions", async (req, res) => {
  const requestStartedAt = Date.now();

  const body = req.body || {};

  const requestedModel = body.model || "gpt-4o";
  const model = resolveModel(requestedModel);

  const stream = body.stream === true;

  const temperature =
    typeof body.temperature === "number"
      ? body.temperature
      : DEFAULT_TEMPERATURE;

  const maxTokens =
    typeof body.max_tokens === "number"
      ? body.max_tokens
      : DEFAULT_MAX_TOKENS;

  const messages = prepareMessages(body.messages);

  console.log("==========================================");
  console.log("NVIDIA REQUEST START");
  console.log(`Model: ${model}`);
  console.log(`Temperature: ${temperature}`);
  console.log(`Max tokens: ${maxTokens}`);
  console.log(`Reasoning effort: ${REASONING_EFFORT}`);
  console.log(`Stream requested: ${stream}`);
  console.log(`Character limit: ${MAX_RESPONSE_CHARS}`);

  if (!NVIDIA_API_KEY) {
    console.log("ERROR: NVIDIA_API_KEY is not configured.");

    return res.status(500).json({
      error: {
        message: "NVIDIA_API_KEY is not configured.",
        type: "configuration_error"
      }
    });
  }

  const nimRequest = {
    model,
    messages,
    temperature,
    max_tokens: maxTokens,
    stream
  };

  if (REASONING_EFFORT) {
    nimRequest.reasoning_effort = REASONING_EFFORT;
  }

  if (ENABLE_THINKING !== undefined) {
    nimRequest.chat_template_kwargs = {
      enable_thinking: ENABLE_THINKING
    };
  }

  const url = `${NVIDIA_API_BASE}/chat/completions`;

  try {
    if (!stream) {
      const response = await axios.post(
        url,
        nimRequest,
        {
          headers: {
            Authorization: `Bearer ${NVIDIA_API_KEY}`,
            "Content-Type": "application/json",
            Accept: "application/json"
          },
          timeout: 180000
        }
      );

      let content = "";

      if (
        response.data &&
        response.data.choices &&
        response.data.choices[0]
      ) {
        const choice = response.data.choices[0];

        if (choice.message) {
          content = choice.message.content || "";
        }
      }

      content = truncateText(
        content,
        MAX_RESPONSE_CHARS
      );

      const totalTime = Date.now() - requestStartedAt;

      console.log(`TIME TOTAL: ${totalTime} ms`);
      console.log(
        `TIME NVIDIA: ${totalTime} ms`
      );
      console.log(
        `Characters sent: ${content.length}`
      );
      console.log(
        `Stream finished: ${
          response.data?.choices?.[0]?.finish_reason || "stop"
        }`
      );

      console.log("==========================================");

      return res.json({
        ...response.data,
        choices: response.data.choices.map((choice) => ({
          ...choice,
          message: {
            ...choice.message,
            content
          }
        }))
      });
    }

    console.log("NVIDIA streaming connection opened.");

    const nvidiaStartedAt = Date.now();

    const response = await axios.post(
      url,
      nimRequest,
      {
        headers: {
          Authorization: `Bearer ${NVIDIA_API_KEY}`,
          "Content-Type": "application/json",
          Accept: "text/event-stream"
        },
        responseType: "stream",
        timeout: 180000
      }
    );

    console.log(`NVIDIA status: ${response.status}`);

    res.status(200);

    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Accel-Buffering", "no");

    if (res.flushHeaders) {
      res.flushHeaders();
    }

    let accumulatedText = "";
    let firstTokenReceived = false;
    let firstTokenTime = null;
    let finished = false;

    const sendFinish = (finishReason = "stop") => {
      if (finished) {
        return;
      }

      finished = true;

      sendSSE(
        res,
        createOpenAIChunk(
          model,
          "",
          finishReason
        )
      );

      sendDone(res);

      const totalTime =
        Date.now() - requestStartedAt;

      const nvidiaTime =
        Date.now() - nvidiaStartedAt;

      console.log(
        `TIME TOTAL: ${totalTime} ms`
      );

      console.log(
        `TIME NVIDIA: ${nvidiaTime} ms`
      );

      console.log(
        `Characters sent: ${accumulatedText.length}`
      );

      console.log(
        `Stream finished: ${finishReason}`
      );

      console.log("==========================================");

      res.end();
    };

    let buffer = "";

    response.data.on("data", (chunk) => {
      if (finished) {
        return;
      }

      buffer += chunk.toString("utf8");

      const lines = buffer.split("\n");

      buffer = lines.pop() || "";

      for (const rawLine of lines) {
        if (finished) {
          break;
        }

        const line = rawLine.trim();

        if (!line) {
          continue;
        }

        if (!line.startsWith("data:")) {
          continue;
        }

        const data = line.slice(5).trim();

        if (!data) {
          continue;
        }

        if (data === "[DONE]") {
          sendFinish("stop");
          continue;
        }

        let parsed;

        try {
          parsed = JSON.parse(data);
        } catch (error) {
          continue;
        }

        if (
          parsed &&
          parsed.error
        ) {
          console.log(
            "NVIDIA stream error:",
            JSON.stringify(parsed.error)
          );

          sendFinish("stop");
          continue;
        }

        const choice =
          parsed?.choices?.[0];

        if (!choice) {
          continue;
        }

        const delta = choice.delta || {};

        let content =
          extractTextFromDelta(delta);

        if (
          !content &&
          choice.text
        ) {
          content = choice.text;
        }

        if (content) {
          if (!firstTokenReceived) {
            firstTokenReceived = true;

            firstTokenTime =
              Date.now() - nvidiaStartedAt;

            console.log(
              `FIRST TOKEN RECEIVED: ${firstTokenTime} ms`
            );

            console.log(
              `FIRST TOKEN TOTAL TIME: ${
                Date.now() - requestStartedAt
              } ms`
            );
          }

          const remaining =
            MAX_RESPONSE_CHARS -
            accumulatedText.length;

          if (remaining > 0) {
            const allowedContent =
              content.slice(0, remaining);

            accumulatedText +=
              allowedContent;

            if (allowedContent) {
              sendSSE(
                res,
                createOpenAIChunk(
                  model,
                  allowedContent,
                  null
                )
              );
            }
          }

          if (
            accumulatedText.length >=
            MAX_RESPONSE_CHARS
          ) {
            sendFinish("length");
            continue;
          }
        }

        if (
          choice.finish_reason
        ) {
          sendFinish(
            choice.finish_reason
          );
        }
      }
    });

    response.data.on("end", () => {
      if (!finished) {
        sendFinish("stop");
      }
    });

    response.data.on("error", (error) => {
      console.log(
        "NVIDIA streaming error:",
        error.message
      );

      if (!finished) {
        sendFinish("stop");
      }
    });

    res.on("close", () => {
      if (
        !finished &&
        !res.writableEnded
      ) {
        try {
          response.data.destroy();
        } catch (error) {
          // Ignore stream cleanup errors.
        }
      }
    });
  } catch (error) {
    const totalTime =
      Date.now() - requestStartedAt;

    console.log(
      `TIME TOTAL: ${totalTime} ms`
    );

    console.log(
      "NVIDIA REQUEST ERROR"
    );

    if (error.response) {
      console.log(
        "NVIDIA status:",
        error.response.status
      );

      if (error.response.data) {
        console.log(
          "NVIDIA error data:",
          typeof error.response.data === "string"
            ? error.response.data
            : JSON.stringify(error.response.data)
        );
      }
    } else {
      console.log(
        "Error:",
        error.message
      );
    }

    console.log("==========================================");

    if (!res.headersSent) {
      return res.status(
        error.response?.status || 500
      ).json({
        error: {
          message:
            error.response?.data?.error?.message ||
            error.message ||
            "Upstream NVIDIA error",
          type: "upstream_error"
        }
      });
    }

    try {
      res.end();
    } catch (endError) {
      // Ignore response close errors.
    }
  }
});

app.use((req, res) => {
  res.status(404).json({
    error: {
      message: "Not found",
      type: "invalid_request_error"
    }
  });
});

app.listen(PORT, () => {
  console.log("==========================================");
  console.log("OpenAI to NVIDIA NIM Proxy");
  console.log(`Server listening on port ${PORT}`);
  console.log(`Default model: ${DEFAULT_MODEL}`);
  console.log(`Max tokens: ${DEFAULT_MAX_TOKENS}`);
  console.log(`Temperature: ${DEFAULT_TEMPERATURE}`);
  console.log(`Reasoning effort: ${REASONING_EFFORT}`);
  console.log(`Show reasoning: ${SHOW_REASONING}`);
  console.log(`Enable thinking: ${ENABLE_THINKING}`);
  console.log(`Character limit: ${MAX_RESPONSE_CHARS}`);
  console.log("==========================================");
});
