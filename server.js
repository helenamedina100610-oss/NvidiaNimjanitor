```javascript
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();

const PORT = process.env.PORT || 3000;

const NVIDIA_API_BASE = (
  process.env.NVIDIA_API_BASE ||
  'https://integrate.api.nvidia.com/v1'
).replace(/\/+$/, '');

const NVIDIA_API_KEY = process.env.NVIDIA_API_KEY || '';

const DEFAULT_MODEL =
  process.env.NVIDIA_MODEL || 'moonshotai/kimi-k3';

const SHOW_REASONING =
  String(process.env.SHOW_REASONING || 'false').toLowerCase() === 'true';

const REASONING_EFFORT =
  process.env.REASONING_EFFORT || 'max';


// ============================================================
// MIDDLEWARE
// ============================================================

app.use(cors());

app.use(
  express.json({
    limit: '5mb'
  })
);

app.use(
  express.urlencoded({
    extended: true,
    limit: '5mb'
  })
);


// ============================================================
// REQUEST LOGGER
// ============================================================

app.use((req, res, next) => {
  console.log(
    `[REQUEST] ${new Date().toISOString()} ${req.method} ${req.path}`
  );

  next();
});


// ============================================================
// MODEL MAPPING
// ============================================================

const MODEL_MAPPING = {
  'gpt-4': DEFAULT_MODEL,
  'gpt-4o': DEFAULT_MODEL,
  'gpt-4o-mini': DEFAULT_MODEL,
  'gpt-4.1': DEFAULT_MODEL,
  'gpt-4.1-mini': DEFAULT_MODEL,
  'gpt-5': DEFAULT_MODEL,
  'gpt-5-mini': DEFAULT_MODEL,
};


// ============================================================
// MODEL RESOLVER
// ============================================================

function resolveModel(requestedModel) {
  if (!requestedModel) {
    return DEFAULT_MODEL;
  }

  const model = String(requestedModel).trim();

  if (!model) {
    return DEFAULT_MODEL;
  }

  if (model.startsWith('nvidia/')) {
    return model;
  }

  if (model.includes('/')) {
    return model;
  }

  return MODEL_MAPPING[model] || DEFAULT_MODEL;
}


// ============================================================
// MESSAGE NORMALIZER
// ============================================================

function normalizeMessages(messages) {
  if (!Array.isArray(messages)) {
    return [];
  }

  return messages.map((message) => {
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

    if (message.reasoning_content !== undefined) {
      normalized.reasoning_content =
        message.reasoning_content;
    }

    return normalized;
  });
}


// ============================================================
// TEXT EXTRACTION
// ============================================================

function extractText(content) {
  if (typeof content === 'string') {
    return content;
  }

  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === 'string') {
          return part;
        }

        if (
          part &&
          typeof part.text === 'string'
        ) {
          return part.text;
        }

        return '';
      })
      .join('');
  }

  return '';
}


// ============================================================
// SSE
// ============================================================

function sendSSE(res, payload) {
  res.write(
    'data: ' +
    JSON.stringify(payload) +
    '\n\n'
  );
}


// ============================================================
// CHAT CHUNK
// ============================================================

function makeChatChunk(
  id,
  model,
  content,
  finishReason = null
) {
  return {
    id,
    object: 'chat.completion.chunk',
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


// ============================================================
// ERROR MESSAGE
// ============================================================

function getNvidiaErrorMessage(error) {
  if (
    error &&
    error.response &&
    error.response.data
  ) {
    const data = error.response.data;

    if (typeof data === 'string') {
      return data;
    }

    if (
      data.error &&
      typeof data.error.message === 'string'
    ) {
      return data.error.message;
    }

    if (typeof data.detail === 'string') {
      return data.detail;
    }

    try {
      return JSON.stringify(data);
    } catch {
      return 'Unknown NVIDIA API error';
    }
  }

  if (
    error &&
    typeof error.message === 'string'
  ) {
    return error.message;
  }

  return 'Unknown NVIDIA API error';
}


// ============================================================
// ROOT
// ============================================================

app.get('/', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    model: DEFAULT_MODEL
  });
});


// ============================================================
// HEALTH
// ============================================================

function healthResponse(req, res) {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    model: DEFAULT_MODEL,
    reasoning_display: SHOW_REASONING,
    reasoning_effort: REASONING_EFFORT,
    nim_api_configured: Boolean(NVIDIA_API_KEY)
  });
}

app.get('/health', healthResponse);
app.get('/health/', healthResponse);


// ============================================================
// MODELS
// ============================================================

app.get('/v1/models', (req, res) => {
  const models = [
    'gpt-4',
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-4.1',
    'gpt-4.1-mini',
    'claude-3',
    'claude-3.5-sonnet',
    'gemini-pro',
    DEFAULT_MODEL
  ];

  const uniqueModels = [
    ...new Set(models)
  ];

  res.json({
    object: 'list',
    data: uniqueModels.map((model) => ({
      id: model,
      object: 'model',
      created: Math.floor(Date.now() / 1000),
      owned_by: 'nvidia-nim'
    }))
  });
});


// ============================================================
// CHAT COMPLETIONS
// ============================================================

app.post(
  '/v1/chat/completions',
  async (req, res) => {

    console.log(
      '[CHAT] ================================================'
    );

    console.log(
      '[CHAT] Received chat completion request.'
    );


    // --------------------------------------------------------
    // API KEY
    // --------------------------------------------------------

    if (!NVIDIA_API_KEY) {
      console.error(
        '[ERROR] NVIDIA_API_KEY is not configured.'
      );

      return res.status(500).json({
        error: {
          message:
            'NVIDIA_API_KEY is not configured.',
          type: 'server_error'
        }
      });
    }


    // --------------------------------------------------------
    // BODY
    // --------------------------------------------------------

    const body = req.body || {};

    console.log(
      '[CHAT] Client requested stream:',
      Boolean(body.stream)
    );

    console.log(
      '[CHAT] Client requested model:',
      body.model || '(none)'
    );

    console.log(
      '[CHAT] Number of messages:',
      Array.isArray(body.messages)
        ? body.messages.length
        : 0
    );


    if (
      !body.messages ||
      !Array.isArray(body.messages) ||
      body.messages.length === 0
    ) {
      return res.status(400).json({
        error: {
          message:
            'messages is required and must be a non-empty array.',
          type: 'invalid_request_error'
        }
      });
    }


    // --------------------------------------------------------
    // MODEL
    // --------------------------------------------------------

    const requestedModel =
      body.model || 'gpt-4o';

    const nimModel =
      resolveModel(requestedModel);

    console.log(
      '[MODEL] Requested:',
      requestedModel
    );

    console.log(
      '[MODEL] NVIDIA model:',
      nimModel
    );


    // --------------------------------------------------------
    // MESSAGES
    // --------------------------------------------------------

    const messages =
      normalizeMessages(body.messages);


    // --------------------------------------------------------
    // PARAMETERS
    // --------------------------------------------------------

    const temperature =
      typeof body.temperature === 'number'
        ? body.temperature
        : 1.0;

    const maxTokens =
      typeof body.max_tokens === 'number'
        ? body.max_tokens
        : typeof body.max_completion_tokens === 'number'
          ? body.max_completion_tokens
          : 16384;

    const stream =
      Boolean(body.stream);


    // --------------------------------------------------------
    // NVIDIA REQUEST
    // --------------------------------------------------------

    const nimRequest = {
      model: nimModel,
      messages,
      temperature,
      max
