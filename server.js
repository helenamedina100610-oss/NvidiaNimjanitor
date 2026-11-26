
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
  process.env.NVIDIA_MODEL || 'moonshotai/kimi-k3.1-pro';

const REASONING_EFFORT =
  process.env.REASONING_EFFORT || 'max';

const SHOW_REASONING =
  String(process.env.SHOW_REASONING || 'false').toLowerCase() === 'true';

const REQUEST_TIMEOUT = 180000;


// ============================================================
// MIDDLEWARE
// ============================================================

app.use(cors());

app.use(express.json({ limit: '10mb' }));

app.use(express.urlencoded({
  extended: true,
  limit: '10mb'
}));


// ============================================================
// LOGGER
// ============================================================

app.use((req, res, next) => {
  console.log(
    new Date().toISOString() +
    ' ' +
    req.method +
    ' ' +
    req.path
  );

  next();
});


// ============================================================
// MODEL MAPPING
// ============================================================

const MODEL_MAPPING = {
  'gpt-3.5-turbo': DEFAULT_MODEL,
  'gpt-3.5-turbo-16k': DEFAULT_MODEL,

  'gpt-4': DEFAULT_MODEL,
  'gpt-4-turbo': DEFAULT_MODEL,
  'gpt-4-turbo-preview': DEFAULT_MODEL,
  'gpt-4o': DEFAULT_MODEL,
  'gpt-4o-mini': DEFAULT_MODEL,
  'gpt-4.1': DEFAULT_MODEL,
  'gpt-4.1-mini': DEFAULT_MODEL,

  'claude-3-opus': DEFAULT_MODEL,
  'claude-3-sonnet': DEFAULT_MODEL,
  'claude-3.5-sonnet': DEFAULT_MODEL,
  'claude-3.7-sonnet': DEFAULT_MODEL,

  'gemini-pro': DEFAULT_MODEL,

  'kimi-k3': DEFAULT_MODEL,
  'kimi-k3.1-pro': DEFAULT_MODEL
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

// Preserva campos adicionais importantes para o Kimi,
// especialmente reasoning_content e tool_calls.

function normalizeMessages(messages) {
  if (!Array.isArray(messages)) {
    return [];
  }

  return messages.map((message) => ({
    ...message
  }));
}


// ============================================================
// ERROR HANDLER
// ============================================================

function getNvidiaErrorMessage(error) {
  if (error && error.response && error.response.data) {
    const data = error.response.data;

    if (typeof data === 'string') {
      return data;
    }

    if (data.error && data.error.message) {
      return data.error.message;
    }

    if (data.detail) {
      return data.detail;
    }

    try {
      return JSON.stringify(data);
    } catch {
      return 'Unknown NVIDIA API error';
    }
  }

  return error && error.message
    ? error.message
    : 'Unknown NVIDIA API error';
}


// ============================================================
// SSE OUTPUT
// ============================================================

function sendSSE(res, payload) {
  if (res.writableEnded || res.destroyed) {
    return;
  }

  res.write(
    'data: ' +
    JSON.stringify(payload) +
    '\n\n'
  );
}


// ============================================================
// SSE EVENT PROCESSOR
// ============================================================

function processSSEEvent(eventText, state, res) {
  const lines = eventText.split(/\r?\n/);

  const dataLines = lines
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice(5).trim());

  if (dataLines.length === 0) {
    return;
  }

  const rawData = dataLines.join('\n');

  if (!rawData) {
    return;
  }

  if (rawData === '[DONE]') {
    state.doneReceived = true;

    console.log('NVIDIA sent [DONE].');

    return;
  }

  let parsed;

  try {
    parsed = JSON.parse(rawData);
  } catch {
    console.log('Ignored invalid SSE event.');
    return;
  }

  if (!state.firstChunkReceived) {
    state.firstChunkReceived = true;

    console.log('First NVIDIA content event received.');
  }

  if (parsed.id) {
    state.completionId = parsed.id;
  }

  if (parsed.model) {
    state.model = parsed.model;
  }

  // Forward usage-only chunks when provided.

  if (
    (!Array.isArray(parsed.choices) ||
      parsed.choices.length === 0) &&
    parsed.usage
  ) {
    sendSSE(res, {
      id: state.completionId,
      object: 'chat.completion.chunk',
      created: state.created,
      model: state.model,
      choices: [],
      usage: parsed.usage
    });

    return;
  }

  if (!Array.isArray(parsed.choices)) {
    return;
  }

  for (const choice of parsed.choices) {
    const originalDelta = choice.delta || {};

    // Preserve reasoning_content as metadata.
    // It is not merged into the visible content.

    const delta = {
      ...originalDelta
    };

    const output = {
      id: state.completionId,
      object: 'chat.completion.chunk',
      created: state.created,
      model: state.model,
      choices: [
        {
          index: choice.index || 0,
          delta,
          finish_reason: choice.finish_reason ?? null
        }
      ]
    };

    sendSSE(res, output);

    if (choice.finish_reason) {
      console.log(
        'NVIDIA finish reason: ' +
        choice.finish_reason
      );
    }
  }
}


// ============================================================
// SSE STREAM READER
// ============================================================

async function forwardNvidiaStream(response, res, model) {
  const state = {
    completionId: 'chatcmpl-' + Date.now(),
    created: Math.floor(Date.now() / 1000),
    model,
    firstChunkReceived: false,
    doneReceived: false
  };

  let buffer = '';

  res.status(200);

  res.setHeader(
    'Content-Type',
    'text/event-stream; charset=utf-8'
  );

  res.setHeader(
    'Cache-Control',
    'no-cache, no-transform'
  );

  res.setHeader(
    'Connection',
    'keep-alive'
  );

  res.setHeader(
    'X-Accel-Buffering',
    'no'
  );

  res.flushHeaders();

  // Destroy upstream connection only if Janitor disconnects early.

  res.on('close', () => {
    if (
      !res.writableEnded &&
      response.data &&
      typeof response.data.destroy === 'function'
    ) {
      console.log('Janitor disconnected. Closing NVIDIA stream.');

      response.data.destroy();
    }
  });

  try {
    for await (const chunk of response.data) {
      if (res.destroyed || res.writableEnded) {
        break;
      }

      buffer += chunk.toString('utf8');

      const events = buffer.split(/\r?\n\r?\n/);

      buffer = events.pop() || '';

      for (const event of events) {
        processSSEEvent(event, state, res);
      }
    }

    // Process a final event that may not end with a blank line.

    if (buffer.trim() && !res.destroyed && !res.writableEnded) {
      processSSEEvent(buffer, state, res);
    }

    if (!res.destroyed && !res.writableEnded) {
      if (!state.firstChunkReceived) {
        console.warn('NVIDIA stream ended without content events.');
      }

      if (!state.doneReceived) {
        console.log('Sending final [DONE] to Janitor.');
      }

      res.write('data: [DONE]\n\n');
      res.end();
    }

    console.log('NVIDIA stream forwarding completed.');

  } catch (error) {
    console.error(
      'NVIDIA stream forwarding error:',
      getNvidiaErrorMessage(error)
    );

    if (!res.headersSent) {
      res.status(502).json({
        error: {
          message: getNvidiaErrorMessage(error),
          type: 'upstream_error'
        }
      });
    } else if (!res.writableEnded) {
      res.end();
    }
  }
}


// ============================================================
// NVIDIA REQUEST BUILDER
// ============================================================

function buildNvidiaRequest(body, model) {
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

  const request = {
    model,
    messages: normalizeMessages(body.messages),
    temperature,
    max_tokens: maxTokens,
    stream: Boolean(body.stream),
    reasoning_effort: REASONING_EFFORT
  };

  // Optional supported parameters.

  if (body.seed !== undefined) {
    request.seed = body.seed;
  }

  if (body.stream_options !== undefined) {
    request.stream_options = body.stream_options;
  }

  if (body.tools !== undefined) {
    request.tools = body.tools;
  }

  if (body.tool_choice !== undefined) {
    request.tool_choice = body.tool_choice;
  }

  if (body.stop !== undefined) {
    request.stop = body.stop;
  }

  return request;
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

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    model: DEFAULT_MODEL,
    reasoning_display: SHOW_REASONING,
    reasoning_effort: REASONING_EFFORT,
    nim_api_configured: Boolean(NVIDIA_API_KEY)
  });
});

app.get('/health/', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    model: DEFAULT_MODEL,
    reasoning_display: SHOW_REASONING,
    reasoning_effort: REASONING_EFFORT,
    nim_api_configured: Boolean(NVIDIA_API_KEY)
  });
});


// ============================================================
// MODELS
// ============================================================

app.get('/v1/models', (req, res) => {
  const models = [
    'gpt-3.5-turbo',
    'gpt-4',
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-4.1',
    'gpt-4.1-mini',
    'claude-3-opus',
    'claude-3.5-sonnet',
    'gemini-pro',
    DEFAULT_MODEL
  ];

  res.json({
    object: 'list',
    data: [...new Set(models)].map((model) => ({
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

app.post('/v1/chat/completions', async (req, res) => {
  console.log('Received chat completion request.');

  if (!NVIDIA_API_KEY) {
    return res.status(500).json({
      error: {
        message: 'NVIDIA_API_KEY is not configured.',
        type: 'server_error'
      }
    });
  }

  const body = req.body || {};

  if (
    !Array.isArray(body.messages) ||
    body.messages.length === 0
  ) {
    return res.status(400).json({
      error: {
        message: 'messages must be a non-empty array.',
        type: 'invalid_request_error'
      }
    });
  }

  const requestedModel = body.model || 'gpt-4o';

  const nimModel = resolveModel(requestedModel);

  const nimRequest = buildNvidiaRequest(body, nimModel);

  console.log('==================================================');
  console.log('NVIDIA REQUEST START');
  console.log('Requested model: ' + requestedModel);
  console.log('NVIDIA model: ' + nimModel);
  console.log('Stream: ' + nimRequest.stream);
  console.log('Temperature: ' + nimRequest.temperature);
  console.log('Max tokens: ' + nimRequest.max_tokens);
  console.log('Reasoning effort: ' + REASONING_EFFORT);
  console.log('Sending request to NVIDIA.');
  console.log('==================================================');

  try {
    const response = await axios({
      method: 'POST',
      url: NVIDIA_API_BASE + '/chat/completions',

      headers: {
        Authorization: 'Bearer ' + NVIDIA_API_KEY,
        'Content-Type': 'application/json',
        Accept: nimRequest.stream
          ? 'text/event-stream'
          : 'application/json'
      },

      data: nimRequest,

      responseType: nimRequest.stream ? 'stream' : 'json',

      timeout: REQUEST_TIMEOUT,

      validateStatus: () => true
    });

    console.log('NVIDIA HTTP status: ' + response.status);

    if (response.status < 200 || response.status >= 300) {
      let errorData = '';

      if (response.data && typeof response.data[Symbol.asyncIterator] === 'function') {
        for await (const chunk of response.data) {
          errorData += chunk.toString('utf8');
        }
      } else {
        errorData = JSON.stringify(response.data);
      }

      console.error('NVIDIA API error: ' + errorData);

      return res.status(response.status).json({
        error: {
          message: errorData || 'NVIDIA API error',
          type: 'upstream_error'
        }
      });
    }

    // --------------------------------------------------------
    // STREAMING RESPONSE
    // --------------------------------------------------------

    if (nimRequest.stream) {
      console.log('NVIDIA streaming connection opened.');

      return await forwardNvidiaStream(
        response,
        res,
        nimModel
      );
    }

    // --------------------------------------------------------
    // NON-STREAMING RESPONSE
    // --------------------------------------------------------

    console.log('NVIDIA non-streaming response received.');

    const data = response.data;

    // Keep reasoning_content in the returned assistant message.
    // Kimi needs it preserved for multi-turn conversations.

    return res.status(200).json(data);

  } catch (error) {
    console.error('NVIDIA request failed.');
    console.error(getNvidiaErrorMessage(error));

    if (!res.headersSent) {
      return res.status(502).json({
        error: {
          message: getNvidiaErrorMessage(error),
          type: 'upstream_error'
        }
      });
    }

    if (!res.writableEnded) {
      res.end();
    }
  }
});


// ============================================================
// 404
// ============================================================

app.use((req, res) => {
  res.status(404).json({
    error: {
      message: 'Endpoint not found.',
      type: 'not_found'
    }
  });
});


// ============================================================
// START SERVER
// ============================================================

app.listen(PORT, '0.0.0.0', () => {
  console.log('==================================================');
  console.log('OpenAI to NVIDIA NIM Proxy');
  console.log('==================================================');

  console.log('Server listening on port ' + PORT);
  console.log('NVIDIA API: ' + NVIDIA_API_BASE);
  console.log('Default model: ' + DEFAULT_MODEL);
  console.log('Temperature default: 1.0');
  console.log('Max tokens default: 16384');
  console.log('Reasoning effort: ' + REASONING_EFFORT);
  console.log('Reasoning display setting: ' + SHOW_REASONING);
  console.log('NVIDIA API configured: ' + Boolean(NVIDIA_API_KEY));

  console.log('==================================================');
});
