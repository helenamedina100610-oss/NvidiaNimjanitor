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

app.use(cors());

app.use(express.json({
  limit: '5mb'
}));

app.use(express.urlencoded({
  extended: true,
  limit: '5mb'
}));

app.use(function (req, res, next) {
  console.log(
    new Date().toISOString(),
    req.method,
    req.path
  );
  next();
});

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
  'nemotron-3-ultra': DEFAULT_MODEL,
  [DEFAULT_MODEL]: DEFAULT_MODEL
};

function resolveModel(model) {
  if (!model) {
    return DEFAULT_MODEL;
  }

  if (model.startsWith('nvidia/')) {
    return model;
  }

  return MODEL_MAPPING[model] || DEFAULT_MODEL;
}

function normalizeMessages(messages) {
  if (!Array.isArray(messages)) {
    return [];
  }

  return messages.map(function (message) {
    return {
      role: message.role || 'user',
      content: message.content == null
        ? ''
        : message.content
    };
  });
}

function extractText(content) {
  if (content == null) {
    return '';
  }

  if (typeof content === 'string') {
    return content;
  }

  if (Array.isArray(content)) {
    return content.map(function (part) {
      if (typeof part === 'string') {
        return part;
      }

      if (part && typeof part.text === 'string') {
        return part.text;
      }

      return '';
    }).join('');
  }

  if (typeof content === 'object') {
    if (typeof content.text === 'string') {
      return content.text;
    }

    return '';
  }

  return String(content);
}

function getNvidiaErrorMessage(data) {
  if (!data) {
    return 'Unknown NVIDIA error';
  }

  if (typeof data === 'string') {
    return data;
  }

  if (data.error) {
    if (typeof data.error === 'string') {
      return data.error;
    }

    if (data.error.message) {
      return data.error.message;
    }
  }

  if (data.message) {
    return data.message;
  }

  try {
    return JSON.stringify(data);
  } catch (error) {
    return 'Unknown NVIDIA error';
  }
}

app.get('/', function (req, res) {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    model: DEFAULT_MODEL
  });
});

app.get('/health', function (req, res) {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    model: DEFAULT_MODEL,
    reasoning_display: SHOW_REASONING,
    reasoning_effort: REASONING_EFFORT,
    nim_api_configured: Boolean(NVIDIA_API_KEY)
  });
});

app.get('/health/', function (req, res) {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    model: DEFAULT_MODEL,
    reasoning_display: SHOW_REASONING,
    reasoning_effort: REASONING_EFFORT,
    nim_api_configured: Boolean(NVIDIA_API_KEY)
  });
});

app.get('/v1/models', function (req, res) {
  const aliases = Object.keys(MODEL_MAPPING);

  const data = aliases.map(function (id) {
    return {
      id: id,
      object: 'model',
      created: Math.floor(Date.now() / 1000),
      owned_by: 'nvidia-nim'
    };
  });

  if (!aliases.includes(DEFAULT_MODEL)) {
    data.push({
      id: DEFAULT_MODEL,
      object: 'model',
      created: Math.floor(Date.now() / 1000),
      owned_by: 'nvidia-nim'
    });
  }

  res.json({
    object: 'list',
    data: data
  });
});

app.post('/v1/chat/completions', async function (req, res) {
  const requestStart = Date.now();

  console.log('==========================================');
  console.log('NVIDIA REQUEST START');
  console.log('Request timer started.');

  if (!NVIDIA_API_KEY) {
    return res.status(500).json({
      error: {
        message: 'NVIDIA_API_KEY is not configured',
        type: 'configuration_error'
      }
    });
  }

  const body = req.body || {};

  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    return res.status(400).json({
      error: {
        message: 'messages is required',
        type: 'invalid_request_error'
      }
    });
  }

  const requestedModel = body.model || 'gpt-4o';
  const nimModel = resolveModel(requestedModel);

  const messages = normalizeMessages(body.messages);

  const temperature =
    body.temperature !== undefined
      ? body.temperature
      : 1.0;

  const maxTokens =
    body.max_tokens !== undefined
      ? body.max_tokens
      : (
          body.max_completion_tokens !== undefined
            ? body.max_completion_tokens
            : 16384
        );

  const stream = Boolean(body.stream);

  console.log(
    'Model requested:',
    requestedModel,
    '-> NVIDIA:',
    nimModel
  );

  console.log(
    'Temperature:',
    temperature
  );

  console.log(
    'Max tokens:',
    maxTokens
  );

  console.log(
    'Stream requested:',
    stream
  );

  const nimRequest = {
    model: nimModel,
    messages: messages,
    temperature: temperature,
    max_tokens: maxTokens,
    stream: false,
    reasoning_effort: REASONING_EFFORT
  };

  if (body.seed !== undefined) {
    nimRequest.seed = body.seed;
  }

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

  try {
    console.log(
      'Sending request to NVIDIA:',
      NVIDIA_API_BASE + '/chat/completions'
    );

    const nvidiaRequestStart = Date.now();

    console.log(
      'NVIDIA timer started.'
    );

    const response = await axios({
      method: 'POST',
      url: NVIDIA_API_BASE + '/chat/completions',
      headers: {
        Authorization: 'Bearer ' + NVIDIA_API_KEY,
        'Content-Type': 'application/json',
        Accept: 'application/json'
      },
      data: nimRequest,
      timeout: 180000,
      validateStatus: function () {
        return true;
      }
    });

    const nvidiaRequestTime =
      Date.now() - nvidiaRequestStart;

    const totalRequestTime =
      Date.now() - requestStart;

    console.log(
      'NVIDIA response received.'
    );

    console.log(
      'NVIDIA status:',
      response.status
    );

    console.log(
      'TIME NVIDIA:',
      nvidiaRequestTime + ' ms',
      '(' + (nvidiaRequestTime / 1000).toFixed(2) + ' seconds)'
    );

    console.log(
      'TIME TOTAL:',
      totalRequestTime + ' ms',
      '(' + (totalRequestTime / 1000).toFixed(2) + ' seconds)'
    );

    if (response.status < 200 || response.status >= 300) {
      console.error(
        'NVIDIA ERROR:',
        response.status,
        getNvidiaErrorMessage(response.data)
      );

      return res.status(response.status).json({
        error: {
          message: getNvidiaErrorMessage(response.data),
          type: 'upstream_error',
          code: response.status
        }
      });
    }

    const data = response.data;

    if (
      !data ||
      !Array.isArray(data.choices) ||
      !data.choices[0]
    ) {
      console.error(
        'NVIDIA returned an invalid response.'
      );

      return res.status(502).json({
        error: {
          message: 'Invalid response from NVIDIA',
          type: 'upstream_error'
        }
      });
    }

    const choice = data.choices[0];
    const message = choice.message || {};

    let content = extractText(message.content);

    console.log(
      'NVIDIA content length:',
      content.length
    );

    if (!content && SHOW_REASONING) {
      content = extractText(
        message.reasoning_content
      );
    }

    if (!content) {
      console.error(
        'NVIDIA returned no visible content.'
      );

      return res.status(502).json({
        error: {
          message: 'NVIDIA returned no visible content',
          type: 'upstream_error'
        }
      });
    }

    const completionId =
      data.id ||
      'chatcmpl-' + Date.now();

    const model =
      data.model ||
      nimModel;

    const finishReason =
      choice.finish_reason ||
      'stop';

    if (!stream) {
      const result = {
        id: completionId,
        object: 'chat.completion',
        created:
          data.created ||
          Math.floor(Date.now() / 1000),
        model: model,
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: content
            },
            finish_reason: finishReason
          }
        ]
      };

      if (data.usage) {
        result.usage = data.usage;
      }

      const finalTime =
        Date.now() - requestStart;

      console.log(
        'Response ready for Janitor.'
      );

      console.log(
        'TIME FINAL:',
        finalTime + ' ms',
        '(' + (finalTime / 1000).toFixed(2) + ' seconds)'
      );

      console.log('==========================================');

      return res.json(result);
    }

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

    if (typeof res.flushHeaders === 'function') {
      res.flushHeaders();
    }

    function sendSSE(payload) {
      if (typeof payload === 'string') {
        res.write('data: ' + payload + '\n\n');
        return;
      }

      res.write(
        'data: ' +
        JSON.stringify(payload) +
        '\n\n'
      );
    }

    function makeChunk(
      contentValue,
      finishReasonValue,
      includeRole
    ) {
      const delta = {};

      if (includeRole) {
        delta.role = 'assistant';
      }

      if (contentValue) {
        delta.content = contentValue;
      }

      return {
        id: completionId,
        object: 'chat.completion.chunk',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: [
          {
            index: 0,
            delta: delta,
            finish_reason: finishReasonValue || null
          }
        ]
      };
    }

    sendSSE(
      makeChunk(
        '',
        null,
        true
      )
    );

    sendSSE(
      makeChunk(
        content,
        null,
        false
      )
    );

    sendSSE(
      makeChunk(
        '',
        finishReason,
        false
      )
    );

    sendSSE('[DONE]');

    const finalTime =
      Date.now() - requestStart;

    console.log(
      'Response ready for Janitor.'
    );

    console.log(
      'TIME FINAL:',
      finalTime + ' ms',
      '(' + (finalTime / 1000).toFixed(2) + ' seconds)'
    );

    console.log('==========================================');

    res.end();

  } catch (error) {
    const errorTime =
      Date.now() - requestStart;

    console.error(
      'PROXY ERROR:',
      error.message
    );

    console.error(
      'TIME UNTIL ERROR:',
      errorTime + ' ms',
      '(' + (errorTime / 1000).toFixed(2) + ' seconds)'
    );

    if (error.response) {
      console.error(
        'NVIDIA response:',
        error.response.status,
        error.response.data
      );
    }

    console.log('==========================================');

    if (!res.headersSent) {
      return res.status(502).json({
        error: {
          message:
            error.message ||
            'Upstream NVIDIA request failed',
          type: 'upstream_error'
        }
      });
    }

    try {
      res.end();
    } catch (endError) {
      console.error(
        'Error ending response:',
        endError.message
      );
    }
  }
});

app.use(function (req, res) {
  res.status(404).json({
    error: {
      message: 'Not found',
      type: 'not_found'
    }
  });
});

app.listen(PORT, function () {
  console.log('==========================================');
  console.log('OpenAI to NVIDIA NIM Proxy');
  console.log('==========================================');
  console.log(
    'Server listening on port ' + PORT
  );
  console.log(
    'NVIDIA API: ' + NVIDIA_API_BASE
  );
  console.log(
    'Default model: ' + DEFAULT_MODEL
  );
  console.log(
    'Reasoning effort: ' + REASONING_EFFORT
  );
  console.log(
    'Reasoning display: ' + SHOW_REASONING
  );
  console.log(
    'Max tokens default: 16384'
  );
  console.log(
    'Temperature default: 1.0'
  );
  console.log(
    'NVIDIA API configured: ' +
    Boolean(NVIDIA_API_KEY)
  );
  console.log('==========================================');
});
