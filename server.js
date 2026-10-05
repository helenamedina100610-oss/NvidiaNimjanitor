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

function sendSSE(res, payload) {
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
  id,
  model,
  content,
  finishReason,
  role
) {
  const delta = {};

  if (role) {
    delta.role = 'assistant';
  }

  if (content) {
    delta.content = content;
  }

  return {
    id: id,
    object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model: model,
    choices: [
      {
        index: 0,
        delta: delta,
        finish_reason: finishReason || null
      }
    ]
  };
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
  console.log('NVIDIA REQUEST START');

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
    stream: stream,
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

    if (!stream) {
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

      console.log(
        'NVIDIA status:',
        response.status
      );

      if (response.status < 200 || response.status >= 300) {
        return res.status(response.status).json({
          error: {
            message: getNvidiaErrorMessage(response.data),
            type: 'upstream_error',
            code: response.status
          }
        });
      }

      return res.json(response.data);
    }

    const response = await axios({
      method: 'POST',
      url: NVIDIA_API_BASE + '/chat/completions',
      headers: {
        Authorization: 'Bearer ' + NVIDIA_API_KEY,
        'Content-Type': 'application/json',
        Accept: 'text/event-stream'
      },
      data: nimRequest,
      responseType: 'stream',
      timeout: 180000,
      validateStatus: function () {
        return true;
      }
    });

    console.log(
      'NVIDIA status:',
      response.status
    );

    if (response.status < 200 || response.status >= 300) {
      let errorBody = '';

      response.data.on('data', function (chunk) {
        errorBody += chunk.toString();
      });

      response.data.on('end', function () {
        console.error(
          'NVIDIA streaming error:',
          response.status,
          errorBody
        );

        if (!res.headersSent) {
          res.status(response.status).json({
            error: {
              message: getNvidiaErrorMessage(errorBody),
              type: 'upstream_error',
              code: response.status
            }
          });
        } else {
          res.end();
        }
      });

      return;
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

    const completionId =
      'chatcmpl-' + Date.now();

    let buffer = '';
    let roleSent = false;
    let finishSent = false;
    let gotContent = false;

    function sendRole() {
      if (roleSent) {
        return;
      }

      roleSent = true;

      sendSSE(
        res,
        makeChunk(
          completionId,
          nimModel,
          '',
          null,
          true
        )
      );
    }

    function sendFinish(reason) {
      if (finishSent) {
        return;
      }

      finishSent = true;

      sendSSE(
        res,
        makeChunk(
          completionId,
          nimModel,
          '',
          reason || 'stop',
          false
        )
      );
    }

    function processEvent(event) {
      const lines = event.split('\n');
      const dataLines = [];

      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];

        if (line.startsWith('data:')) {
          dataLines.push(
            line.slice(5).trimStart()
          );
        }
      }

      if (dataLines.length === 0) {
        return;
      }

      const dataText = dataLines.join('\n');

      if (dataText === '[DONE]') {
        return;
      }

      let parsed;

      try {
        parsed = JSON.parse(dataText);
      } catch (error) {
        return;
      }

      const choices = parsed.choices;

      if (!Array.isArray(choices) || !choices[0]) {
        return;
      }

      const choice = choices[0];
      const delta = choice.delta || {};

      sendRole();

      let content = '';

      if (
        SHOW_REASONING &&
        delta.reasoning_content
      ) {
        content += extractText(
          delta.reasoning_content
        );
      }

      if (delta.content) {
        content += extractText(
          delta.content
        );
      }

      if (content) {
        gotContent = true;

        sendSSE(
          res,
          makeChunk(
            completionId,
            parsed.model || nimModel,
            content,
            null,
            false
          )
        );
      }

      if (choice.finish_reason) {
        sendFinish(
          choice.finish_reason
        );
      }
    }

    response.data.on('data', function (chunk) {
      buffer += chunk.toString('utf8');

      buffer = buffer.replace(/\r\n/g, '\n');

      let separatorIndex;

      while (
        (separatorIndex = buffer.indexOf('\n\n')) !== -1
      ) {
        const event =
          buffer.slice(0, separatorIndex);

        buffer =
          buffer.slice(separatorIndex + 2);

        if (event.trim()) {
          processEvent(event);
        }
      }
    });

    response.data.on('end', function () {
      console.log('NVIDIA stream ended.');

      if (buffer.trim()) {
        processEvent(buffer);
      }

      if (!roleSent) {
        sendRole();
      }

      if (!finishSent) {
        sendFinish('stop');
      }

      sendSSE(res, '[DONE]');

      res.end();

      console.log(
        'Streaming response sent to Janitor.'
      );

      console.log(
        'Content received:',
        gotContent
      );
    });

    response.data.on('error', function (error) {
      console.error(
        'NVIDIA stream error:',
        error.message
      );

      if (!res.writableEnded) {
        res.end();
      }
    });

    req.on('aborted', function () {
      console.log(
        'Janitor connection aborted.'
      );

      if (
        response.data &&
        typeof response.data.destroy === 'function'
      ) {
        response.data.destroy();
      }
    });

  } catch (error) {
    console.error(
      'PROXY ERROR:',
      error.message
    );

    if (error.response) {
      console.error(
        'NVIDIA response:',
        error.response.status,
        error.response.data
      );
    }

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
