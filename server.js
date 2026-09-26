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

app.use(
  (req, res, next) => {
    console.log(
      new Date().toISOString() +
      ' ' +
      req.method +
      ' ' +
      req.path
    );

    next();
  }
);


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

  'claude-3': DEFAULT_MODEL,
  'claude-3-opus': DEFAULT_MODEL,
  'claude-3-sonnet': DEFAULT_MODEL,
  'claude-3-haiku': DEFAULT_MODEL,
  'claude-3.5-sonnet': DEFAULT_MODEL,
  'claude-3.7-sonnet': DEFAULT_MODEL,

  'gemini-pro': DEFAULT_MODEL,
  'gemini-1.5-pro': DEFAULT_MODEL,
  'gemini-2.0-flash': DEFAULT_MODEL,

  'nemotron': DEFAULT_MODEL,
  'nemotron-3-super': DEFAULT_MODEL,
  'nemotron-3-ultra': DEFAULT_MODEL
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
// SSE HELPER
// ============================================================

function sendSSE(res, payload) {

  res.write(
    'data: ' +
    JSON.stringify(payload) +
    '\n\n'
  );
}


// ============================================================
// CHAT COMPLETION CHUNK
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
// NVIDIA ERROR HANDLER
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
// OPENAI MODELS ENDPOINT
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
      'Received chat completion request.'
    );


    // --------------------------------------------------------
    // API KEY CHECK
    // --------------------------------------------------------

    if (!NVIDIA_API_KEY) {

      console.error(
        'NVIDIA_API_KEY is not configured.'
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
    // REQUEST BODY
    // --------------------------------------------------------

    const body = req.body || {};


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
      'Model requested: ' +
      requestedModel +
      ' -> NVIDIA: ' +
      nimModel
    );


    // --------------------------------------------------------
    // MESSAGES
    // --------------------------------------------------------

    const messages =
      normalizeMessages(body.messages);


    // --------------------------------------------------------
    // TEMPERATURE
    // --------------------------------------------------------

    const temperature =
      typeof body.temperature === 'number'
        ? body.temperature
        : 1.0;


    // --------------------------------------------------------
    // MAX TOKENS
    // --------------------------------------------------------

    const maxTokens =
      typeof body.max_tokens === 'number'
        ? body.max_tokens
        : typeof body.max_completion_tokens === 'number'
          ? body.max_completion_tokens
          : 16384;


    // --------------------------------------------------------
    // STREAM
    // --------------------------------------------------------

    const stream =
      Boolean(body.stream);


    // --------------------------------------------------------
    // NVIDIA REQUEST
    // --------------------------------------------------------

    const nimRequest = {
      model: nimModel,
      messages,
      temperature,
      max_tokens: maxTokens,
      stream,
      reasoning_effort: REASONING_EFFORT
    };


    // --------------------------------------------------------
    // OPTIONAL SEED
    // --------------------------------------------------------

    if (body.seed !== undefined) {
      nimRequest.seed = body.seed;
    }


    // --------------------------------------------------------
    // OPTIONAL STREAM OPTIONS
    // --------------------------------------------------------

    if (
      body.stream_options !== undefined
    ) {
      nimRequest.stream_options =
        body.stream_options;
    }


    console.log(
      'Sending request to NVIDIA: ' +
      NVIDIA_API_BASE +
      '/chat/completions'
    );


    // ========================================================
    // STREAMING REQUEST
    // ========================================================

    if (stream) {

      try {

        const response =
          await axios({
            method: 'POST',

            url:
              NVIDIA_API_BASE +
              '/chat/completions',

            headers: {
              Authorization:
                'Bearer ' +
                NVIDIA_API_KEY,

              'Content-Type':
                'application/json',

              Accept:
                'text/event-stream'
            },

            data: nimRequest,

            responseType: 'stream',

            timeout: 180000,

            validateStatus: () => true
          });


        if (
          response.status < 200 ||
          response.status >= 300
        ) {

          let errorData = '';

          response.data.on(
            'data',
            (chunk) => {
              errorData +=
                chunk.toString();
            }
          );

          response.data.on(
            'end',
            () => {

              console.error(
                'NVIDIA streaming error: ' +
                response.status +
                ' ' +
                errorData
              );

              if (!res.headersSent) {

                res.status(
                  response.status
                );

                res.json({
                  error: {
                    message:
                      errorData ||
                      'NVIDIA API error',
                    type:
                      'upstream_error'
                  }
                });
              }
            }
          );

          return;
        }


        res.status(200);

        res.setHeader(
          'Content-Type',
          'text/event-stream'
        );

        res.setHeader(
          'Cache-Control',
          'no-cache'
        );

        res.setHeader(
          'Connection',
          'keep-alive'
        );

        res.flushHeaders();


        const completionId =
          'chatcmpl-' +
          Date.now();


        let buffer = '';


        response.data.on(
          'data',
          (chunk) => {

            buffer +=
              chunk.toString();


            const events =
              buffer.split('\n\n');


            buffer =
              events.pop() || '';


            for (
              const event of events
            ) {

              const lines =
                event.split('\n');


              for (
                const line of lines
              ) {

                if (
                  !line.startsWith(
                    'data:'
                  )
                ) {
                  continue;
                }


                const rawData =
                  line
                    .slice(5)
                    .trim();


                if (
                  !rawData ||
                  rawData === '[DONE]'
                ) {

                  continue;
                }


                let parsed;

                try {

                  parsed =
                    JSON.parse(
                      rawData
                    );

                } catch {

                  continue;
                }


                const choice =
                  parsed &&
                  parsed.choices &&
                  parsed.choices[0];


                if (!choice) {
                  continue;
                }


                const delta =
                  choice.delta || {};


                let content =
                  delta.content || '';


                if (
                  !SHOW_REASONING &&
                  delta.reasoning_content
                ) {
                  content = '';
                }


                const finishReason =
                  choice.finish_reason ||
                  null;


                const output =
                  makeChatChunk(
                    completionId,
                    nimModel,
                    extractText(content),
                    finishReason
                  );


                sendSSE(
                  res,
                  output
                );
              }
            }
          }
        );


        response.data.on(
          'end',
          () => {

            sendSSE(
              res,
              makeChatChunk(
                completionId,
                nimModel,
                '',
                'stop'
              )
            );


            res.write(
              'data: [DONE]\n\n'
            );

            res.end();
          }
        );


        response.data.on(
          'error',
          (error) => {

            console.error(
              'NVIDIA stream connection error:',
              error
            );

            if (!res.writableEnded) {
              res.end();
            }
          }
        );


        req.on(
          'close',
          () => {

            if (
              response.data &&
              typeof response.data.destroy ===
                'function'
            ) {

              response.data.destroy();
            }
          }
        );


      } catch (error) {

        console.error(
          'NVIDIA streaming request failed:',
          error
        );


        if (!res.headersSent) {

          return res.status(500).json({
            error: {
              message:
                getNvidiaErrorMessage(
                  error
                ),
              type:
                'upstream_error'
            }
          });
        }


        if (!res.writableEnded) {
          res.end();
        }
      }


      return;
    }


    // ========================================================
    // NON-STREAMING REQUEST
    // ========================================================

    try {

      const response =
        await axios({
          method: 'POST',

          url:
            NVIDIA_API_BASE +
            '/chat/completions',

          headers: {
            Authorization:
              'Bearer ' +
              NVIDIA_API_KEY,

            'Content-Type':
              'application/json',

            Accept:
              'application/json'
          },

          data: nimRequest,

          timeout: 180000,

          validateStatus: () => true
        });


      if (
        response.status < 200 ||
        response.status >= 300
      ) {

        console.error(
          'NVIDIA API error: ' +
          response.status,
          response.data
        );


        return res
          .status(response.status)
          .json(
            response.data || {
              error: {
                message:
                  'NVIDIA API error',
                type:
                  'upstream_error'
              }
            }
          );
      }


      const data =
        response.data;


      if (
        !SHOW_REASONING &&
        data &&
        data.choices
      ) {

        data.choices =
          data.choices.map(
            (choice) => {

              if (
                choice &&
                choice.message
              ) {

                const message =
                  choice.message;


                if (
                  message.reasoning_content
                ) {

                  delete
                    message.reasoning_content;
                }
              }


              return choice;
            }
          );
      }


      return res
        .status(200)
        .json(data);


    } catch (error) {

      console.error(
        'NVIDIA request failed:',
        error
      );


      return res.status(500).json({
        error: {
          message:
            getNvidiaErrorMessage(
              error
            ),
          type:
            'upstream_error'
        }
      });
    }
  }
);


// ============================================================
// 404
// ============================================================

app.use(
  (req, res) => {

    res.status(404).json({
      error: {
        message:
          'Endpoint not found.',
        type:
          'not_found'
      }
    });
  }
);


// ============================================================
// START SERVER
// ============================================================

app.listen(
  PORT,
  '0.0.0.0',
  () => {

    console.log(
      '=================================================='
    );

    console.log(
      'OpenAI to NVIDIA NIM Proxy'
    );

    console.log(
      '=================================================='
    );

    console.log(
      'Server listening on port ' +
      PORT
    );

    console.log(
      'NVIDIA API: ' +
      NVIDIA_API_BASE
    );

    console.log(
      'Default model: ' +
      DEFAULT_MODEL
    );

    console.log(
      'Temperature default: 1.0'
    );

    console.log(
      'Max tokens default: 16384'
    );

    console.log(
      'Reasoning effort: ' +
      REASONING_EFFORT
    );

    console.log(
      'Reasoning display: ' +
      SHOW_REASONING
    );

    console.log(
      'NVIDIA API configured: ' +
      Boolean(NVIDIA_API_KEY)
    );

    console.log(
      '=================================================='
    );
  }
);
