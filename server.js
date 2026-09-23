// server.js - OpenAI-compatible proxy for NVIDIA NIM
// Designed for Render + Janitor AI

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
  process.env.NVIDIA_MODEL ||
  moonshotai/kimi-k3;

// Thinking is enabled for the model,
// but reasoning content is NOT shown to Janitor.
const SHOW_REASONING =
  String(process.env.SHOW_REASONING || 'false').toLowerCase() === 'true';

const ENABLE_THINKING =
  String(process.env.ENABLE_THINKING || 'true').toLowerCase() === 'true';

app.use(cors());

app.use(express.json({ limit: '5mb' }));

app.use(express.urlencoded({
  extended: true,
  limit: '5mb'
}));


// ============================================================
// REQUEST LOGGER
// ============================================================

app.use((req, res, next) => {

  console.log(
    `${new Date().toISOString()} ${req.method} ${req.path}`
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

  'nemotron-3-ultra': DEFAULT_MODEL,
  'nemotron-3-ultra-550b-a55b': DEFAULT_MODEL,

  [DEFAULT_MODEL]: DEFAULT_MODEL

};


// ============================================================
// HELPERS
// ============================================================

function resolveModel(requestedModel) {

  if (!requestedModel) {
    return DEFAULT_MODEL;
  }

  // Allow NVIDIA model IDs to pass directly.
  if (requestedModel.startsWith('nvidia/')) {
    return requestedModel;
  }

  return MODEL_MAPPING[requestedModel] || DEFAULT_MODEL;

}


function makeError(
  message,
  type = 'invalid_request_error',
  code = 400
) {

  return {

    error: {

      message,
      type,
      code

    }

  };

}


function normalizeMessages(messages) {

  if (!Array.isArray(messages)) {
    return [];
  }

  return messages.map((message) => {

    if (!message || typeof message !== 'object') {
      return message;
    }

    return {

      role: message.role,

      content: message.content,

      ...(message.name
        ? { name: message.name }
        : {}),

      ...(message.tool_calls
        ? { tool_calls: message.tool_calls }
        : {}),

      ...(message.tool_call_id
        ? { tool_call_id: message.tool_call_id }
        : {})

    };

  });

}


function extractText(value) {

  if (value == null) {
    return '';
  }

  if (typeof value === 'string') {
    return value;
  }

  if (Array.isArray(value)) {

    return value
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

  return String(value);

}


function sendSSE(res, payload) {

  if (typeof payload === 'string') {

    res.write(
      `data: ${payload}\n\n`
    );

    return;

  }

  res.write(
    `data: ${JSON.stringify(payload)}\n\n`
  );

}


function makeChatChunk({
  id,
  model,
  created,
  delta,
  finish_reason = null
}) {

  return {

    id,

    object:
      'chat.completion.chunk',

    created,

    model,

    choices: [

      {

        index: 0,

        delta,

        finish_reason

      }

    ]

  };

}


function getNvidiaErrorMessage(error) {

  if (error.response?.data) {

    const data =
      error.response.data;

    if (typeof data === 'string') {
      return data;
    }

    if (data.error?.message) {
      return data.error.message;
    }

    if (data.message) {
      return data.message;
    }

    try {

      return JSON.stringify(data);

    } catch {

      return 'NVIDIA API returned an error.';

    }

  }

  return (
    error.message ||
    'NVIDIA API request failed.'
  );

}


// ============================================================
// HEALTH CHECK
// ============================================================

function healthResponse(req, res) {

  res.json({

    status:
      'ok',

    service:
      'OpenAI to NVIDIA NIM Proxy',

    model:
      DEFAULT_MODEL,

    reasoning_display:
      SHOW_REASONING,

    thinking_mode:
      ENABLE_THINKING,

    nim_api_configured:
      Boolean(NVIDIA_API_KEY)

  });

}


app.get('/health', healthResponse);

app.get('/health/', healthResponse);


// ============================================================
// MODELS ENDPOINT
// ============================================================

app.get('/v1/models', (req, res) => {

  const aliases =
    Object.keys(MODEL_MAPPING);

  const models =
    aliases.map((model) => ({

      id: model,

      object:
        'model',

      created:
        Math.floor(
          Date.now() / 1000
        ),

      owned_by:
        'nvidia-nim'

    }));


  if (
    !models.some(
      (model) =>
        model.id === DEFAULT_MODEL
    )
  ) {

    models.push({

      id:
        DEFAULT_MODEL,

      object:
        'model',

      created:
        Math.floor(
          Date.now() / 1000
        ),

      owned_by:
        'nvidia'

    });

  }


  res.json({

    object:
      'list',

    data:
      models

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

      return res
        .status(500)
        .json(

          makeError(

            'NVIDIA API key is not configured on the proxy server.',

            'server_error',

            500

          )

        );

    }


    const body =
      req.body || {};


    const requestedModel =
      body.model;


    const messages =
      normalizeMessages(
        body.messages
      );


    // --------------------------------------------------------
    // REQUIRED MESSAGE CHECK
    // --------------------------------------------------------

    if (!messages.length) {

      return res
        .status(400)
        .json(

          makeError(

            'Missing required field: messages',

            'invalid_request_error',

            400

          )

        );

    }


    // --------------------------------------------------------
    // MODEL
    // --------------------------------------------------------

    const nimModel =
      resolveModel(
        requestedModel
      );


    console.log(
      `Model requested: ${
        requestedModel || '(none)'
      } -> NVIDIA: ${nimModel}`
    );


    // --------------------------------------------------------
    // GENERATION PARAMETERS
    // --------------------------------------------------------

    // NVIDIA recommends temperature=1.0
    // and top_p=0.95 for Nemotron-3-Super.

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

      model:
        nimModel,

      messages,

      temperature,

      top_p:
        typeof body.top_p === 'number'
          ? body.top_p
          : 0.95,

      max_tokens:
        maxTokens,

      stream

    };


    // --------------------------------------------------------
    // OPTIONAL PARAMETERS
    // --------------------------------------------------------

    if (
      typeof body.presence_penalty === 'number'
    ) {

      nimRequest.presence_penalty =
        body.presence_penalty;

    }


    if (
      typeof body.frequency_penalty === 'number'
    ) {

      nimRequest.frequency_penalty =
        body.frequency_penalty;

    }


    if (
      body.stop !== undefined
    ) {

      nimRequest.stop =
        body.stop;

    }


    // ========================================================
    // NEMOTRON THINKING
    // ========================================================

    // IMPORTANT:
    //
    // NVIDIA's current documentation expects
    // chat_template_kwargs DIRECTLY in the request body.
    //
    // We intentionally do NOT use:
    //
    // nimRequest.extra_body = ...
    //
    // because the NVIDIA endpoint previously rejected that
    // structure for this deployment.

    nimRequest.chat_template_kwargs = {

      enable_thinking:
        ENABLE_THINKING

    };


    console.log(
      `Thinking mode: ${
        ENABLE_THINKING
          ? 'ENABLED'
          : 'DISABLED'
      }`
    );


    console.log(
      `Generation: temperature=${temperature}, top_p=${
        nimRequest.top_p
      }, max_tokens=${maxTokens}`
    );


    console.log(
      `Sending request to NVIDIA: ${
        NVIDIA_API_BASE
      }/chat/completions`
    );


    // --------------------------------------------------------
    // SEND
    // --------------------------------------------------------

    try {

      if (stream) {

        await handleStreamingRequest(
          req,
          res,
          nimRequest,
          nimModel
        );

      } else {

        await handleNormalRequest(
          res,
          nimRequest,
          nimModel
        );

      }

    } catch (error) {

      console.error(
        'Proxy error:',
        error.message
      );

      console.error(
        'NVIDIA error details:',
        error.response?.data ||
          error
      );


      if (res.headersSent) {

        try {

          res.end();

        } catch (_) {}

        return;

      }


      const status =
        error.response?.status ||
        500;


      res
        .status(status)
        .json(

          makeError(

            getNvidiaErrorMessage(
              error
            ),

            status >= 500
              ? 'server_error'
              : 'invalid_request_error',

            status

          )

        );

    }

  }
);


// ============================================================
// NORMAL REQUEST
// ============================================================

async function handleNormalRequest(
  res,
  nimRequest,
  nimModel
) {

  const response =
    await axios.post(

      `${NVIDIA_API_BASE}/chat/completions`,

      nimRequest,

      {

        headers: {

          Authorization:
            `Bearer ${NVIDIA_API_KEY}`,

          'Content-Type':
            'application/json',

          Accept:
            'application/json'

        },

        timeout:
          180000,

        validateStatus:
          () => true

      }

    );


  // ----------------------------------------------------------
  // NVIDIA ERROR
  // ----------------------------------------------------------

  if (
    response.status < 200 ||
    response.status >= 300
  ) {

    console.error(
      'NVIDIA API error:',
      response.status,
      response.data
    );


    return res
      .status(response.status)
      .json(

        makeError(

          response.data?.error?.message ||
            response.data?.message ||
            'NVIDIA API request failed.',

          'invalid_request_error',

          response.status

        )

      );

  }


  const data =
    response.data;


  const created =
    data.created ||
    Math.floor(
      Date.now() / 1000
    );


  const id =
    data.id ||
    `chatcmpl-${Date.now()}`;


  // ----------------------------------------------------------
  // CHOICES
  // ----------------------------------------------------------

  const choices =
    Array.isArray(data.choices)

      ? data.choices.map(
          (choice, index) => {

            const message =
              choice.message || {};


            let content =
              extractText(
                message.content
              );


            const reasoning =
              extractText(
                message.reasoning_content
              );


            // Reasoning is intentionally hidden
            // unless SHOW_REASONING=true.

            if (
              SHOW_REASONING &&
              reasoning
            ) {

              content =
                `<think>${reasoning}</think>\n\n${content}`;

            }


            return {

              index,

              message: {

                role:
                  message.role ||
                  'assistant',

                content

              },

              finish_reason:
                choice.finish_reason ||
                'stop'

            };

          }

        )

      : [

          {

            index: 0,

            message: {

              role:
                'assistant',

              content:
                ''

            },

            finish_reason:
              'stop'

          }

        ];


  // ----------------------------------------------------------
  // OPENAI RESPONSE
  // ----------------------------------------------------------

  return res.json({

    id,

    object:
      'chat.completion',

    created,

    model:
      nimModel,

    choices,

    usage:
      data.usage || {

        prompt_tokens:
          0,

        completion_tokens:
          0,

        total_tokens:
          0

      }

  });

}


// ============================================================
// STREAMING REQUEST
// ============================================================

async function handleStreamingRequest(
  req,
  res,
  nimRequest,
  nimModel
) {

  const response =
    await axios.post(

      `${NVIDIA_API_BASE}/chat/completions`,

      nimRequest,

      {

        headers: {

          Authorization:
            `Bearer ${NVIDIA_API_KEY}`,

          'Content-Type':
            'application/json',

          Accept:
            'text/event-stream'

        },

        responseType:
          'stream',

        timeout:
          180000,

        validateStatus:
          () => true

      }

    );


  // ----------------------------------------------------------
  // STREAM ERROR
  // ----------------------------------------------------------

  if (
    response.status < 200 ||
    response.status >= 300
  ) {

    let errorBody = '';


    response.data.on(
      'data',
      (chunk) => {

        errorBody +=
          chunk.toString();

      }
    );


    await new Promise(
      (resolve) => {

        response.data.on(
          'end',
          resolve
        );

      }
    );


    console.error(
      'NVIDIA streaming error:',
      response.status,
      errorBody
    );


    if (!res.headersSent) {

      res
        .status(response.status)
        .json(

          makeError(

            (() => {

              try {

                const parsed =
                  JSON.parse(
                    errorBody
                  );

                return (
                  parsed.error?.message ||
                  parsed.message ||
                  errorBody ||
                  'NVIDIA API request failed.'
                );

              } catch {

                return (
                  errorBody ||
                  'NVIDIA API request failed.'
                );

              }

            })(),

            'invalid_request_error',

            response.status

          )

        );

    }


    return;

  }


  // ----------------------------------------------------------
  // SSE HEADERS
  // ----------------------------------------------------------

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


  if (res.flushHeaders) {
    res.flushHeaders();
  }


  const id =
    `chatcmpl-${Date.now()}`;


  const created =
    Math.floor(
      Date.now() / 1000
    );


  let buffer = '';

  let reasoningStarted =
    false;

  let finished =
    false;


  // ----------------------------------------------------------
  // SEND CONTENT
  // ----------------------------------------------------------

  const writeContent =
    (text) => {

      if (!text) {
        return;
      }


      sendSSE(

        res,

        makeChatChunk({

          id,

          model:
            nimModel,

          created,

          delta: {

            role:
              'assistant',

            content:
              text

          }

        })

      );

    };


  // ----------------------------------------------------------
  // SEND REASONING
  // ----------------------------------------------------------

  const writeReasoning =
    (text) => {

      if (
        !SHOW_REASONING ||
        !text
      ) {

        return;

      }


      sendSSE(

        res,

        makeChatChunk({

          id,

          model:
            nimModel,

          created,

          delta: {

            role:
              'assistant',

            content:
              text

          }

        })

      );

    };


  // ----------------------------------------------------------
  // PROCESS SSE EVENT
  // ----------------------------------------------------------

  const processEvent =
    (eventText) => {

      const lines =
        eventText.split('\n');


      for (
        const rawLine of lines
      ) {

        const line =
          rawLine.trim();


        if (
          !line ||
          !line.startsWith('data:')
        ) {

          continue;

        }


        const payload =
          line
            .slice(5)
            .trim();


        if (!payload) {
          continue;
        }


        if (
          payload === '[DONE]'
        ) {

          continue;

        }


        let data;


        try {

          data =
            JSON.parse(
              payload
            );

        } catch (error) {

          console.error(
            'Error parsing NVIDIA stream chunk:',
            error
          );

          continue;

        }


        const choice =
          data.choices?.[0];


        if (!choice) {
          continue;
        }


        const delta =
          choice.delta || {};


        const reasoning =
          extractText(
            delta.reasoning_content
          );


        const content =
          extractText(
            delta.content
          );


        // ----------------------------------------------------
        // REASONING
        // ----------------------------------------------------

        if (
          SHOW_REASONING &&
          reasoning
        ) {

          if (
            !reasoningStarted
          ) {

            writeReasoning(
              '<think>'
            );

            reasoningStarted =
              true;

          }


          writeReasoning(
            reasoning
          );

        }


        // ----------------------------------------------------
        // CONTENT
        // ----------------------------------------------------

        if (content) {

          if (
            SHOW_REASONING &&
            reasoningStarted
          ) {

            writeReasoning(
              '</think>\n\n'
            );

            reasoningStarted =
              false;

          }


          writeContent(
            content
          );

        }


        // ----------------------------------------------------
        // FINISH
        // ----------------------------------------------------

        if (
          choice.finish_reason
        ) {

          if (
            SHOW_REASONING &&
            reasoningStarted
          ) {

            writeReasoning(
              '</think>\n\n'
            );

            reasoningStarted =
              false;

          }


          sendSSE(

            res,

            makeChatChunk({

              id,

              model:
                nimModel,

              created,

              delta: {},

              finish_reason:
                choice.finish_reason

            })

          );

        }

      }

    };


  // ----------------------------------------------------------
  // RECEIVE STREAM
  // ----------------------------------------------------------

  response.data.on(
    'data',
    (chunk) => {

      buffer +=
        chunk.toString('utf8');


      const events =
        buffer.split(
          /\r?\n\r?\n/
        );


      buffer =
        events.pop() || '';


      for (
        const event of events
      ) {

        processEvent(
          event
        );

      }

    }
  );


  // ----------------------------------------------------------
  // STREAM END
  // ----------------------------------------------------------

  response.data.on(
    'end',
    () => {

      if (finished) {
        return;
      }


      finished =
        true;


      if (
        SHOW_REASONING &&
        reasoningStarted
      ) {

        writeReasoning(
          '</think>\n\n'
        );

      }


      sendSSE(
        res,
        '[DONE]'
      );


      res.end();

    }
  );


  // ----------------------------------------------------------
  // STREAM ERROR
  // ----------------------------------------------------------

  response.data.on(
    'error',
    (error) => {

      if (finished) {
        return;
      }


      finished =
        true;


      console.error(
        'NVIDIA stream error:',
        error
      );


      if (!res.headersSent) {

        res
          .status(500)
          .json(

            makeError(

              'NVIDIA streaming connection failed.',

              'server_error',

              500

            )

          );

      } else {

        res.end();

      }

    }
  );


  // ----------------------------------------------------------
  // CLIENT DISCONNECTED
  // ----------------------------------------------------------

  req.on(
    'close',
    () => {

      if (!finished) {

        try {

          response.data.destroy();

        } catch (_) {}

      }

    }
  );

}


// ============================================================
// ROOT
// ============================================================

app.get('/', (req, res) => {

  res.json({

    status:
      'online',

    service:
      'OpenAI to NVIDIA NIM Proxy',

    endpoints: {

      health:
        '/health',

      models:
        '/v1/models',

      chat:
        '/v1/chat/completions'

    },

    model:
      DEFAULT_MODEL

  });

});


// ============================================================
// 404
// ============================================================

app.use(
  (req, res) => {

    console.log(
      `404: ${req.method} ${req.path} not found`
    );


    res
      .status(404)
      .json(

        makeError(

          `Endpoint ${req.method} ${req.path} not found.`,

          'invalid_request_error',

          404

        )

      );

  }
);


// ============================================================
// START SERVER
// ============================================================

const server =
  app.listen(
    PORT,
    '0.0.0.0',
    () => {

      console.log(
        '=============================================='
      );

      console.log(
        `OpenAI -> NVIDIA NIM Proxy running on port ${PORT}`
      );

      console.log(
        `Health: http://localhost:${PORT}/health`
      );

      console.log(
        `Models: http://localhost:${PORT}/v1/models`
      );

      console.log(
        `Chat:   POST http://localhost:${PORT}/v1/chat/completions`
      );

      console.log(
        `NVIDIA API: ${NVIDIA_API_BASE}`
      );

      console.log(
        `Default model: ${DEFAULT_MODEL}`
      );

      console.log(
        `Reasoning display: ${
          SHOW_REASONING
            ? 'ENABLED'
            : 'DISABLED'
        }`
      );

      console.log(
        `Thinking mode: ${
          ENABLE_THINKING
            ? 'ENABLED'
            : 'DISABLED'
        }`
      );

      console.log(
        `NVIDIA API key configured: ${
          NVIDIA_API_KEY
            ? 'YES'
            : 'NO'
        }`
      );

      console.log(
        '=============================================='
      );

    }
  );


server.on(
  'error',
  (error) => {

    console.error(
      'Server error:',
      error
    );

  }
);


// ============================================================
// EXPORT
// ============================================================

module.exports = app;
