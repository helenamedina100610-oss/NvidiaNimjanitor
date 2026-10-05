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

  if (typeof payload === 'string') {

    res.write(
      'data: ' +
      payload +
      '\n\n'
    );

    return;
  }

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
  finishReason = null,
  includeRole = false
) {

  const delta = {};

  if (includeRole) {
    delta.role = 'assistant';
  }

  if (content) {
    delta.content = content;
  }

  return {
    id,
    object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        delta,
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

        console.log(
          'NVIDIA REQUEST START'
        );


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


        console.log(
          'NVIDIA status: ' +
          response.status
        );

        console.log(
          'NVIDIA HTTP response received.'
        );


        // ----------------------------------------------------
        // NVIDIA ERROR
        // ----------------------------------------------------

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


        // ----------------------------------------------------
        // JANITOR SSE HEADERS
        // ----------------------------------------------------

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


        // ----------------------------------------------------
        // COMPLETION ID
        // ----------------------------------------------------

        const completionId =
          'chatcmpl-' +
          Date.now();


        // ----------------------------------------------------
        // STREAM STATE
        // ----------------------------------------------------

        let buffer = '';

        let firstDataReceived = false;

        let roleSent = false;

        let finishSent = false;

        let doneReceived = false;

        let streamFinished = false;

        let clientClosed = false;


        // ----------------------------------------------------
        // SEND ASSISTANT ROLE
        // ----------------------------------------------------

        function sendAssistantRole() {

          if (
            roleSent ||
            res.writableEnded
          ) {
            return;
          }

          roleSent = true;

          const roleChunk =
            makeChatChunk(
              completionId,
              nimModel,
              '',
              null,
              true
            );

          sendSSE(
            res,
            roleChunk
          );
        }


        // ----------------------------------------------------
        // SEND FINISH
        // ----------------------------------------------------

        function sendFinish(reason) {

          if (
            finishSent ||
            res.writableEnded
          ) {
            return;
          }

          finishSent = true;

          const finishChunk =
            makeChatChunk(
              completionId,
              nimModel,
              '',
              reason || 'stop',
              false
            );

          sendSSE(
            res,
            finishChunk
          );
        }


        // ----------------------------------------------------
        // PROCESS ONE SSE EVENT
        // ----------------------------------------------------

        function processSSEEvent(event) {

          if (
            !event ||
            doneReceived ||
            clientClosed
          ) {
            return;
          }


          const lines =
            event.split(/\r?\n/);


          const dataLines = [];


          for (
            const rawLine of lines
          ) {

            if (
              rawLine.startsWith('data:')
            ) {

              dataLines.push(
                rawLine
                  .slice(5)
                  .trim()
              );
            }
          }


          if (
            dataLines.length === 0
          ) {
            return;
          }


          const rawData =
            dataLines.join('\n');


          // --------------------------------------------------
          // DONE
          // --------------------------------------------------

          if (
            rawData === '[DONE]'
          ) {

            doneReceived = true;

            return;
          }


          if (!rawData) {
            return;
          }


          let parsed;

          try {

            parsed =
              JSON.parse(rawData);

          } catch (error) {

            console.error(
              'Could not parse NVIDIA SSE JSON:',
              rawData
            );

            return;
          }


          if (!firstDataReceived) {

            firstDataReceived = true;

            console.log(
              'First NVIDIA stream data received.'
            );
          }


          // --------------------------------------------------
          // CHOICE
          // --------------------------------------------------

          const choice =
            parsed &&
            Array.isArray(parsed.choices) &&
            parsed.choices.length > 0
              ? parsed.choices[0]
              : null;


          if (!choice) {
            return;
          }


          // --------------------------------------------------
          // DELTA
          // --------------------------------------------------

          const delta =
            choice.delta || {};


          // --------------------------------------------------
          // ROLE
          // --------------------------------------------------

          if (
            delta.role === 'assistant' ||
            !roleSent
          ) {

            sendAssistantRole();
          }


          // --------------------------------------------------
          // REASONING
          // --------------------------------------------------

          let content = '';

          if (
            SHOW_REASONING &&
            delta.reasoning_content
          ) {

            content =
              extractText(
                delta.reasoning_content
              );
          }


          // --------------------------------------------------
          // NORMAL CONTENT
          // --------------------------------------------------

          if (delta.content) {

            content +=
              extractText(
                delta.content
              );
          }


          // --------------------------------------------------
          // SEND CONTENT
          // --------------------------------------------------

          if (content) {

            const output =
              makeChatChunk(
                completionId,
                nimModel,
                content,
                null,
                false
              );

            sendSSE(
              res,
              output
            );
          }


          // --------------------------------------------------
          // FINISH REASON
          // --------------------------------------------------

          if (
            choice.finish_reason &&
            !finishSent
          ) {

            sendFinish(
              choice.finish_reason
            );
          }
        }


        // ----------------------------------------------------
        // RECEIVE NVIDIA STREAM
        // ----------------------------------------------------

        response.data.on(
          'data',
          (chunk) => {

            if (
              clientClosed ||
              streamFinished
            ) {
              return;
            }


            buffer +=
              chunk.toString('utf8');


            // Support both LF and CRLF.
            buffer =
              buffer.replace(
                /\r\n/g,
                '\n'
              );


            // ------------------------------------------------
            // PROCESS COMPLETE SSE EVENTS
            // ------------------------------------------------

            while (true) {

              const separator =
                buffer.indexOf('\n\n');


              if (
                separator === -1
              ) {
                break;
              }


              const event =
                buffer.slice(
                  0,
                  separator
                );


              buffer =
                buffer.slice(
                  separator + 2
                );


              processSSEEvent(
                event
              );


              if (
                doneReceived
              ) {
                break;
              }
            }
          }
        );


        // ----------------------------------------------------
        // STREAM END
        // ----------------------------------------------------

        response.data.on(
          'end',
          () => {

            if (
              streamFinished
            ) {
              return;
            }


            streamFinished = true;


            console.log(
              'NVIDIA stream ended.'
            );


            // ------------------------------------------------
            // PROCESS REMAINING BUFFER
            // ------------------------------------------------

            if (
              buffer.trim() &&
              !doneReceived
            ) {

              processSSEEvent(
                buffer
              );

              buffer = '';
            }


            // ------------------------------------------------
            // ENSURE ROLE
            // ------------------------------------------------

            if (
              !roleSent &&
              !res.writableEnded
            ) {

              sendAssistantRole();
            }


            // ------------------------------------------------
            // SEND FINAL STOP
            // ------------------------------------------------

            if (
              !finishSent &&
              !res.writableEnded
            ) {

              sendFinish('stop');
            }


            // ------------------------------------------------
            // OPENAI DONE
            // ------------------------------------------------

            if (
              !res.writableEnded
            ) {

              sendSSE(
                res,
                '[DONE]'
              );

              res.end();
            }
          }
        );


        // ----------------------------------------------------
        // STREAM ERROR
        // ----------------------------------------------------

        response.data.on(
          'error',
          (error) => {

            if (
              streamFinished
            ) {
              return;
            }


            streamFinished = true;


            console.error(
              'NVIDIA stream connection error:',
              error
            );


            if (
              !res.writableEnded
            ) {

              res.end();
            }
          }
        );


        // ----------------------------------------------------
        // CLIENT ABORTED
        // ----------------------------------------------------

        req.on(
          'aborted',
          () => {

            clientClosed = true;

            console.log(
              'Janitor client aborted the request.'
            );


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


        if (
          !res.headersSent
        ) {

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


        if (
          !res.writableEnded
        ) {

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
