#!/usr/bin/env bash
# Smoketests for glm53. See lib.sh for the conventions.
#
# These test what clients rely on, not raw capability. The one that matters
# most here is fluency-vs-correctness: a TP/shard/quantisation mismatch on this
# model loads cleanly, reports healthy and serves fluent nonsense, so every
# case asks for an answer that can be checked.
source "$(dirname "$0")/lib.sh" "$@"

get() { # PATH -> STATUS, BODY, SECS
  local out
  out=$(curl -s --max-time "$TIMEOUT" -w '\n%{http_code} %{time_total}' "$BASE$1")
  BODY="${out%$'\n'*}"; read -r STATUS SECS <<<"${out##*$'\n'}"
}

chat() { # JSON-extra MAX_TOKENS USER-TEXT -> request body
  jq -n --arg m "$NAME" --argjson x "$1" --argjson n "$2" --arg q "$3" \
    '{model:$m, max_tokens:$n, temperature:0, messages:[{role:"user", content:$q}]} + $x'
}
OFF='{"chat_template_kwargs":{"enable_thinking":false}}'

# The served name alone proves nothing: any server can be told to answer to
# glm53. The root is the path the engine loaded from, where this compose mounts
# the checkpoint, so this fails against another model even when every other
# case would pass. It passes through mentat-serve unchanged.
t_models_lists_glm_checkpoint() {
  get /v1/models
  check "should list $NAME" "[.data[].id] | index(\"$NAME\") != null" || return
  check "$NAME should be a GLM-5.3 checkpoint" \
    ".data[] | select(.id == \"$NAME\") | .root | test(\"glm-5\\\\.3\"; \"i\")" || return
  ok "$(jqb --arg n "$NAME" '.data[] | select(.id == $n) | .root')"
}

# Thinking is on by default: a correct answer, with the trace in the reasoning
# field rather than in the content.
t_thinking_on_by_default() {
  post /v1/chat/completions "$(chat '{}' 2048 'What is 17*23? Reply with the number only.')"
  check "should finish" '.choices[0].finish_reason == "stop"' || return
  check "content should be 391" '.choices[0].message.content | test("391")' || return
  check "reasoning should be present" '(.choices[0].message.reasoning // .choices[0].message.reasoning_content // "") | length > 0' || return
  ok "$(jqb '.usage.completion_tokens') tok"
}

# Thinking off uses the template's empty thinking block; the answer must be
# clean and the parser must not classify it as reasoning.
t_thinking_off_per_request() {
  post /v1/chat/completions "$(chat "$OFF" 512 'What is 17*23? Reply with the number only.')"
  check "content should be 391" '.choices[0].message.content | test("391")' || return
  check "content should be short, not a trace" '.choices[0].message.content | length < 40' || return
  check "content should hold no think tags" '.choices[0].message.content | test("think>") | not' || return
  check "reasoning should be absent" '(.choices[0].message.reasoning // .choices[0].message.reasoning_content // "") | length == 0' || return
  ok "$(jqb '.usage.completion_tokens') tok"
}

# A known fact, as the in-image self-test asks. Corrupted weights answer
# fluently and wrongly.
t_fact() {
  post /v1/chat/completions "$(chat "$OFF" 512 'What is the capital of Australia? One word.')"
  check "should answer Canberra" '.choices[0].message.content | test("Canberra")' || return
  ok
}

# The tool parser: a well-formed call to an offered tool must come back parsed.
# The fail-closed parser (TOOL_PARSER=glm47_failclosed) was once silently
# dropped from production by a compose override.
t_tool_call_parsed() {
  post /v1/chat/completions "$(jq -n --arg m "$NAME" '{
    model:$m, max_tokens:1024, temperature:0, tool_choice:"auto",
    chat_template_kwargs:{enable_thinking:false},
    tools:[{type:"function", function:{name:"get_weather", description:"Get the current weather for a city.",
            parameters:{type:"object", properties:{city:{type:"string"}}, required:["city"]}}}],
    messages:[{role:"user", content:"What is the weather in Paris right now? Use the tool."}]}')"
  check "should call get_weather" '.choices[0].message.tool_calls[0].function.name == "get_weather"' || return
  check "arguments should name Paris" '.choices[0].message.tool_calls[0].function.arguments | fromjson | .city | test("Paris")' || return
  ok
}

# A tool continuation must return a real separate trace and the right call.
t_tool_continuation_reasoning() {
  post /v1/chat/completions "$(jq -n --arg m "$NAME" '{
    model:$m, max_tokens:2048, temperature:0, tool_choice:"auto",
    chat_template_kwargs:{enable_thinking:true,clear_thinking:false},
    tools:[
      {type:"function",function:{name:"get_factors",description:"Fetch two integer factors.",parameters:{type:"object",properties:{}}}},
      {type:"function",function:{name:"submit_product",description:"Submit their product.",parameters:{type:"object",properties:{product:{type:"integer"}},required:["product"]}}}
    ],
    messages:[
      {role:"user",content:"Get the factors, multiply them, and call submit_product with the result."},
      {role:"assistant",content:null,tool_calls:[{id:"factors-1",type:"function",function:{name:"get_factors",arguments:"{}"}}]},
      {role:"tool",tool_call_id:"factors-1",content:"{\"left\":37,\"right\":29}"}
    ]}')"
  check "should finish with a tool call" '.choices[0].finish_reason == "tool_calls"' || return
  check "reasoning should be present after the result" '(.choices[0].message.reasoning // .choices[0].message.reasoning_content // "") | length > 0' || return
  check "should call submit_product" '.choices[0].message.tool_calls[0].function.name == "submit_product"' || return
  check "product should be 1073" '.choices[0].message.tool_calls[0].function.arguments | fromjson | .product == 1073' || return
  ok "$(jqb '.usage.completion_tokens') tok"
}

# Multimodal. A text-only chat template once made images fail here while text
# kept working.
t_image_question() {
  post /v1/chat/completions "$(jq -n --arg m "$NAME" --rawfile b64 <(base64 < "$HERE/page-table.png" | tr -d '\n') '{
    model:$m, max_tokens:64, temperature:0, chat_template_kwargs:{enable_thinking:false},
    messages:[{role:"user", content:[
      {type:"image_url", image_url:{url:("data:image/png;base64," + $b64)}},
      {type:"text", text:"In the table on this page, what is the Q4 value for the North region? Answer with the number only."}]}]}')"
  check "should read 1450 from the table" '.choices[0].message.content | test("1450")' || return
  ok
}

t_max_tokens_exact() {
  post /v1/chat/completions "$(chat "$OFF" 5 'Count from one to fifty in words.')"
  check "should stop on length" '.choices[0].finish_reason == "length"' || return
  check "should emit exactly 5 tokens" '.usage.completion_tokens == 5' || return
  ok
}

# The limit comes from the server, so this follows a MAX_MODEL_LEN change.
t_context_limit_refused() {
  post /tokenize "$(jq -n --arg m "$NAME" '{model:$m, prompt:"x"}')"
  local max; max=$(jqb '.max_model_len')
  post /v1/chat/completions "$(chat "$OFF" "$max" 'Say hi.')"
  [ "$STATUS" = 400 ] || { BODY="{\"status\": $STATUS}"; check "status should be 400" 'false'; return; }
  check "error should name the context length" '(.error.message // .message) | test("context length")' || return
  ok "limit $max"
}

run_all
