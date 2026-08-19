-- Atomic sliding-window-counter rate limit check/reconcile, one round trip.
--
-- KEYS[1] = rate limit key, e.g. "{tenant_api_key}:{model}"
-- ARGV[1] = now_ms
-- ARGV[2] = window_s
-- ARGV[3] = rpm_limit
-- ARGV[4] = tpm_limit
-- ARGV[5] = requested_tokens        (mode "check")
-- ARGV[6] = mode: "check" | "reconcile"
-- ARGV[7] = token_delta             (mode "reconcile"; may be negative)
--
-- Returns {allowed(0|1), weighted_tokens, weighted_requests, elapsed_ms}.
-- elapsed_ms (not seconds) is returned because Redis truncates
-- non-integer Lua numbers to integers on the way back to the client;
-- the caller derives Retry-After in Python from window_s - elapsed_ms/1000.
--
-- Approach: fixed-window counters per window id, weighted against the
-- previous window by how far into the current window we are. This is an
-- approximation (bounded error at window boundaries), traded deliberately
-- for O(1) work per request instead of an exact sorted-set sliding log.

local key = KEYS[1]
local now_ms = tonumber(ARGV[1])
local window_s = tonumber(ARGV[2])
local rpm_limit = tonumber(ARGV[3])
local tpm_limit = tonumber(ARGV[4])
local requested_tokens = tonumber(ARGV[5])
local mode = ARGV[6]

local window_ms = window_s * 1000
local window_id = math.floor(now_ms / window_ms)
local elapsed_ms = now_ms - (window_id * window_ms)
local prev_weight = 1 - (elapsed_ms / window_ms)

local curr_key = key .. ":" .. window_id
local prev_key = key .. ":" .. (window_id - 1)

local curr = redis.call("HMGET", curr_key, "tokens", "requests")
local prev = redis.call("HMGET", prev_key, "tokens", "requests")

local curr_tokens = tonumber(curr[1]) or 0
local curr_requests = tonumber(curr[2]) or 0
local prev_tokens = tonumber(prev[1]) or 0
local prev_requests = tonumber(prev[2]) or 0

local weighted_tokens = prev_tokens * prev_weight + curr_tokens
local weighted_requests = prev_requests * prev_weight + curr_requests

if mode == "reconcile" then
    local token_delta = tonumber(ARGV[7])
    redis.call("HINCRBY", curr_key, "tokens", token_delta)
    redis.call("EXPIRE", curr_key, window_s * 2)
    return { 1, weighted_tokens, weighted_requests, elapsed_ms }
end

local allowed = 1
if (weighted_tokens + requested_tokens) > tpm_limit or (weighted_requests + 1) > rpm_limit then
    allowed = 0
else
    redis.call("HINCRBY", curr_key, "tokens", requested_tokens)
    redis.call("HINCRBY", curr_key, "requests", 1)
    redis.call("EXPIRE", curr_key, window_s * 2)
end

return { allowed, weighted_tokens, weighted_requests, elapsed_ms }
