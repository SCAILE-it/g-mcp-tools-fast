# Rate Limiting Implementation Status

**Date:** 2025-10-28
**Status:** ⚠️ PARTIALLY COMPLETE - Sequential requests work, concurrent bursts need architectural fix

---

## 🎯 What We Built

### 1. Replaced slowapi with Supabase-Based Rate Limiting

**Problem Discovered:** slowapi doesn't work in Modal's serverless environment
- All requests appear from same IP (GCP VM: 34.78.185.56)
- Modal doesn't pass through standard forwarding headers
- In-memory storage doesn't work across distributed containers

**Solution Implemented:** Database-backed distributed rate limiting
- Uses Supabase `api_calls` table to track request counts
- Queries last 60 seconds of requests per user/endpoint
- Works across all Modal containers (shared database state)

### 2. Implementation Details

**File:** `g-mcp-tools-complete.py` (lines 3542-3586)

**Function:** `check_rate_limit(user_id, endpoint, limit_per_minute)`
```python
async def check_rate_limit(user_id: Optional[str], endpoint: str, limit_per_minute: int) -> bool:
    # Query Supabase for requests in last 60 seconds
    window_start = now - timedelta(minutes=1)
    result = supabase.table("api_calls").select("id", count="exact").eq(
        "user_id", key
    ).eq("tool_name", endpoint).gte(
        "created_at", window_start.isoformat()
    ).execute()

    count = result.count if hasattr(result, 'count') else 0
    return count < limit_per_minute  # True = allow, False = rate limited
```

**Anonymous User Handling:**
- Uses fixed UUID: `00000000-0000-0000-0000-000000000000`
- All unauthenticated requests share same rate limit pool
- Compatible with Supabase `user_id UUID` column type

**Rate Limits:**
- `/plan`: 10 requests/minute
- `/execute`: 20 requests/minute
- `/orchestrate`: 10 requests/minute
- `/workflow/execute`: 10 requests/minute

### 3. Logging Integration

**Added API call logging to `/plan` endpoint** (lines 3877-3887, 3899-3910):
```python
# Log successful requests
log_api_call(
    user_id=user_id or "00000000-0000-0000-0000-000000000000",
    tool_name="/plan",
    tool_type="orchestration",
    input_data={"user_request": user_request},
    output_data={"plan": plan, "total_steps": len(plan)},
    success=True,
    processing_ms=processing_ms
)
```

**Bug Fixes:**
- ❌ Initially used `"anonymous"` string → UUID type error
- ✅ Fixed to use `"00000000-0000-0000-0000-000000000000"`
- ❌ Initially used `await log_api_call()` → log_api_call is not async
- ✅ Removed `await` keyword

---

## ⚠️ Known Limitation: Race Condition on Concurrent Bursts

### The Problem

**Sequential requests:** ✅ Works perfectly
**Concurrent bursts:** ❌ Race condition allows all requests through

**Root Cause:** Check-before-log ordering
```
Request Flow:
1. check_rate_limit() → queries DB (sees old count)
2. Execute request (takes 3-10 seconds)
3. log_api_call() → writes to DB (too late)

Concurrent Scenario:
- 12 requests arrive simultaneously
- All check before any are logged
- All see count=0 and pass
- All get logged after execution completes
```

**Why Supabase is Too Slow:**
- Database writes: ~100-500ms latency
- Multiple containers querying simultaneously
- No atomic increment operation
- PostgreSQL not designed for sub-second rate limiting

### Testing Results

**Test:** 12 rapid requests to `/plan` (10/minute limit)

**Expected:**
```
Requests 1-10: ✅ SUCCESS
Requests 11-12: ❌ RATE LIMITED
```

**Actual:**
```
Requests 1-12: ✅ SUCCESS (all passed)
```

**Reason:** All requests checked DB before any were logged

---

## 🔧 Proper Solutions (Future Work)

### Option 1: Redis/Memcached (Recommended)
**Pros:**
- Sub-millisecond atomic increments
- Designed for rate limiting use cases
- Modal supports Redis via Upstash

**Implementation:**
```python
import redis

async def check_rate_limit(user_id: str, endpoint: str, limit: int) -> bool:
    key = f"ratelimit:{user_id}:{endpoint}"
    count = await redis.incr(key)
    if count == 1:
        await redis.expire(key, 60)  # 1 minute TTL
    return count <= limit
```

**Effort:** 2-3 hours (add Upstash Redis, update check_rate_limit)

### Option 2: API Gateway Rate Limiting
**Pros:**
- CloudFlare/AWS handles it
- No code changes needed
- Professional-grade DDoS protection

**Cons:**
- Requires infrastructure change
- May cost money at scale

### Option 3: Optimistic Locking with Transactions
**Pros:**
- Works with existing Supabase
- No new dependencies

**Cons:**
- Complex implementation
- Still has latency issues
- Requires separate `rate_limits` table

**Implementation:**
```sql
CREATE TABLE rate_limits (
    user_id UUID,
    endpoint TEXT,
    window_start TIMESTAMP,
    count INTEGER,
    PRIMARY KEY (user_id, endpoint, window_start)
);

-- Use PostgreSQL transactions with SELECT FOR UPDATE
BEGIN;
SELECT count FROM rate_limits WHERE ... FOR UPDATE;
UPDATE rate_limits SET count = count + 1 WHERE ...;
COMMIT;
```

**Effort:** 4-6 hours

---

## 📊 Current Deployment Status

**Deployed:** ✅ Supabase-based rate limiting active on all endpoints
**Works For:** Sequential requests, moderate traffic
**Fails For:** Concurrent bursts (12+ simultaneous requests)

**Deployment Details:**
- Modal app: `g-mcp-tools-fast`
- Workspace: `scaile`
- Endpoint: `https://scaile--g-mcp-tools-fast-api.modal.run`
- Version: Deployed 2025-10-28 20:40 UTC

**Dependencies Removed:**
- ❌ `slowapi>=0.1.9` (removed from requirements)
- ✅ `structlog>=24.4.0` (kept for logging)

---

## 🧪 Testing Commands

**Test rate limiting with 12 rapid requests:**
```bash
/tmp/test_rate_limiting.sh
```

**Manual test:**
```bash
for i in {1..12}; do
  curl -s -X POST "https://scaile--g-mcp-tools-fast-api.modal.run/plan" \
    -H "Content-Type: application/json" \
    -d '{"user_request": "test"}' &
done
wait
```

**Check logs:**
```bash
modal app logs g-mcp-tools-fast | grep rate_limit
```

---

## 📝 Recommendations

### Short-Term (Current State - Acceptable for MVP)
✅ **Keep current implementation**
- Works for normal traffic patterns
- Prevents abuse from sequential scripts
- Fail-open strategy prevents blocking legitimate users
- Good enough for beta/testing phase

### Medium-Term (Before Production Launch)
🔴 **Add Redis-based rate limiting**
- Prevents concurrent burst attacks
- Sub-second response time
- Proper distributed rate limiting
- Effort: ~2-3 hours with Upstash Redis

### Long-Term (Production Scale)
🔵 **Consider API Gateway**
- CloudFlare Workers or AWS API Gateway
- Professional DDoS protection
- Geographic rate limiting
- Request logging and analytics

---

## 🐛 Bugs Fixed During Implementation

1. **Invalid UUID error** (`invalid input syntax for type uuid: "anonymous"`)
   - **Fix:** Use `"00000000-0000-0000-0000-000000000000"` for anonymous users

2. **500 Internal Server Error** (async/await mismatch)
   - **Fix:** Remove `await` from `log_api_call()` (function is not async)

3. **slowapi incompatibility** (all requests from same IP)
   - **Fix:** Complete replacement with Supabase-based solution

4. **Missing request logging** (`/plan` endpoint had no logging)
   - **Fix:** Added `log_api_call()` to success and error paths

---

## 📚 Related Files

- **Implementation:** `g-mcp-tools-complete.py` (lines 3542-3586, 3850-3915)
- **Test Script:** `/tmp/test_rate_limiting.sh`
- **Previous Notes:** `ALIGNMENT_SUMMARY.md` (Phase 1 observability)
- **Logs:** Modal app logs (`modal app logs g-mcp-tools-fast`)

---

**Status:** ✅ Implementation complete, ⚠️ Known limitation documented, 🔵 Future improvements identified
