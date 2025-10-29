# Architecture Alignment Decisions

**Date:** 2025-10-27
**Purpose:** Align `bulk-gpt-minimal` (frontend) with `g-mcp-tools-complete` (backend)
**Status:** ✅ DECISIONS FINALIZED

**See ALIGNMENT_SUMMARY.md for finalized decisions and implementation tasks.**

---

## 🎯 EXECUTIVE SUMMARY

### Top 3 Priorities (Must Fix Before V1)

#### 1. Build `/api/execute` Endpoint 🔴 CRITICAL
- **What**: Generic execution endpoint the frontend expects
- **Why**: Frontend architecture relies on this for all tool calls
- **Current State**: Backend has individual tools, `/bulk`, `/orchestrate` but no generic `/execute`
- **Frontend Expects**:
  ```typescript
  POST /api/execute
  {
    executionId: string
    tool: string
    data: any[]
    params: Record<string, any>
  }
  ```
- **Effort**: 4-6 hours
- **Action**: Add to `g-mcp-tools-complete.py` as wrapper around existing tools
- **Decision**: [ ] Build `/execute` endpoint OR [ ] Change frontend to use `/bulk`?

---

#### 2. Standardize SSE Format 🔴 CRITICAL
- **What**: Change backend from `event` field to `type` field
- **Why**: Frontend won't parse progress events correctly
- **Current Backend**:
  ```json
  { "event": "step_start", "data": {...} }
  { "event": "step_complete", "data": {...} }
  ```
- **Frontend Expects**:
  ```json
  { "type": "progress", "processed": 250, "total": 500, "percentage": 50 }
  { "type": "result", "data": [...], "success": true }
  ```
- **Effort**: 30 minutes
- **Action**: Update `/orchestrate` endpoint SSE events in `g-mcp-tools-complete.py`
- **Decision**: [ ] Change backend to use `type` OR [ ] Update frontend architecture doc?

---

#### 3. Document Sync Strategy 🔴 CRITICAL
- **What**: Pick SessionContext OR event emitters (not both)
- **Why**: Architecture shows redundant sync mechanisms
- **Current Architecture**:
  - Lines 1230-1342: SessionContext for state management
  - Lines 1330-1342: Event emitters for chat ↔ UI sync
- **Redundant Pattern**:
  ```typescript
  // Both of these accomplish the same thing:
  sessionContext.actions.setTool('phone-validation')
  chat.emit('tool:suggest', { tool: 'phone-validation' })
  ```
- **Effort**: 0 hours (documentation only)
- **Action**: Update `ARCHITECTURE.md` to remove event emitters
- **Recommendation**: Use SessionContext only (React-idiomatic)
- **Decision**: [ ] SessionContext only OR [ ] Event emitters only OR [ ] Keep both?

---

## 📋 BACKEND ALIGNMENT ISSUES

### 🔴 Critical Issues

#### Issue #1: Missing `/api/execute` Endpoint
**Location**: Frontend expects this at `ARCHITECTURE.md:1419-1430`
**Current State**: Backend has individual tools but no generic endpoint
**Impact**: Frontend can't call tools generically as designed

**Frontend Architecture Shows**:
```typescript
// POST /api/execute
{
  executionId: string
  tool: string        // 'phone-validation' | 'email-intel' | etc.
  data: any[]         // Array of rows to process
  params: Record<string, any>  // Tool-specific params
}

// Response: SSE stream
{ type: 'progress', processed: 250, total: 500, percentage: 50 }
{ type: 'result', data: [...], success: true }
```

**Backend Currently Has**:
- ✅ Individual tools: `/phone-validation`, `/email-intel`, etc.
- ✅ Batch processing: `/bulk`, `/bulk/auto`
- ✅ AI orchestration: `/orchestrate`
- ❌ Generic execution wrapper: `/execute`

**Options**:
1. **Build `/execute` endpoint** - Wrapper that routes to existing tools
2. **Change frontend** - Use `/bulk/auto` instead of `/execute`
3. **Hybrid** - Use `/orchestrate` for multi-step, `/bulk` for single tool

**Recommendation**: Build `/execute` as thin wrapper (4-6 hours)

**Decision**:
- [ ] Build `/execute` endpoint
- [ ] Change frontend to use `/bulk/auto`
- [ ] Use `/orchestrate` endpoint instead
- [ ] Other: _________________

---

#### Issue #2: SSE Event Format Mismatch
**Location**: `ARCHITECTURE.md:1419` vs `g-mcp-tools-complete.py:2900-2950`
**Current State**: Backend uses `event` field, frontend expects `type`
**Impact**: Frontend won't parse progress updates correctly

**Backend Currently Sends**:
```json
event: plan_init
data: {"plan": [...], "total_steps": 5}

event: step_start
data: {"step": 1, "tool_name": "phone-validation"}

event: step_complete
data: {"step": 1, "success": true, "data": {...}}

event: complete
data: {"success": true, "results": [...]}
```

**Frontend Expects**:
```json
{"type": "progress", "processed": 250, "total": 500, "percentage": 50}
{"type": "result", "data": [...], "success": true}
```

**SSE Standards**: Both formats are valid SSE, but `type` is more common in JSON-based SSE

**Options**:
1. **Change backend** - Use `type` field in JSON data (30 min)
2. **Change frontend** - Parse `event` field from SSE events
3. **Support both** - Backend sends both fields

**Recommendation**: Change backend to use `type` field (matches most SSE libraries)

**Decision**:
- [ ] Change backend to use `type` field
- [ ] Change frontend architecture to use `event` field
- [ ] Support both formats
- [ ] Other: _________________

---

#### Issue #3: Agent Planning Overlap
**Location**: `ARCHITECTURE.md:1348-1477` (Two-Agent System)
**Current State**: Unclear division of responsibilities
**Impact**: Potential duplicate work or planning conflicts

**App Agent Responsibilities** (Lines 1372-1394):
- Conversation & UI guidance
- **Intent detection** ← Does this mean planning?
- Form validation
- No tool execution

**Backend Orchestrator Responsibilities** (Lines 1410-1433):
- Receives execution requests
- **AI orchestration with Planner** ← Complex multi-step planning
- Tool execution on Modal.com
- Progress streaming

**Overlap Question**: Who does planning?
- Simple intent detection ("user wants phone validation") = App Agent
- Complex multi-step plans ("validate phone, then enrich email, then...") = Backend Planner

**Current Backend Implementation**:
- ✅ Planner class (lines 1041-1071 in `g-mcp-tools-complete.py`)
- ✅ Generates multi-step execution plans
- ✅ Uses Gemini 2.0 Flash Exp

**Options**:
1. **Backend only** - App Agent just routes, all planning in backend
2. **Both** - App Agent detects simple intents, backend plans complex workflows
3. **Frontend only** - Remove Planner from backend, App Agent plans everything

**Recommendation**: Keep planning in backend only (already built, tested, working)

**Decision**:
- [ ] Backend planning only (App Agent just routes)
- [ ] Both agents plan (clarify division)
- [ ] Frontend planning only (remove backend Planner)
- [ ] Other: _________________

---

### 🟡 Medium Issues

#### Issue #4: Progress Format (Row-based vs Step-based)
**Location**: `ARCHITECTURE.md:700-745`
**Current State**: Different granularity levels
**Impact**: Minor - step-level sufficient for MVP

**Frontend Expects**:
```json
{
  "type": "progress",
  "processed": 250,      // Rows completed
  "total": 500,          // Total rows
  "percentage": 50,
  "currentRow": {...}    // Data being processed
}
```

**Backend Streams**:
```json
{
  "event": "step_complete",
  "step": 2,             // Steps completed
  "total_steps": 5,      // Total steps
  "tool_name": "email-intel",
  "success": true
}
```

**Options**:
1. **Add row-level progress** - Backend tracks rows within each step (2-3 hours)
2. **Keep step-level** - Defer row-level to V2
3. **Hybrid** - Show steps for multi-tool, rows for single tool

**Recommendation**: Defer to V2 - step-level sufficient for MVP

**Decision**:
- [ ] Add row-level progress now
- [ ] Keep step-level only (defer to V2)
- [ ] Hybrid approach
- [ ] Other: _________________

---

#### Issue #5: Missing `/bulk/results/{batch_id}` Endpoint
**Location**: Implied by batch processing architecture
**Current State**: Results returned in POST response, no separate retrieval
**Impact**: Minor - results available, just not via separate endpoint

**Current Backend**:
- POST `/bulk` returns complete results in response
- POST `/orchestrate` returns complete results in final SSE event
- No separate GET endpoint for retrieving past results

**Options**:
1. **Add endpoint** - GET `/bulk/results/{batch_id}` (30 min)
2. **Document alternative** - Explain results in POST response
3. **Use database** - Query `api_calls` table for past results

**Recommendation**: Defer to V2 - not critical for MVP

**Decision**:
- [ ] Add `/bulk/results/{batch_id}` endpoint
- [ ] Document that results are in POST response
- [ ] Query database for past results
- [ ] Other: _________________

---

## 📋 FRONTEND ARCHITECTURE ISSUES

### 🔴 Critical Issues (Fix Before Implementation)

#### Issue #6: Dual Sync Mechanism (SessionContext + Event Emitters)
**Location**: `ARCHITECTURE.md:1230-1342`
**Impact**: Redundant code, potential sync bugs, increased complexity

**Problem**: Architecture shows TWO ways to sync state:

**Method 1: SessionContext** (Lines 1231-1281)
```typescript
interface SessionContextValue {
  state: SessionState      // Single source of truth
  actions: SessionActions  // State update methods
}

// Usage:
const { state, actions } = useSessionContext()
actions.setTool('phone-validation')
actions.updateProgress({ processed: 100, total: 500 })
```

**Method 2: Event Emitters** (Lines 1330-1342)
```typescript
// Chat → UI communication
chat.emit('tool:suggest', { tool: 'phone-validation' })
chat.emit('file:uploaded', { fileId, rows })

// UI → Chat communication
ui.emit('execution:start', { executionId, tool })
ui.emit('execution:complete', { success, results })
```

**Why This Is Problematic**:
- Redundant: Both accomplish the same state sync
- Bug-prone: State can get out of sync between Context and events
- Complexity: Developers must update two systems
- Not React-idiomatic: Context is the standard pattern

**Recommendation**: Remove event emitters, use SessionContext only

**Decision**:
- [ ] SessionContext only (remove event emitters)
- [ ] Event emitters only (remove SessionContext)
- [ ] Keep both (explain why)
- [ ] Other: _________________

**Action if SessionContext only**: Update `ARCHITECTURE.md` lines 1330-1342

---

#### Issue #7: Multiple Sessions Contradiction
**Location**: `ARCHITECTURE.md:435-463` (Phase 8) vs `ARCHITECTURE.md:1231-1242` (SessionContext)
**Impact**: Can't implement as currently designed

**Phase 8 Architecture** (Lines 435-463):
- Multiple concurrent sessions with dropdown selector
- Background execution with SSE connections per session
- Session status indicators (🟢 running, 🔵 completed, ⚪ queued, 🔴 failed)
- **Requires**: Array of sessions

**SessionContext Implementation** (Lines 1231-1242):
```typescript
interface SessionState {
  id: string              // Singular session
  currentTab: TabName
  tool: ToolName | null
  // ... rest of state
}

const SessionContext = createContext<SessionContextValue>(null)
```

**Contradiction**: SessionContext holds ONE session, but Phase 8 needs MULTIPLE

**Options**:
1. **Single session for V1** - Defer multiple sessions to Phase 8 (V2)
2. **Implement SessionManager** - Array of sessions in context
3. **Separate contexts** - `SessionContext` (current) + `SessionsContext` (list)

**Recommendation**: Single session for V1, defer Phase 8 to V2 (reduces complexity)

**Decision**:
- [ ] Single session for V1 (defer Phase 8 to V2)
- [ ] Implement SessionManager now (300 LOC, +2 weeks)
- [ ] Separate contexts (current + list)
- [ ] Other: _________________

**Action if single session**: Update `ARCHITECTURE.md` to clarify V1 = single session

---

#### Issue #8: Endpoint Naming Conflict
**Location**: `ARCHITECTURE.md:284` vs `ARCHITECTURE.md:1419`
**Impact**: Confusion about which endpoint to build/call

**Line 284** (Phase 3: TASK Config):
```typescript
// Call /api/enrich with { tool, params, data }
const response = await fetch('/api/enrich', {
  method: 'POST',
  body: JSON.stringify({ tool, params, data })
})
```

**Line 1419** (Backend Integration):
```typescript
// POST /api/execute
{
  executionId: string
  tool: string
  data: any[]
  params: Record<string, any>
}
```

**Question**: Is it `/api/enrich` or `/api/execute`?

**Options**:
1. **Use `/api/execute`** - More generic, matches execution semantics
2. **Use `/api/enrich`** - Specific to enrichment tools
3. **Both** - `/enrich` for enrichment, `/execute` for all tools

**Recommendation**: Use `/api/execute` (more generic, extensible)

**Decision**:
- [ ] `/api/execute` (update line 284)
- [ ] `/api/enrich` (update line 1419)
- [ ] Both endpoints (clarify use cases)
- [ ] Other: _________________

**Action**: Update `ARCHITECTURE.md` line 284 or 1419 to match decision

---

### 🟡 Medium Issues (Simplify for V1)

#### Issue #9: Two Execution Modes
**Location**: `ARCHITECTURE.md:700-745`
**Impact**: Complex state management, user confusion

**Mode 1: Chat-Driven Execution** (Lines 705-725)
- User configures via chat conversation
- Progress shown in BOTH chat panel AND UI
- AI narrates progress ("Processing row 100 of 500...")
- State synced between chat and UI

**Mode 2: Direct TASK Tab** (Lines 730-745)
- User configures directly in TASK tab
- Progress shown ONLY in RHS panel
- Chat panel remains independent
- No AI narration

**Complexity**:
```typescript
// State must track execution mode
interface SessionState {
  execution: {
    mode: 'chat-driven' | 'direct-task'  // Which mode?
    showInChat: boolean                   // Show progress in chat?
    aiNarration: boolean                  // Enable narration?
  }
}
```

**Options**:
1. **Always Mode 1** - Always show progress in both panels (simpler)
2. **Keep both modes** - Give users choice (more complex)
3. **Always Mode 2** - Never show in chat (defeats chat-first purpose)

**Recommendation**: Always Mode 1 (simpler, aligns with "chat-first" vision)

**Decision**:
- [ ] Always Mode 1 (remove Mode 2)
- [ ] Keep both modes (explain use case)
- [ ] Always Mode 2 (explain why)
- [ ] Other: _________________

**Action if Mode 1 only**: Update `ARCHITECTURE.md` lines 730-745

---

#### Issue #10: Three-Tier Storage Over-Engineered
**Location**: `ARCHITECTURE.md:367-384`
**Impact**: 3× APIs, 3× UI, user confusion about where files go

**Current Architecture**:
1. **Global Storage** - Shared across all users (admin uploaded)
2. **Project Storage** - Shared within project/team
3. **Session Storage** - Ephemeral, deleted after session

**Complexity**:
- 3 storage APIs to build
- 3 UI sections (Global Library, My Projects, Session Files)
- User must decide: Global? Project? Session?
- Project requires multi-tenant (Phase 11)

**Options**:
1. **Session + User (2 tiers)** - Simplify for V1
   - Session storage (ephemeral, auto-delete)
   - User storage (persistent, saved to account)
2. **Keep all 3 tiers** - Implement as designed
3. **Session only** - Simplest, but lose persistence

**Recommendation**: Session + User for V1, add Projects in V2 (Phase 11)

**Decision**:
- [ ] Session + User only (defer Projects to V2)
- [ ] Keep all 3 tiers
- [ ] Session only (no persistence)
- [ ] Other: _________________

**Action if 2 tiers**: Update `ARCHITECTURE.md` lines 367-384

---

#### Issue #11: AI Narration Logic Undefined
**Location**: `ARCHITECTURE.md:800` and throughout Phase 4
**Impact**: Can't implement without clear spec

**Architecture Says**: "AI narrates at milestones"

**Undefined**:
- What defines a "milestone"?
- Template-based ("Processing row {{current}} of {{total}}") or dynamic?
- How often? Every 10%? Every 100 rows?
- What if processing is fast (<5 seconds)?

**Options**:
1. **Hardcoded templates** at fixed milestones (25%, 50%, 75%, 100%)
   ```typescript
   const templates = {
     25: "Great progress! I've completed {{processed}} of {{total}} rows (25%).",
     50: "Halfway there! {{processed}} of {{total}} rows processed.",
     75: "Almost done! Just {{remaining}} rows left.",
     100: "All done! Successfully processed {{total}} rows."
   }
   ```
2. **Dynamic AI generation** - Gemini generates contextual narration
3. **Time-based** - Narrate every 10 seconds
4. **Hybrid** - Templates for milestones + dynamic for errors

**Recommendation**: Hardcoded templates at 25%, 50%, 75%, 100% (simple, predictable)

**Decision**:
- [ ] Hardcoded templates at fixed percentages
- [ ] Dynamic AI generation (Gemini API calls)
- [ ] Time-based narration
- [ ] Hybrid approach
- [ ] Other: _________________

**Action**: Add narration spec to `ARCHITECTURE.md` Phase 4

---

#### Issue #12: Intent Classification Method Unclear
**Location**: `ARCHITECTURE.md:1121` (Phase 2)
**Impact**: Can't implement without knowing approach

**Architecture Says**: "Intent detection for tab navigation"

**Unclear**:
- Keyword matching ("upload" → INPUT tab)?
- Gemini API call for classification?
- Regex patterns?
- Hybrid approach?

**Example Mentioned** (Line 1135):
```typescript
// Seems to extract context, not just detect intent
{
  intent: 'phone_validation',
  context: {
    csvColumns: ['phone', 'name', 'email'],
    sampleData: [...],
    userGoal: 'validate and format phone numbers'
  }
}
```

**Options**:
1. **Gemini API** - More accurate, handles ambiguity
   ```typescript
   const intent = await gemini.classify(userMessage, {
     options: ['input', 'task', 'output', 'profile']
   })
   ```
2. **Keyword matching** - Simpler, faster, less accurate
   ```typescript
   if (message.includes('upload') || message.includes('file')) {
     return 'input'
   }
   ```
3. **Hybrid** - Keywords for obvious cases, Gemini for ambiguous

**Recommendation**: Gemini API (better UX, already integrated)

**Decision**:
- [ ] Gemini API classification
- [ ] Keyword matching
- [ ] Hybrid approach
- [ ] Other: _________________

**Action**: Update `ARCHITECTURE.md` Phase 2 with implementation method

---

#### Issue #13: Bidirectional File Upload UX Confusion
**Location**: `ARCHITECTURE.md:386-399`
**Impact**: Users may lose files unknowingly

**Current Design**:
- **INPUT tab** → Defaults to User storage (persistent)
- **TASK tab** → Defaults to Session storage (ephemeral, deleted after session)

**Problem**: User uploads file in TASK tab, doesn't realize it's temporary, loses it after session

**Example Flow**:
1. User in TASK tab, clicks "Upload CSV"
2. File saved to Session storage (ephemeral)
3. User executes tool, sees results
4. User closes browser
5. Session deleted, file LOST ❌

**Options**:
1. **Prompt after execution** - "Save results to account? [Yes] [No]"
2. **Change default** - TASK tab also saves to User storage
3. **Remove session storage** - Always persistent (Issue #10 related)
4. **Clear warning** - "⚠️ Files in TASK tab are temporary"

**Recommendation**: Prompt after execution completes (best UX)

**Decision**:
- [ ] Prompt "Save to account?" after execution
- [ ] Change TASK tab default to User storage
- [ ] Remove session storage entirely
- [ ] Add clear warning only
- [ ] Other: _________________

**Action**: Update `ARCHITECTURE.md` lines 386-399 with UX flow

---

## ✅ WHAT'S ALREADY ALIGNED (No Action Needed)

### Backend Features Implemented & Aligned

✅ **Scheduled Jobs** (`saved_queries` table)
- CREATE, READ, UPDATE, DELETE saved jobs
- Schedule presets (daily, weekly, monthly)
- `is_scheduled`, `schedule_preset`, `next_run_at` fields
- Manual execution via `/jobs/saved/{id}/run`

✅ **API Call Logging** (`api_calls` table)
- Logs every tool execution
- Tracks: tool_name, tool_type, user_id, success, processing_ms, error_message
- Input/output data stored for debugging
- Created_at timestamp for analytics

✅ **Quota Management** (`check_and_increment_quota` RPC)
- User-based monthly quotas
- Automatic monthly resets
- Quota limit enforcement
- Usage tracking in `user_quotas` table

✅ **Bulk Endpoints**
- POST `/bulk` - Explicit tool selection
- POST `/bulk/auto` - Auto-detect enrichment tools
- Batch processing with async workers
- Status tracking with batch IDs

✅ **Tool Schema Alignment**
All 14 tools registered in backend match architecture:

**Enrichment Tools (9)**:
1. email-intel
2. email-finder
3. email-pattern
4. phone-validation
5. company-data
6. tech-stack
7. whois
8. github-intel
9. scrape (web scraper)

**GTM Tools (5)**:
1. web-search
2. deep-research
3. blog-create
4. aeo-health-check
5. aeo-mentions

✅ **Supabase Integration**
- Shared database connection
- JWT authentication ready
- RLS policies structure defined
- Edge Functions compatible

✅ **Error Handling & Retry Logic**
- ErrorHandler with exponential backoff
- Retry attempts (max 3)
- Fallback mechanisms
- Error classification by type

---

## 🚀 RECOMMENDED V1 SCOPE (Simplifications)

### ✅ Keep for V1

**Core Architecture**:
- Two-agent architecture (App Agent + Backend Orchestrator)
- SessionContext for state management (remove event emitters)
- SSE progress streaming
- Single active session (defer multiple to Phase 8)
- Intent-based navigation with Gemini API
- assistant-ui library for chat UI

**Storage**:
- Session storage (ephemeral)
- User storage (persistent)
- Defer Project storage to V2

**Progress & Narration**:
- Step-level progress (defer row-level to V2)
- Hardcoded narration templates at milestones
- Always show progress in both chat + UI (Mode 1 only)

**Features**:
- All 14 tools (9 enrichment + 5 GTM)
- Scheduled jobs
- API logging
- Quota management

---

### ⏸️ Defer to V2

**Phase 8: Multiple Sessions** (300 LOC, +2 weeks)
- SessionManager implementation
- Dropdown selector UI
- Background execution per session
- Session status indicators

**Phase 10: Token Tracking** (160 LOC, +1 week)
- Input/output token counting
- Usage dashboard
- Time range filters

**Phase 11: Multi-Tenant** (220 LOC, +3 weeks)
- Real authentication (currently mock)
- RLS policies
- Team/project management

**Advanced Features**:
- Row-level progress granularity
- `/bulk/results/{batch_id}` endpoint
- Project-level storage (3rd tier)
- Dynamic AI narration (beyond templates)
- Connected apps (Airtable, Sheets, Zapier)

---

### ❌ Remove from Architecture

**Redundant Patterns**:
- Event emitters (use SessionContext only)
- Mode 2 execution (always show in both panels)
- `/api/enrich` endpoint naming (use `/api/execute`)

**Over-Engineering**:
- Three-tier storage (reduce to 2 tiers)
- Complex intent detection (use Gemini API, not keywords)

---

## 📝 IMMEDIATE ACTION ITEMS

### Backend Tasks (`g-mcp-tools-complete.py`)

**Priority 1: Build `/api/execute` Endpoint** (4-6 hours)
```python
@app.post("/execute")
async def execute_tool(request: ExecuteRequest, user_id: str = Depends(verify_jwt)):
    """
    Generic tool execution endpoint.
    Routes to appropriate tool based on request.tool parameter.
    """
    tool_name = request.tool
    params = request.params
    data = request.data

    # Route to existing tool implementation
    if tool_name in TOOLS:
        result = await execute_tool_internal(tool_name, params, data, user_id)
        return result
    else:
        raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")
```

**Files to modify**:
- `g-mcp-tools-complete.py:2800-2850` (add endpoint)
- Add `ExecuteRequest` schema
- Add routing logic to existing tools

**Testing**:
```bash
curl -X POST "https://scaile--g-mcp-tools-fast-api.modal.run/execute" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -d '{
    "executionId": "exec_123",
    "tool": "phone-validation",
    "data": [{"+14155551234"}],
    "params": {}
  }'
```

---

**Priority 2: Standardize SSE Format** (30 minutes)
```python
# Change from:
yield f"event: step_complete\ndata: {json.dumps(data)}\n\n"

# To:
yield f"data: {json.dumps({'type': 'step_complete', **data})}\n\n"
```

**Files to modify**:
- `g-mcp-tools-complete.py:2900-2950` (orchestrate endpoint)
- Update all SSE event yields
- Test with frontend SSE parsing

**Before**:
```
event: plan_init
data: {"plan": [...], "total_steps": 5}

event: step_start
data: {"step": 1, "tool_name": "phone-validation"}
```

**After**:
```
data: {"type": "plan_init", "plan": [...], "total_steps": 5}

data: {"type": "step_start", "step": 1, "tool_name": "phone-validation"}
```

---

### Frontend Tasks (`ARCHITECTURE.md`)

**Priority 1: Remove Event Emitter References** (30 minutes)
- Lines 1330-1342: Remove event emitter documentation
- Clarify SessionContext as single state management approach
- Update examples to only show SessionContext usage

**Priority 2: Clarify Endpoint Naming** (5 minutes)
- Line 284: Change `/api/enrich` to `/api/execute`
- Ensure all examples use `/api/execute`

**Priority 3: Document V1 Simplifications** (1 hour)
- Add "V1 Scope" section
- Clarify single session for V1 (Phase 8 deferred)
- Reduce storage to 2 tiers (Session + User)
- Remove Mode 2 execution mode

**Priority 4: Add AI Narration Spec** (30 minutes)
- Define hardcoded templates at 25%, 50%, 75%, 100%
- Specify when narration triggers
- Add example implementation

**Priority 5: Clarify Intent Detection** (15 minutes)
- Specify Gemini API for classification
- Remove keyword matching references
- Add implementation example

**Priority 6: Add Upload UX Prompt** (15 minutes)
- Specify "Save to account?" prompt after execution
- Clarify Session vs User storage behavior
- Add UX flow diagram

**Priority 7: Update Storage Architecture** (30 minutes)
- Remove Project storage references
- Simplify to Session + User only
- Update UI mockups

---

## 📊 DECISION MATRIX

| # | Issue | Severity | Component | Fix Now? | Defer to V2? | Effort |
|---|-------|----------|-----------|----------|--------------|--------|
| 1 | Missing `/execute` endpoint | 🔴 CRITICAL | Backend | ✅ Yes | - | 4-6h |
| 2 | SSE format (`type` vs `event`) | 🔴 CRITICAL | Backend | ✅ Yes | - | 30m |
| 3 | Agent planning overlap | 🟡 MEDIUM | Both | Clarify | - | Discussion |
| 4 | Progress format (row vs step) | 🟡 MEDIUM | Backend | - | ✅ Yes | 2-3h |
| 5 | Missing `/bulk/results` | 🟢 LOW | Backend | - | ✅ Yes | 30m |
| 6 | Dual sync mechanism | 🔴 CRITICAL | Frontend | ✅ Yes | - | 30m (docs) |
| 7 | Multiple sessions contradiction | 🔴 CRITICAL | Frontend | Clarify | ✅ Yes (Phase 8) | Discussion |
| 8 | Endpoint naming conflict | 🔴 CRITICAL | Frontend | ✅ Yes | - | 5m (docs) |
| 9 | Two execution modes | 🟡 MEDIUM | Frontend | ✅ Yes | - | 30m (docs) |
| 10 | Three-tier storage | 🟡 MEDIUM | Frontend | ✅ Yes | ✅ Yes (Projects) | 30m (docs) |
| 11 | AI narration undefined | 🟡 MEDIUM | Frontend | ✅ Yes | - | 30m (docs) |
| 12 | Intent classification unclear | 🟡 MEDIUM | Frontend | ✅ Yes | - | 15m (docs) |
| 13 | Upload UX confusion | 🟡 MEDIUM | Frontend | ✅ Yes | - | 15m (docs) |

**Total Estimated Effort**:
- Backend: 4-6 hours (critical path)
- Frontend: 3-4 hours (documentation updates)
- **Grand Total**: ~8-10 hours

---

## 📌 DECISION TRACKING

### Backend Decisions

**Decision #1: `/api/execute` Endpoint**
- [ ] Build `/execute` endpoint (4-6h)
- [ ] Change frontend to use `/bulk/auto`
- [ ] Use `/orchestrate` instead
- [ ] Other: _________________

**Notes**: _________________________________________________

---

**Decision #2: SSE Event Format**
- [ ] Change backend to use `type` field (30m)
- [ ] Change frontend to use `event` field
- [ ] Support both formats
- [ ] Other: _________________

**Notes**: _________________________________________________

---

**Decision #3: Agent Planning**
- [ ] Backend planning only (App Agent just routes)
- [ ] Both agents plan (clarify division)
- [ ] Frontend planning only
- [ ] Other: _________________

**Notes**: _________________________________________________

---

**Decision #4: Progress Granularity**
- [ ] Add row-level progress now (2-3h)
- [ ] Keep step-level only (defer to V2)
- [ ] Hybrid approach
- [ ] Other: _________________

**Notes**: _________________________________________________

---

**Decision #5: `/bulk/results` Endpoint**
- [ ] Add endpoint (30m)
- [ ] Document POST response alternative
- [ ] Query database for past results
- [ ] Other: _________________

**Notes**: _________________________________________________

---

### Frontend Decisions

**Decision #6: Sync Mechanism**
- [ ] SessionContext only (remove event emitters)
- [ ] Event emitters only (remove SessionContext)
- [ ] Keep both (explain why)
- [ ] Other: _________________

**Notes**: _________________________________________________

---

**Decision #7: Multiple Sessions**
- [ ] Single session for V1 (defer Phase 8 to V2)
- [ ] Implement SessionManager now (300 LOC, +2 weeks)
- [ ] Separate contexts (current + list)
- [ ] Other: _________________

**Notes**: _________________________________________________

---

**Decision #8: Endpoint Naming**
- [ ] `/api/execute` (update line 284)
- [ ] `/api/enrich` (update line 1419)
- [ ] Both endpoints (clarify use cases)
- [ ] Other: _________________

**Notes**: _________________________________________________

---

**Decision #9: Execution Modes**
- [ ] Always Mode 1 (remove Mode 2)
- [ ] Keep both modes (explain use case)
- [ ] Always Mode 2 (explain why)
- [ ] Other: _________________

**Notes**: _________________________________________________

---

**Decision #10: Storage Tiers**
- [ ] Session + User only (defer Projects to V2)
- [ ] Keep all 3 tiers
- [ ] Session only (no persistence)
- [ ] Other: _________________

**Notes**: _________________________________________________

---

**Decision #11: AI Narration**
- [ ] Hardcoded templates at fixed percentages
- [ ] Dynamic AI generation (Gemini API calls)
- [ ] Time-based narration
- [ ] Hybrid approach
- [ ] Other: _________________

**Notes**: _________________________________________________

---

**Decision #12: Intent Classification**
- [ ] Gemini API classification
- [ ] Keyword matching
- [ ] Hybrid approach
- [ ] Other: _________________

**Notes**: _________________________________________________

---

**Decision #13: Upload UX**
- [ ] Prompt "Save to account?" after execution
- [ ] Change TASK tab default to User storage
- [ ] Remove session storage entirely
- [ ] Add clear warning only
- [ ] Other: _________________

**Notes**: _________________________________________________

---

## 📅 TIMELINE RECOMMENDATION

### Week 1: Backend Alignment (4-6 hours)
- [ ] Build `/api/execute` endpoint
- [ ] Standardize SSE format
- [ ] Deploy and test

### Week 1: Frontend Documentation (3-4 hours)
- [ ] Remove event emitter references
- [ ] Clarify endpoint naming
- [ ] Document V1 simplifications
- [ ] Add narration spec
- [ ] Clarify intent detection
- [ ] Add upload UX prompt
- [ ] Update storage architecture

### Week 2-5: V1 Implementation
- [ ] Phase 1: Foundation with assistant-ui (250 LOC)
- [ ] Phase 2: Navigation (200 LOC)
- [ ] Phase 3: TASK Config (300 LOC)
- [ ] Phase 4: TASK Execution (250 LOC)
- [ ] Phase 5: Storage System (300 LOC)
- [ ] Phase 6: OUTPUT & Analytics (100 LOC)
- [ ] Phase 7: Polish & Profile (50 LOC)

**Total V1**: 1,450 LOC + 150 LOC assistant-ui = ~4 weeks

---

## 🔗 REFERENCES

**Architecture Documentation**:
- `/tmp/bulk-gpt-minimal/ARCHITECTURE.md` (3,908 lines)
- `/tmp/bulk-gpt-minimal/docs/sessions/20251027_architecture_completion.md`

**Backend Implementation**:
- `/home/federicodeponte/zola-crawl4ai-20251023_133846/modal/g-mcp-tools-complete.py`
- Current deployment: `https://scaile--g-mcp-tools-fast-api.modal.run`

**Frontend Codebase**:
- `/tmp/bulk-gpt-minimal/` (chat-first GTM app)

---

**Status**: 🟡 DRAFT - Awaiting decisions on 13 alignment issues

**Next Steps**: Review, make decisions, update this document with chosen options
