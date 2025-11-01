# Database Migration Guide - Workflow System

**Date:** 2025-10-28
**Status:** ⚠️ BLOCKED - Database credentials required
**Goal:** Apply workflow system migrations to enable `/workflow/execute` endpoint

---

## Current Blocker

**Problem:** Cannot apply migrations programmatically due to missing database credentials

**What we tried:**
1. ✅ `supabase link` - Failed with permissions error ("account does not have the necessary privileges")
2. ❌ `supabase db push` - Requires database password (not available)
3. ❌ Python script with supabase-py - Library doesn't support arbitrary SQL execution
4. ❌ Direct PostgreSQL connection - Requires database password

**Root Cause:** The Supabase CLI access token (`~/.supabase/access-token`) doesn't have management API permissions to link projects or push migrations.

---

## Migration Files Ready to Apply

**Location:** `/home/federicodeponte/projects/gtm-power-app-frontend/supabase/migrations/`

**Files (2):**
1. **`20251028100000_workflow_system.sql`** (25,601 bytes)
   - Updates `workflow_templates` table (adds scope, is_system, tags)
   - Creates `prompt_templates` table (reusable prompt library)
   - Creates `tool_definitions` table (extensible tool registry)
   - Creates `user_integrations` table (third-party auth)

2. **`20251028120000_add_backend_tables.sql`** (9,280 bytes)
   - Additional backend support tables

**Total SQL to execute:** ~35KB

---

## Manual Solution (Recommended)

### Option 1: Supabase Dashboard SQL Editor

**Steps:**
1. Go to [Supabase Dashboard](https://supabase.com/dashboard/project/qwymdxrtelvqgdqvtzzv/sql/new)
2. Copy contents of `20251028100000_workflow_system.sql`
3. Paste into SQL Editor
4. Click "Run"
5. Repeat for `20251028120000_add_backend_tables.sql`

**Pros:**
- ✅ No credentials needed (already logged in)
- ✅ Visual confirmation of success
- ✅ Error messages displayed clearly

**Cons:**
- ❌ Manual process (not automated)
- ❌ Requires copy-pasting

**Files to copy:**
```bash
# File 1 (main workflow system)
cat /home/federicodeponte/projects/gtm-power-app-frontend/supabase/migrations/20251028100000_workflow_system.sql

# File 2 (backend tables)
cat /home/federicodeponte/projects/gtm-power-app-frontend/supabase/migrations/20251028120000_add_backend_tables.sql
```

### Option 2: Get Database Password

**Where to find:**
1. Supabase Dashboard → Project Settings → Database
2. Under "Connection string" → Click "Reset database password"
3. Copy new password

**Then run:**
```bash
cd /home/federicodeponte/projects/gtm-power-app-frontend
export PGPASSWORD="your-password-here"
supabase db push
```

**Pros:**
- ✅ Automated migration application
- ✅ Repeatable process
- ✅ Tracks migration history

**Cons:**
- ❌ Requires database password reset (security concern)
- ❌ Need to update password everywhere it's used

### Option 3: Upgrade Supabase CLI Access Token

**Steps:**
1. Log out of Supabase CLI: `supabase logout`
2. Log in with full permissions: `supabase login`
3. Grant management API access when prompted
4. Retry: `supabase link --project-ref qwymdxrtelvqgdqvtzzv`
5. Apply migrations: `supabase db push`

**Pros:**
- ✅ Proper authentication flow
- ✅ Future migrations work seamlessly
- ✅ No database password needed

**Cons:**
- ❌ May require organization admin privileges
- ❌ Current token might be restricted by policy

---

## What Happens After Migration

**Tables Created:**

1. **`prompt_templates`** - Reusable prompt library
   ```sql
   - id, user_id, project_id
   - name, description, scope, is_system, tags
   - template_text, variables
   ```

2. **`tool_definitions`** - Extensible tool registry
   ```sql
   - id, tool_name, tool_type, category
   - display_name, description
   - config (JSONB: internal_fn | http_url | mcp_server)
   - is_active
   ```

3. **`user_integrations`** - Third-party auth
   ```sql
   - id, user_id, integration_name
   - credentials (encrypted JSONB)
   - is_active, last_verified_at
   ```

4. **`workflow_templates` (updated)** - Adds columns
   ```sql
   - scope (system | global | project | session)
   - is_system (boolean)
   - tags (JSONB array)
   ```

**Endpoints Unlocked:**

✅ **`POST /workflow/execute`** - Execute saved workflows
```bash
curl -X POST "https://scaile--g-mcp-tools-fast-api.modal.run/workflow/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_id": "uuid-here",
    "inputs": {"email": "test@example.com"}
  }'
```

**Backend Changes Required:** None - code is already deployed and waiting for tables

---

## Verification After Migration

**Test 1: Check tables exist**
```sql
-- Run in Supabase SQL Editor
SELECT tablename FROM pg_tables WHERE schemaname = 'public'
  AND tablename IN ('prompt_templates', 'tool_definitions', 'user_integrations');
```

**Expected:** 3 rows returned

**Test 2: Test workflow save**
```bash
curl -X POST "https://scaile--g-mcp-tools-fast-api.modal.run/workflow/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "user_request": "Validate email and check platforms",
    "save": true
  }'
```

**Expected:** `saved_id` field populated in response

**Test 3: Test workflow execution**
```bash
curl -X POST "https://scaile--g-mcp-tools-fast-api.modal.run/workflow/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_id": "[id from Test 2]",
    "inputs": {"email": "test@example.com"}
  }'
```

**Expected:** Workflow executes successfully, returns results

---

## Migration Safety

**All migrations are IDEMPOTENT** - Safe to run multiple times:
- `CREATE TABLE IF NOT EXISTS` - Won't fail if table exists
- `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` - Won't fail if column exists
- Foreign keys use `ON DELETE CASCADE` - Automatic cleanup

**Rollback not needed** - Migrations only add tables/columns, don't modify data

---

## Current Workflow System Status

**What Works:**
- ✅ `/tools` - Returns 15 tools
- ✅ `/workflow/generate` - AI-powered workflow generation
- ✅ `/workflow/documentation` - API docs (assumed)

**What's Blocked:**
- ⚠️ `/workflow/execute` - Requires `workflow_templates` table
- ⚠️ Workflow save feature - Requires database tables
- ⚠️ Workflow persistence - Requires database tables

**Code Status:**
- ✅ Backend implementation complete (g-mcp-tools-complete.py lines 1735-2200)
- ✅ WorkflowExecutor class deployed
- ✅ Variable substitution working (tested with /generate)
- ✅ Conditional execution working (tested with /generate)
- ⚠️ Database integration pending migration

---

## Recommendation

**For immediate testing:** Use **Option 1 (Supabase Dashboard)** - fastest path to unblock workflow execution

**For long-term:** Use **Option 3 (Upgrade CLI token)** - enables future automated migrations

**Time estimate:**
- Option 1: 5 minutes (copy-paste SQL)
- Option 2: 10 minutes (reset password + run command)
- Option 3: 15 minutes (re-authenticate + permissions setup)

---

## Files Reference

**Backend repo:** `/home/federicodeponte/gtm-power-app-backend/`
- Backend code: `g-mcp-tools-complete.py`
- Status docs: `WORKFLOW_TESTING_STATUS.md`, `RATE_LIMITING_STATUS.md`

**Frontend repo:** `/home/federicodeponte/projects/gtm-power-app-frontend/`
- Migrations: `supabase/migrations/*.sql`
- Config: `.env.local` (has Supabase URL + keys)

**Project Details:**
- Supabase URL: `https://qwymdxrtelvqgdqvtzzv.supabase.co`
- Project Ref: `qwymdxrtelvqgdqvtzzv`
- Region: AWS us-east-1

---

**Next Step:** Choose Option 1, 2, or 3 above and apply migrations to unblock workflow execution endpoint.
