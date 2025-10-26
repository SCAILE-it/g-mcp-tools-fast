# g-mcp-tools-fast - Production-Ready Enrichment API

**Enterprise-grade data intelligence API with 9 powerful enrichment tools**

🚀 **Status:** Production-Ready | SaaS-Ready | Fully Deployed
🔗 **Live Endpoint:** `https://scaile--g-mcp-tools-fast-api.modal.run`
📚 **Interactive Docs:** [Swagger UI](https://scaile--g-mcp-tools-fast-api.modal.run/docs) | [ReDoc](https://scaile--g-mcp-tools-fast-api.modal.run/redoc)

---

## 🎯 Overview

A complete data enrichment API built on Modal.com, combining AI-powered web scraping with 8 specialized intelligence tools. Perfect for sales intelligence, market research, lead enrichment, and data validation.

### Key Features

✅ **9 Enrichment Tools** - Web scraping, email intel, company data, phone validation, and more
✅ **AI-Powered Extraction** - Uses Gemini 2.5 Flash for intelligent data extraction
✅ **Production-Ready** - Authentication, health checks, comprehensive error handling
✅ **Auto-Scaling** - Serverless architecture handles traffic spikes automatically
✅ **24-Hour Cache** - Reduces costs and improves response times
✅ **OpenAPI Docs** - Swagger/ReDoc for easy integration
✅ **Type-Safe** - Pydantic models for all inputs/outputs

---

## 🛠️ Available Tools

### 1. **Web Scraper** (`/scrape`)
Extract structured data from any website using natural language prompts.

**Capabilities:**
- AI-powered extraction with Gemini 2.5 Flash
- Multi-page scraping with auto-discovery
- Custom JSON schema support
- Link extraction
- 24-hour intelligent caching

**Example:**
```bash
curl -X POST https://scaile--g-mcp-tools-fast-api.modal.run/scrape \
  -H 'Content-Type: application/json' \
  -d '{
    "url": "https://anthropic.com",
    "prompt": "Extract the company mission and product names",
    "max_pages": 1
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "company_mission": "Build safe, beneficial AI...",
    "product_names": ["Claude", "Claude Code", "Opus", "Sonnet", "Haiku"]
  },
  "metadata": {
    "extraction_time": 10.31,
    "pages_scraped": 1,
    "model": "gemini-2.5-flash"
  }
}
```

---

### 2. **Email Intel** (`/email-intel`)
Check which platforms an email is registered on (holehe).

```bash
curl -X POST https://scaile--g-mcp-tools-fast-api.modal.run/email-intel \
  -H 'Content-Type: application/json' \
  -d '{"email": "user@example.com"}'
```

---

### 3. **Email Finder** (`/email-finder`)
Find email addresses associated with a domain (theHarvester).

```bash
curl -X POST https://scaile--g-mcp-tools-fast-api.modal.run/email-finder \
  -H 'Content-Type: application/json' \
  -d '{"domain": "anthropic.com", "limit": 10}'
```

---

### 4. **Company Data** (`/company-data`)
Get company registration and corporate information (OpenCorporates).

```bash
curl -X POST https://scaile--g-mcp-tools-fast-api.modal.run/company-data \
  -H 'Content-Type: application/json' \
  -d '{"companyName": "Anthropic", "domain": "anthropic.com"}'
```

---

### 5. **Phone Validation** (`/phone-validation`)
Validate phone numbers with carrier, location, and line type info.

```bash
curl -X POST https://scaile--g-mcp-tools-fast-api.modal.run/phone-validation \
  -H 'Content-Type: application/json' \
  -d '{"phoneNumber": "+14155552671", "defaultCountry": "US"}'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "valid": true,
    "formatted": {
      "e164": "+14155552671",
      "international": "+1 415-555-2671",
      "national": "(415) 555-2671"
    },
    "country": "San Francisco, CA",
    "carrier": "Unknown",
    "lineType": "FIXED_LINE_OR_MOBILE",
    "lineTypeCode": 2
  }
}
```

---

### 6. **Tech Stack** (`/tech-stack`)
Detect technologies and frameworks used by a website.

```bash
curl -X POST https://scaile--g-mcp-tools-fast-api.modal.run/tech-stack \
  -H 'Content-Type: application/json' \
  -d '{"domain": "anthropic.com"}'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "domain": "anthropic.com",
    "technologies": [
      {"name": "Next.js", "category": "Framework"},
      {"name": "cloudflare", "category": "Web Server"}
    ],
    "totalFound": 2
  }
}
```

---

### 7. **Email Pattern** (`/email-pattern`)
Generate common email patterns for a domain.

```bash
curl -X POST https://scaile--g-mcp-tools-fast-api.modal.run/email-pattern \
  -H 'Content-Type: application/json' \
  -d '{"domain": "anthropic.com", "firstName": "John", "lastName": "Doe"}'
```

---

### 8. **WHOIS** (`/whois`)
Look up domain registration information.

```bash
curl -X POST https://scaile--g-mcp-tools-fast-api.modal.run/whois \
  -H 'Content-Type: application/json' \
  -d '{"domain": "anthropic.com"}'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "domain": "anthropic.com",
    "registrar": "MarkMonitor, Inc.",
    "creationDate": "2001-10-02",
    "expirationDate": "2033-10-02",
    "nameServers": ["ISLA.NS.CLOUDFLARE.COM", "RANDY.NS.CLOUDFLARE.COM"]
  }
}
```

---

### 9. **GitHub Intel** (`/github-intel`)
Analyze GitHub user profiles and repositories.

```bash
curl -X POST https://scaile--g-mcp-tools-fast-api.modal.run/github-intel \
  -H 'Content-Type: application/json' \
  -d '{"username": "anthropics"}'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "username": "anthropics",
    "name": "Anthropic",
    "location": "United States of America",
    "publicRepos": 54,
    "followers": 14565,
    "languages": {
      "Python": 6,
      "TypeScript": 3,
      "JavaScript": 1
    }
  }
}
```

---

## 🔐 Authentication

The API supports optional API key authentication via the `x-api-key` header.

### Enable Authentication

1. **Create Modal secret:**
```bash
modal secret create modal-api-key MODAL_API_KEY=your-secret-key-here
```

2. **Redeploy the API:**
```bash
./DEPLOY_G_MCP_TOOLS.sh
```

3. **Include API key in requests:**
```bash
curl -X POST https://scaile--g-mcp-tools-fast-api.modal.run/scrape \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: your-secret-key-here' \
  -d '{"url": "https://example.com", "prompt": "Extract data"}'
```

**Note:** If `MODAL_API_KEY` is not set, the API is publicly accessible (useful for development).

---

## 🚀 Deployment

### Prerequisites

1. **Install Modal CLI:**
```bash
pip install modal
```

2. **Authenticate:**
```bash
modal setup
```

3. **Create Gemini API secret:**
```bash
modal secret create gemini-secret GOOGLE_GENERATIVE_AI_API_KEY=your-gemini-key
```

### Deploy

```bash
chmod +x DEPLOY_G_MCP_TOOLS.sh
./DEPLOY_G_MCP_TOOLS.sh
```

Or manually:
```bash
modal deploy g-mcp-tools-complete.py
```

---

## 🏥 Health Check

Monitor API status:

```bash
curl https://scaile--g-mcp-tools-fast-api.modal.run/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "g-mcp-tools-fast",
  "version": "1.0.0",
  "tools": 9,
  "timestamp": "2025-10-26T17:30:00.000000Z"
}
```

---

## 📊 Response Format

All endpoints follow a consistent response format:

### Success Response
```json
{
  "success": true,
  "data": { ... },
  "metadata": {
    "source": "tool-name",
    "timestamp": "2025-10-26T17:30:00.000000Z"
  }
}
```

### Error Response
```json
{
  "success": false,
  "error": "Error message",
  "metadata": {
    "source": "tool-name",
    "timestamp": "2025-10-26T17:30:00.000000Z"
  }
}
```

---

## 💰 Cost Optimization

The API includes several cost-saving features:

1. **24-Hour Cache** - Repeated requests return cached results
2. **Timeouts** - Prevents runaway processes (30s default, 120s max)
3. **Container Idle Timeout** - Containers shut down after 120s of inactivity
4. **Efficient Resource Usage** - Only runs when needed

**Estimated costs (Modal pricing):**
- Web scraping: ~$0.001 per request
- Other tools: ~$0.0001 per request
- Cache hits: $0 (served from memory)

---

## 🧪 Testing

### Run All Tests
```bash
# Test all 9 endpoints
./test-all-endpoints.sh
```

### Individual Endpoint Tests
```bash
# Email pattern
curl -X POST https://scaile--g-mcp-tools-fast-api.modal.run/email-pattern \
  -H 'Content-Type: application/json' \
  -d '{"domain": "anthropic.com"}'

# Phone validation
curl -X POST https://scaile--g-mcp-tools-fast-api.modal.run/phone-validation \
  -H 'Content-Type: application/json' \
  -d '{"phoneNumber": "+14155552671"}'

# GitHub intel
curl -X POST https://scaile--g-mcp-tools-fast-api.modal.run/github-intel \
  -H 'Content-Type: application/json' \
  -d '{"username": "anthropics"}'
```

---

## 📈 SaaS Readiness Checklist

- [x] **Health Check Endpoint** - `/health` for monitoring
- [x] **API Authentication** - Optional `x-api-key` header
- [x] **OpenAPI Documentation** - Swagger UI + ReDoc
- [x] **Error Handling** - Comprehensive error responses
- [x] **Input Validation** - Pydantic models
- [x] **Rate Limiting** - Handled by Modal platform
- [x] **Monitoring** - Modal dashboard + logs
- [x] **Auto-Scaling** - Serverless architecture
- [x] **Cost Optimization** - Caching + timeouts
- [x] **Type Safety** - TypeScript-style typing

### Ready to Sell As:
✅ B2B SaaS API
✅ Data Enrichment Service
✅ Lead Intelligence Platform
✅ Market Research Tool

---

## 🔧 Monitoring & Logs

### View Logs
```bash
modal app logs g-mcp-tools-fast --follow
```

### Check App Status
```bash
modal app list | grep g-mcp-tools
```

### View Secrets
```bash
modal secret list
```

---

## 🏗️ Architecture

```
Client Request
    ↓
FastAPI (Modal ASGI)
    ↓
Authentication Check (optional)
    ↓
Input Validation (Pydantic)
    ↓
Cache Check (24h TTL)
    ↓ (cache miss)
Tool Execution
    ├→ Web Scraper (crawl4ai + Gemini)
    ├→ Email Intel (holehe)
    ├→ Email Finder (theHarvester)
    ├→ Company Data (OpenCorporates API)
    ├→ Phone Validation (libphonenumber)
    ├→ Tech Stack (custom detection)
    ├→ Email Pattern (pattern generation)
    ├→ WHOIS (python-whois)
    └→ GitHub Intel (GitHub API)
    ↓
Cache Result
    ↓
JSON Response
```

---

## 📝 License

See parent repository for license information.

---

## 🤝 Support

- **Documentation:** [Swagger UI](https://scaile--g-mcp-tools-fast-api.modal.run/docs)
- **Issues:** Report via GitHub Issues
- **Modal Support:** [modal.com/docs](https://modal.com/docs)

---

## 🎯 Use Cases

### Sales Intelligence
- Enrich lead data with company info
- Find contact emails and phone numbers
- Validate contact information

### Market Research
- Scrape competitor websites
- Analyze tech stacks
- Track company changes via WHOIS

### Developer Intelligence
- Analyze GitHub profiles
- Detect technologies used
- Research developer ecosystems

### Data Validation
- Validate phone numbers
- Verify email patterns
- Check domain registrations

---

**Built with:** Modal.com | FastAPI | Gemini 2.5 Flash | crawl4ai
