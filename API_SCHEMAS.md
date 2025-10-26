# g-mcp-tools-fast API - Complete Input/Output Schemas

**Last Updated:** October 26, 2025
**Version:** 1.0.0

---

## Authentication (All Endpoints)

**Optional Header:**
```
x-api-key: your-secret-key-here
```

**Note:** Authentication is disabled by default. To enable, set Modal secret `MODAL_API_KEY`.

---

## Response Format (All Endpoints)

### Success Response
```json
{
  "success": true,
  "data": { /* endpoint-specific data */ },
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
  "error": "Error message describing what went wrong",
  "metadata": {
    "source": "tool-name",
    "timestamp": "2025-10-26T17:30:00.000000Z"
  }
}
```

---

## 1. Health Check

### `GET /health`

**Input:** None

**Output:**
```json
{
  "status": "healthy",
  "service": "g-mcp-tools-fast",
  "version": "1.0.0",
  "tools": 9,
  "timestamp": "2025-10-26T17:30:00.515222Z"
}
```

**Use Case:** Service monitoring, uptime checks

---

## 2. Web Scraper

### `POST /scrape`

**Input Schema:**
```json
{
  "url": "string (required)",
  "prompt": "string (required, min 1 char)",
  "schema": {
    "field1": "description",
    "field2": "description"
  },
  "actions": [
    {
      "type": "click|scroll|wait|type|screenshot",
      "selector": "string (optional)",
      "text": "string (optional)",
      "milliseconds": "number (optional)",
      "pixels": "number (optional)"
    }
  ],
  "max_pages": "number (1-50, default: 1)",
  "timeout": "number (5-120 seconds, default: 30)",
  "extract_links": "boolean (default: false)",
  "use_context_analysis": "boolean (default: true)",
  "auto_discover_pages": "boolean (default: false)"
}
```

**Minimal Example:**
```json
{
  "url": "https://anthropic.com",
  "prompt": "Extract company mission and product names"
}
```

**Advanced Example:**
```json
{
  "url": "https://example.com",
  "prompt": "Extract all job listings",
  "schema": {
    "jobs": "array of job objects with title, location, salary",
    "company_name": "string"
  },
  "max_pages": 3,
  "auto_discover_pages": true
}
```

**Output Schema:**
```json
{
  "success": true,
  "data": {
    /* AI-extracted data matching your prompt/schema */
    "company_mission": "string",
    "product_names": ["string", "string"]
  },
  "metadata": {
    "extraction_time": 10.31,
    "pages_scraped": 1,
    "cached": false,
    "model": "gemini-2.5-flash",
    "timestamp": "2025-10-26T17:30:00.000000Z"
  }
}
```

**Quality Check:**
- ✅ URL validation (auto-adds https://)
- ✅ 24-hour caching (same URL+prompt returns cached)
- ✅ Multi-page scraping with AI discovery
- ⚠️ **Issue Found:** Error messages too verbose (exposes stack traces)

---

## 3. Email Intelligence

### `POST /email-intel`

**Input Schema:**
```json
{
  "email": "string (required, must be valid email format)"
}
```

**Example:**
```json
{
  "email": "user@example.com"
}
```

**Output Schema:**
```json
{
  "success": true,
  "data": {
    "email": "user@example.com",
    "platforms": [
      {
        "name": "Twitter",
        "exists": true,
        "url": null
      },
      {
        "name": "GitHub",
        "exists": true,
        "url": null
      }
    ],
    "totalFound": 2
  },
  "metadata": {
    "source": "holehe",
    "timestamp": "2025-10-26T17:30:00.000000Z"
  }
}
```

**Quality Check:**
- ✅ Returns consistent structure even with 0 results
- ✅ Clean error messages
- ⚠️ Platform detection depends on holehe database (may miss some services)

---

## 4. Email Finder

### `POST /email-finder`

**Input Schema:**
```json
{
  "domain": "string (required)",
  "limit": "number (optional, default: 50)",
  "sources": "string (optional, default: 'google,bing')"
}
```

**Example:**
```json
{
  "domain": "anthropic.com",
  "limit": 10
}
```

**Output Schema:**
```json
{
  "success": true,
  "data": {
    "domain": "anthropic.com",
    "emails": [
      {
        "email": "contact@anthropic.com",
        "source": "theHarvester"
      }
    ],
    "totalFound": 1,
    "searchMethod": "theHarvester-google,bing"
  },
  "metadata": {
    "source": "theHarvester",
    "timestamp": "2025-10-26T17:30:00.000000Z"
  }
}
```

**Quality Check:**
- ✅ Clean output structure
- ✅ No duplicate emails
- ⚠️ Results depend on public data availability (may return 0 for privacy-conscious companies)

---

## 5. Company Data

### `POST /company-data`

**Input Schema:**
```json
{
  "companyName": "string (required)",
  "domain": "string (optional)"
}
```

**Example:**
```json
{
  "companyName": "Anthropic",
  "domain": "anthropic.com"
}
```

**Output Schema:**
```json
{
  "success": true,
  "data": {
    "companyName": "Anthropic",
    "domain": "anthropic.com",
    "sources": [
      {
        "name": "OpenCorporates",
        "data": {
          "jurisdiction": "us_de",
          "companyNumber": "123456",
          "status": "Active",
          "incorporationDate": "2021-01-01"
        }
      }
    ]
  },
  "metadata": {
    "source": "company-data",
    "timestamp": "2025-10-26T17:30:00.000000Z"
  }
}
```

**Quality Check:**
- ✅ Returns valid structure even with empty sources
- ⚠️ OpenCorporates has limited US coverage (better for UK/EU companies)
- 💡 **Recommendation:** Add Clearbit or Crunchbase as fallback

---

## 6. Phone Validation

### `POST /phone-validation`

**Input Schema:**
```json
{
  "phoneNumber": "string (required, can be any format)",
  "defaultCountry": "string (optional, 2-letter ISO code, default: 'US')"
}
```

**Examples:**
```json
// US number with country code
{
  "phoneNumber": "+14155552671"
}

// US number without country code (uses defaultCountry)
{
  "phoneNumber": "(415) 555-2671",
  "defaultCountry": "US"
}

// UK number
{
  "phoneNumber": "+44 20 7946 0958",
  "defaultCountry": "GB"
}
```

**Output Schema:**
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
  },
  "metadata": {
    "source": "phone-validation",
    "timestamp": "2025-10-26T17:30:00.000000Z"
  }
}
```

**Line Types:**
- `FIXED_LINE` - Landline
- `MOBILE` - Mobile phone
- `FIXED_LINE_OR_MOBILE` - Could be either
- `TOLL_FREE` - 1-800 numbers
- `PREMIUM_RATE` - Premium services
- `VOIP` - VoIP numbers
- `UNKNOWN` - Cannot determine

**Quality Check:**
- ✅ Excellent output quality
- ✅ Multiple format options
- ✅ Human-readable line types
- ✅ Clear error messages for invalid numbers

---

## 7. Tech Stack Detection

### `POST /tech-stack`

**Input Schema:**
```json
{
  "domain": "string (required, with or without https://)"
}
```

**Example:**
```json
{
  "domain": "anthropic.com"
}
```

**Output Schema:**
```json
{
  "success": true,
  "data": {
    "domain": "anthropic.com",
    "technologies": [
      {
        "name": "Next.js",
        "category": "Framework"
      },
      {
        "name": "React",
        "category": "JavaScript Framework"
      },
      {
        "name": "cloudflare",
        "category": "Web Server"
      }
    ],
    "totalFound": 3
  },
  "metadata": {
    "source": "tech-stack",
    "timestamp": "2025-10-26T17:30:00.000000Z"
  }
}
```

**Detection Capabilities:**
- ✅ JavaScript frameworks (React, Next.js, Vue, Angular)
- ✅ Web servers (from HTTP headers)
- ⚠️ Basic detection only (not as comprehensive as Wappalyzer)

**Quality Check:**
- ✅ Accurate for common technologies
- ⚠️ Misses less obvious tech stack components
- 💡 **Recommendation:** Integrate BuiltWith API for comprehensive detection

---

## 8. Email Pattern Generator

### `POST /email-pattern`

**Input Schema:**
```json
{
  "domain": "string (required)",
  "firstName": "string (optional)",
  "lastName": "string (optional)"
}
```

**Examples:**
```json
// Generic patterns
{
  "domain": "anthropic.com"
}

// Personalized patterns
{
  "domain": "anthropic.com",
  "firstName": "John",
  "lastName": "Doe"
}
```

**Output Schema:**
```json
{
  "success": true,
  "data": {
    "domain": "anthropic.com",
    "patterns": [
      {
        "pattern": "{first}.{last}@{domain}",
        "example": "john.doe@anthropic.com",
        "confidence": 0.9
      },
      {
        "pattern": "{first}@{domain}",
        "example": "john@anthropic.com",
        "confidence": 0.7
      },
      {
        "pattern": "{last}@{domain}",
        "example": "doe@anthropic.com",
        "confidence": 0.5
      },
      {
        "pattern": "{f}{last}@{domain}",
        "example": "jdoe@anthropic.com",
        "confidence": 0.8
      }
    ],
    "totalPatterns": 4
  },
  "metadata": {
    "source": "email-pattern",
    "timestamp": "2025-10-26T17:30:00.000000Z"
  }
}
```

**Confidence Scores:**
- `0.9` - Very common pattern (first.last@)
- `0.8` - Common pattern (firstinitiallast@)
- `0.7` - Somewhat common (firstname@)
- `0.5` - Less common (lastname@)

**Quality Check:**
- ✅ Excellent output quality
- ✅ Covers most common patterns
- ✅ Confidence scores help prioritization
- ✅ Personalized examples when names provided

---

## 9. WHOIS Lookup

### `POST /whois`

**Input Schema:**
```json
{
  "domain": "string (required, without protocol)"
}
```

**Example:**
```json
{
  "domain": "anthropic.com"
}
```

**Output Schema:**
```json
{
  "success": true,
  "data": {
    "domain": "anthropic.com",
    "registrar": "MarkMonitor, Inc.",
    "creationDate": "2001-10-02 18:10:32+00:00",
    "expirationDate": "2033-10-02 18:10:32+00:00",
    "nameServers": [
      "ISLA.NS.CLOUDFLARE.COM",
      "RANDY.NS.CLOUDFLARE.COM"
    ]
  },
  "metadata": {
    "source": "whois",
    "timestamp": "2025-10-26T17:30:00.000000Z"
  }
}
```

**Quality Check:**
- ✅ Accurate data
- ✅ Complete registration info
- ✅ Clean date formatting
- ✅ Handles domains without WHOIS privacy

---

## 10. GitHub Intelligence

### `POST /github-intel`

**Input Schema:**
```json
{
  "username": "string (required, GitHub username or org)"
}
```

**Example:**
```json
{
  "username": "anthropics"
}
```

**Output Schema:**
```json
{
  "success": true,
  "data": {
    "username": "anthropics",
    "name": "Anthropic",
    "bio": "AI safety company",
    "company": null,
    "location": "United States of America",
    "publicRepos": 54,
    "followers": 14565,
    "following": 0,
    "languages": {
      "Python": 6,
      "TypeScript": 3,
      "JavaScript": 1,
      "C#": 1,
      "Go": 1,
      "Kotlin": 1,
      "PHP": 1,
      "Ruby": 1,
      "Java": 1
    },
    "profileUrl": "https://github.com/anthropics"
  },
  "metadata": {
    "source": "github-intel",
    "timestamp": "2025-10-26T17:30:00.000000Z"
  }
}
```

**Quality Check:**
- ✅ Comprehensive profile data
- ✅ Language analysis from top 20 repos
- ✅ Useful for developer recruitment/research
- ⚠️ GitHub API rate limits apply (60/hour unauthenticated)

---

## Error Handling Summary

### Common Error Types

**Missing Required Field:**
```json
{
  "success": false,
  "error": "domain required"
}
```

**Invalid Input:**
```json
{
  "success": false,
  "error": "(1) The string supplied did not seem to be a phone number.",
  "metadata": {
    "source": "phone-validation",
    "timestamp": "2025-10-26T17:30:00.000000Z"
  }
}
```

**Authentication Error (if API key enabled):**
```json
{
  "detail": "Invalid or missing API key"
}
```

**Internal Error:**
```json
{
  "success": false,
  "error": "Unexpected error: [error description]"
}
```

---

## Quality Issues Found During Testing

### Critical Issues ⚠️
1. **Web Scraper Error Verbosity** - Stack traces exposed in error messages (security risk)

### Minor Issues 💡
2. **Error Consistency** - Some endpoints include metadata in errors, others don't
3. **Company Data Coverage** - OpenCorporates has limited data for US companies
4. **Tech Stack Detection** - Basic detection, misses many technologies

### Recommendations
1. **Sanitize scraper errors** - Show user-friendly messages, log details server-side
2. **Standardize error format** - All errors should include metadata
3. **Add fallback data sources** - Clearbit, Crunchbase for company data
4. **Enhance tech detection** - Integrate BuiltWith or Wappalyzer API

---

## Overall Schema Quality: 90/100

**Strengths:**
- ✅ Consistent response format across all endpoints
- ✅ Rich metadata (timestamps, sources)
- ✅ Clear field names (camelCase)
- ✅ Comprehensive data in responses

**Weaknesses:**
- ⚠️ Some error messages too technical
- ⚠️ Input validation could be stricter (use Pydantic models for all endpoints)
- ⚠️ No rate limit info in responses

**Production Readiness:** YES, with minor improvements recommended
