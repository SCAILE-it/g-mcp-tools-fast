#!/bin/bash
# Deployment script for Modal enrichment MCPs service
# Run this from your Mac where Modal CLI is installed

set -e

echo "🚀 Deploying enrichment-mcps to Modal..."

# Check if Modal CLI is installed
if ! command -v modal &> /dev/null; then
    echo "❌ Modal CLI not found. Install with: pip install modal"
    exit 1
fi

# Check if authenticated
if ! modal token get &> /dev/null; then
    echo "❌ Not authenticated. Run: modal setup"
    exit 1
fi

# Deploy the service
echo "📦 Deploying enrichment_mcps.py..."
modal deploy enrichment_mcps.py

echo "✅ Deployment complete!"
echo ""
echo "📋 Next steps:"
echo "1. Get the endpoint URL from Modal dashboard (https://modal.com/apps)"
echo "2. Add to .env.local: MODAL_ENRICHMENT_ENDPOINT=https://..."
echo "3. Test endpoints:"
echo ""
echo "   # Email Intel"
echo "   curl -X POST [endpoint]/email-intel \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"email\": \"test@example.com\"}'"
echo ""
echo "   # Company Data"
echo "   curl -X POST [endpoint]/company-data \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"companyName\": \"Anthropic\", \"domain\": \"anthropic.com\"}'"
echo ""
echo "   # Phone Validation"
echo "   curl -X POST [endpoint]/phone-validation \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"phoneNumber\": \"+1 555 123 4567\"}'"
echo ""
echo "   # Tech Stack"
echo "   curl -X POST [endpoint]/tech-stack \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"domain\": \"anthropic.com\"}'"
echo ""
echo "   # Social Search"
echo "   curl -X POST [endpoint]/social-search \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"query\": \"username\", \"queryType\": \"username\"}'"
echo ""
echo "   # Email Finder"
echo "   curl -X POST [endpoint]/email-finder \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"domain\": \"anthropic.com\", \"limit\": 10}'"
echo ""
echo "   # Email Pattern"
echo "   curl -X POST [endpoint]/email-pattern \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"domain\": \"anthropic.com\", \"firstName\": \"John\", \"lastName\": \"Doe\"}'"
echo ""
echo "   # WHOIS"
echo "   curl -X POST [endpoint]/whois \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"domain\": \"anthropic.com\"}'"
echo ""
echo "   # Wikipedia"
echo "   curl -X POST [endpoint]/wikipedia \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"query\": \"Anthropic\"}'"
echo ""
echo "   # GitHub Intel"
echo "   curl -X POST [endpoint]/github-intel \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"username\": \"torvalds\"}'"
echo ""
