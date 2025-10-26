#!/bin/bash
# Deployment script for g-mcp-tools-fast Modal API
# Production-ready enrichment API with 9 tools

set -e

echo "🚀 Deploying g-mcp-tools-fast to Modal..."
echo ""

# Check if Modal CLI is installed
if ! command -v modal &> /dev/null; then
    echo "❌ Modal CLI not found. Install with:"
    echo "   pip install modal"
    exit 1
fi

# Check if authenticated
if ! modal token get &> /dev/null; then
    echo "❌ Not authenticated. Run:"
    echo "   modal setup"
    exit 1
fi

# Check if gemini-secret exists
echo "📋 Checking Modal secrets..."
if ! modal secret list | grep -q "gemini-secret"; then
    echo "⚠️  Warning: 'gemini-secret' not found"
    echo ""
    echo "Create it with:"
    echo "   modal secret create gemini-secret GOOGLE_GENERATIVE_AI_API_KEY=your-key-here"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Optional: Check for MODAL_API_KEY secret for authentication
if ! modal secret list | grep -q "modal-api-key"; then
    echo "ℹ️  Note: 'modal-api-key' secret not found"
    echo "   API will be publicly accessible without authentication"
    echo "   To enable auth, create secret:"
    echo "   modal secret create modal-api-key MODAL_API_KEY=your-secret-key"
    echo ""
fi

# Deploy the service
echo "📦 Deploying g-mcp-tools-complete.py..."
modal deploy g-mcp-tools-complete.py

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📋 Your API is now live at:"
echo "   https://scaile--g-mcp-tools-fast-api.modal.run"
echo ""
echo "📚 Interactive API Documentation:"
echo "   https://scaile--g-mcp-tools-fast-api.modal.run/docs"
echo "   https://scaile--g-mcp-tools-fast-api.modal.run/redoc"
echo ""
echo "🏥 Health Check:"
echo "   curl https://scaile--g-mcp-tools-fast-api.modal.run/health"
echo ""
echo "🔧 Test Endpoints:"
echo ""
echo "   # Email Pattern (no auth required)"
echo "   curl -X POST https://scaile--g-mcp-tools-fast-api.modal.run/email-pattern \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"domain\": \"anthropic.com\"}'"
echo ""
echo "   # Web Scraper"
echo "   curl -X POST https://scaile--g-mcp-tools-fast-api.modal.run/scrape \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"url\": \"https://anthropic.com\", \"prompt\": \"Extract company info\"}'"
echo ""
echo "   # Phone Validation"
echo "   curl -X POST https://scaile--g-mcp-tools-fast-api.modal.run/phone-validation \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"phoneNumber\": \"+14155552671\"}'"
echo ""
echo "   # GitHub Intel"
echo "   curl -X POST https://scaile--g-mcp-tools-fast-api.modal.run/github-intel \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"username\": \"anthropics\"}'"
echo ""
echo "🔐 To enable authentication:"
echo "   1. Create Modal secret: modal secret create modal-api-key MODAL_API_KEY=your-secret-key"
echo "   2. Redeploy: modal deploy g-mcp-tools-complete.py"
echo "   3. Include header in requests: -H 'x-api-key: your-secret-key'"
echo ""
echo "📊 Monitor logs:"
echo "   modal app logs g-mcp-tools-fast --follow"
echo ""
