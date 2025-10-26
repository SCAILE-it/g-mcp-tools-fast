#!/bin/bash
# Deployment script for Modal flexible-scraper service
# Run this from your Mac where Modal CLI is installed

set -e

echo "🚀 Deploying flexible-scraper to Modal..."

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
echo "📦 Deploying flexible_scraper.py..."
modal deploy flexible_scraper.py

echo "✅ Deployment complete!"
echo ""
echo "📋 Next steps:"
echo "1. Get the endpoint URL from Modal dashboard"
echo "2. Add to .env.local: MODAL_SCRAPER_ENDPOINT=https://..."
echo "3. Test with: curl -X POST [endpoint] -H 'Content-Type: application/json' -d '{\"url\": \"https://anthropic.com\", \"prompt\": \"Extract company info\"}'"
