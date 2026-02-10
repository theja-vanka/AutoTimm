#!/usr/bin/env bash
# Build documentation and enhance sitemap
set -e

echo "📚 Building documentation..."
zensical build --clean

echo ""
echo "🔧 Enhancing sitemap with SEO metadata..."
python scripts/enhance_sitemap.py

echo ""
echo "✅ Documentation build complete!"
echo "   View at: site/index.html"
