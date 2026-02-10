# Documentation Build Scripts

This directory contains utility scripts for building and maintaining the AutoTimm documentation.

## Scripts

### `build_docs.sh`

Build the documentation site with enhanced sitemap.

**Usage:**
```bash
./scripts/build_docs.sh
```

This script:
1. Runs `zensical build --clean` to generate the documentation
2. Enhances the sitemap with SEO metadata (lastmod, changefreq, priority)
3. Copies `robots.txt` to the site directory
4. Creates a backup of the original sitemap at `site/sitemap.xml.bak`

### `enhance_sitemap.py`

Enhances an existing sitemap.xml with proper SEO metadata.

**Usage:**
```bash
python scripts/enhance_sitemap.py
```

This script adds:
- **lastmod**: Current date in W3C format
- **changefreq**: Update frequency based on content type
  - Weekly: Home, Getting Started, User Guide
  - Monthly: API docs, Examples, Troubleshooting
- **priority**: URL priority (0.6-1.0) based on importance
  - 1.0: Homepage
  - 0.9: Getting Started
  - 0.8: User Guide & Examples
  - 0.7: API & Troubleshooting
  - 0.6: Other pages
- **robots.txt**: Copies to site directory for search engine discovery

## CI/CD Integration

Both GitHub Actions workflows automatically run the sitemap enhancement:
- `.github/workflows/docs.yml` - Deploy on push to dev/zen branches
- `.github/workflows/docs-tags.yml` - Deploy on version tags

## Local Development

To build docs locally:

```bash
# Option 1: Use the convenience script
./scripts/build_docs.sh

# Option 2: Manual steps
zensical build --clean
python scripts/enhance_sitemap.py
```

## Troubleshooting

If the sitemap is not found:
- Ensure you've run `zensical build` first
- Check that `site/sitemap.xml` exists
- Verify you're running from the project root directory

## Files Generated

- `site/sitemap.xml` - Enhanced sitemap with SEO metadata
- `site/sitemap.xml.bak` - Backup of the original sitemap
- `site/robots.txt` - Copy of robots.txt for search engine crawlers
