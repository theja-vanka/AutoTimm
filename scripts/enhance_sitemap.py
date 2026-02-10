#!/usr/bin/env python3
"""
Enhance sitemap.xml with proper metadata (lastmod, changefreq, priority).
This resolves common search engine processing errors.
"""
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

def enhance_sitemap(sitemap_path: str):
    """Add lastmod, changefreq, and priority to sitemap URLs."""
    
    # Parse the XML
    tree = ET.parse(sitemap_path)
    root = tree.getroot()
    
    # Define the namespace
    ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
    
    # Register the namespace to avoid ns0 prefixes
    ET.register_namespace('', 'http://www.sitemaps.org/schemas/sitemap/0.9')
    
    # Get current date in W3C format
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Process each URL entry
    for url in root.findall('sm:url', ns):
        loc = url.find('sm:loc', ns)
        if loc is None:
            continue
            
        url_path = loc.text
        
        # Determine priority based on URL depth and type
        if url_path.endswith('/AutoTimm/'):
            priority = '1.0'
            changefreq = 'weekly'
        elif '/getting-started/' in url_path:
            priority = '0.9'
            changefreq = 'weekly'
        elif '/user-guide/' in url_path:
            priority = '0.8'
            changefreq = 'weekly'
        elif '/api/' in url_path:
            priority = '0.7'
            changefreq = 'monthly'
        elif '/examples/' in url_path:
            priority = '0.8'
            changefreq = 'monthly'
        elif '/troubleshooting/' in url_path:
            priority = '0.7'
            changefreq = 'monthly'
        else:
            priority = '0.6'
            changefreq = 'monthly'
        
        # Check if elements already exist
        lastmod = url.find('sm:lastmod', ns)
        changefreq_elem = url.find('sm:changefreq', ns)
        priority_elem = url.find('sm:priority', ns)
        
        # Add or update lastmod
        if lastmod is None:
            lastmod = ET.SubElement(url, 'lastmod')
        lastmod.text = current_date
        
        # Add or update changefreq
        if changefreq_elem is None:
            changefreq_elem = ET.SubElement(url, 'changefreq')
        changefreq_elem.text = changefreq
        
        # Add or update priority
        if priority_elem is None:
            priority_elem = ET.SubElement(url, 'priority')
        priority_elem.text = priority
    
    # Write the enhanced sitemap
    tree.write(sitemap_path, encoding='UTF-8', xml_declaration=True)
    
    # Pretty print (optional - format the XML)
    with open(sitemap_path, 'r') as f:
        content = f.read()
    
    # Add proper formatting
    content = content.replace('><', '>\n<')
    
    with open(sitemap_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Enhanced sitemap with metadata: {sitemap_path}")
    print(f"  - Added lastmod: {current_date}")
    print(f"  - Added changefreq based on content type")
    print(f"  - Added priority based on URL importance")

if __name__ == '__main__':
    # Support running from both project root and scripts directory
    script_dir = Path(__file__).parent
    if script_dir.name == 'scripts':
        project_root = script_dir.parent
    else:
        project_root = script_dir
    
    sitemap_path = project_root / 'site' / 'sitemap.xml'
    
    if not sitemap_path.exists():
        print(f"Error: Sitemap not found at {sitemap_path}")
        exit(1)
    
    # Backup original
    backup_path = sitemap_path.with_suffix('.xml.bak')
    import shutil
    shutil.copy(sitemap_path, backup_path)
    print(f"✓ Created backup: {backup_path}")
    
    enhance_sitemap(str(sitemap_path))
