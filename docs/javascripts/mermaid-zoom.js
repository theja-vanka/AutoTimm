/**
 * Mermaid Diagram Zoom and Pan Functionality
 * Adds interactive zoom, pan, and fullscreen capabilities to mermaid diagrams.
 * Works with MkDocs Material's built-in mermaid rendering.
 */

/**
 * Configure Mermaid theme based on current color scheme
 */
function configureMermaidTheme() {
    if (typeof mermaid === 'undefined') return;
    
    const scheme = document.querySelector('[data-md-color-scheme]')?.getAttribute('data-md-color-scheme');
    const isDark = scheme === 'slate';
    
    mermaid.initialize({
        startOnLoad: true,
        theme: isDark ? 'dark' : 'default',
        themeVariables: isDark ? {
            primaryColor: '#42A5F5',
            primaryTextColor: '#E0E0E0',
            primaryBorderColor: '#90CAF9',
            lineColor: '#90CAF9',
            secondaryColor: '#1E3A5F',
            tertiaryColor: '#2C5282',
            noteBkgColor: '#1E3A5F',
            noteBorderColor: '#42A5F5',
            noteTextColor: '#E0E0E0',
            textColor: '#E0E0E0',
            mainBkg: '#1E3A5F',
            secondBkg: '#2C5282',
            border1: '#90CAF9',
            border2: '#42A5F5',
            arrowheadColor: '#90CAF9',
            fontFamily: 'Roboto, sans-serif',
            fontSize: '16px',
            labelBackground: '#1A1A1A',
            labelTextColor: '#E0E0E0',
            edgeLabelBackground: '#1A1A1A',
            clusterBkg: '#1E3A5F',
            clusterBorder: '#90CAF9',
            defaultLinkColor: '#90CAF9',
            titleColor: '#E0E0E0',
            actorBorder: '#90CAF9',
            actorBkg: '#2C5282',
            actorTextColor: '#E0E0E0',
            actorLineColor: '#90CAF9',
            signalColor: '#E0E0E0',
            signalTextColor: '#E0E0E0',
            labelBoxBkgColor: '#1A1A1A',
            labelBoxBorderColor: '#90CAF9',
            loopTextColor: '#E0E0E0',
            activationBorderColor: '#90CAF9',
            activationBkgColor: '#2C5282',
            sequenceNumberColor: '#E0E0E0'
        } : {
            primaryColor: '#2196F3',
            primaryTextColor: '#212121',
            primaryBorderColor: '#1976D2',
            lineColor: '#1976D2',
            secondaryColor: '#E3F2FD',
            tertiaryColor: '#BBDEFB',
            noteBkgColor: '#E3F2FD',
            noteBorderColor: '#2196F3',
            noteTextColor: '#212121',
            textColor: '#212121',
            mainBkg: '#E3F2FD',
            secondBkg: '#BBDEFB',
            border1: '#1976D2',
            border2: '#2196F3',
            fontFamily: 'Roboto, sans-serif',
            fontSize: '16px'
        },
        securityLevel: 'loose',
        logLevel: 'error'
    });
}

function initializeMermaidZoom() {
    const mermaidDiagrams = document.querySelectorAll('.mermaid');

    mermaidDiagrams.forEach(function(diagram) {
        // Skip if already wrapped
        if (diagram.closest('.mermaid-zoom-wrapper')) return;

        const svg = diagram.querySelector('svg');
        if (!svg) return;

        // Wrap diagram in a container for zoom controls
        const wrapper = document.createElement('div');
        wrapper.className = 'mermaid-zoom-wrapper';
        wrapper.style.position = 'relative';
        diagram.parentNode.insertBefore(wrapper, diagram);
        wrapper.appendChild(diagram);

        // Add zoom controls
        const controls = document.createElement('div');
        controls.className = 'mermaid-zoom-controls';
        controls.innerHTML = `
            <button class="zoom-btn zoom-in" title="Zoom In" aria-label="Zoom In">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20">
                    <path fill="currentColor" d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14zm.5-7H9v2H7v1h2v2h1v-2h2V9h-2z"/>
                </svg>
            </button>
            <button class="zoom-btn zoom-out" title="Zoom Out" aria-label="Zoom Out">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20">
                    <path fill="currentColor" d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14zM7 9h5v1H7z"/>
                </svg>
            </button>
            <button class="zoom-btn zoom-reset" title="Reset Zoom" aria-label="Reset Zoom">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20">
                    <path fill="currentColor" d="M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z"/>
                </svg>
            </button>
            <button class="zoom-btn zoom-fullscreen" title="Fullscreen" aria-label="Toggle Fullscreen">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20">
                    <path fill="currentColor" d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/>
                </svg>
            </button>
        `;
        wrapper.insertBefore(controls, diagram);

        // Make SVG responsive and zoomable
        svg.style.maxWidth = '100%';
        svg.style.height = 'auto';
        svg.style.cursor = 'grab';

        // Zoom state
        let scale = 1;
        let translateX = 0;
        let translateY = 0;
        let isDragging = false;
        let startX = 0;
        let startY = 0;

        function applyTransform() {
            svg.style.transform = 'translate(' + translateX + 'px, ' + translateY + 'px) scale(' + scale + ')';
            svg.style.transformOrigin = '0 0';
        }

        controls.querySelector('.zoom-in').addEventListener('click', function(e) {
            e.stopPropagation();
            scale = Math.min(scale * 1.2, 5);
            applyTransform();
        });

        controls.querySelector('.zoom-out').addEventListener('click', function(e) {
            e.stopPropagation();
            scale = Math.max(scale / 1.2, 0.5);
            applyTransform();
        });

        controls.querySelector('.zoom-reset').addEventListener('click', function(e) {
            e.stopPropagation();
            scale = 1;
            translateX = 0;
            translateY = 0;
            applyTransform();
        });

        controls.querySelector('.zoom-fullscreen').addEventListener('click', function(e) {
            e.stopPropagation();
            if (!document.fullscreenElement) {
                wrapper.requestFullscreen().catch(function(err) {
                    console.log('Fullscreen error:', err);
                });
                wrapper.classList.add('fullscreen-active');
            } else {
                document.exitFullscreen();
                wrapper.classList.remove('fullscreen-active');
            }
        });

        diagram.addEventListener('wheel', function(e) {
            e.preventDefault();
            var delta = e.deltaY > 0 ? 0.9 : 1.1;
            scale = Math.min(Math.max(scale * delta, 0.5), 5);
            applyTransform();
        });

        svg.addEventListener('mousedown', function(e) {
            if (e.button === 0) {
                isDragging = true;
                startX = e.clientX - translateX;
                startY = e.clientY - translateY;
                svg.style.cursor = 'grabbing';
            }
        });

        document.addEventListener('mousemove', function(e) {
            if (isDragging) {
                translateX = e.clientX - startX;
                translateY = e.clientY - startY;
                applyTransform();
            }
        });

        document.addEventListener('mouseup', function() {
            if (isDragging) {
                isDragging = false;
                svg.style.cursor = 'grab';
            }
        });

        // Touch support
        var touchStartX = 0;
        var touchStartY = 0;
        var initialDistance = 0;
        var initialScale = 1;

        svg.addEventListener('touchstart', function(e) {
            if (e.touches.length === 1) {
                touchStartX = e.touches[0].clientX - translateX;
                touchStartY = e.touches[0].clientY - translateY;
            } else if (e.touches.length === 2) {
                var dx = e.touches[0].clientX - e.touches[1].clientX;
                var dy = e.touches[0].clientY - e.touches[1].clientY;
                initialDistance = Math.sqrt(dx * dx + dy * dy);
                initialScale = scale;
            }
        });

        svg.addEventListener('touchmove', function(e) {
            e.preventDefault();
            if (e.touches.length === 1) {
                translateX = e.touches[0].clientX - touchStartX;
                translateY = e.touches[0].clientY - touchStartY;
                applyTransform();
            } else if (e.touches.length === 2) {
                var dx = e.touches[0].clientX - e.touches[1].clientX;
                var dy = e.touches[0].clientY - e.touches[1].clientY;
                var distance = Math.sqrt(dx * dx + dy * dy);
                scale = Math.min(Math.max(initialScale * (distance / initialDistance), 0.5), 5);
                applyTransform();
            }
        });
    });
}

// Use MutationObserver to detect when Material finishes rendering mermaid SVGs
function observeMermaidRendering() {
    var observer = new MutationObserver(function(mutations) {
        var hasMermaidSvg = false;
        mutations.forEach(function(mutation) {
            mutation.addedNodes.forEach(function(node) {
                if (node.nodeName === 'svg' || (node.querySelector && node.querySelector('svg'))) {
                    var parent = node.closest ? node.closest('.mermaid') : null;
                    if (!parent && node.parentElement) {
                        parent = node.parentElement.closest('.mermaid');
                    }
                    if (parent) hasMermaidSvg = true;
                }
            });
        });
        if (hasMermaidSvg) {
            initializeMermaidZoom();
        }
    });

    observer.observe(document.body, { childList: true, subtree: true });
    return observer;
}

// Initial setup
var mermaidObserver = observeMermaidRendering();

// Configure theme on initial load
configureMermaidTheme();

// Also run on initial load in case SVGs are already rendered
document.addEventListener('DOMContentLoaded', function() {
    configureMermaidTheme();
    initializeMermaidZoom();
});

// Watch for color scheme changes
var schemeObserver = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
        if (mutation.attributeName === 'data-md-color-scheme') {
            configureMermaidTheme();
            // Re-render mermaid diagrams with new theme
            if (typeof mermaid !== 'undefined') {
                document.querySelectorAll('.mermaid').forEach(function(element) {
                    var code = element.getAttribute('data-mermaid-src');
                    if (code) {
                        element.removeAttribute('data-processed');
                        element.innerHTML = code;
                    }
                });
                mermaid.init(undefined, document.querySelectorAll('.mermaid:not([data-processed])'));
                setTimeout(initializeMermaidZoom, 100);
            }
        }
    });
});

// Observe the document element for color scheme changes
schemeObserver.observe(document.querySelector('body'), { 
    attributes: true, 
    subtree: true,
    attributeFilter: ['data-md-color-scheme']
});

// Re-initialize on Material instant navigation
if (typeof document$ !== 'undefined') {
    document$.subscribe(function() {
        // Disconnect old observer and start a new one for the new page
        if (mermaidObserver) mermaidObserver.disconnect();
        mermaidObserver = observeMermaidRendering();
        configureMermaidTheme();
        // Also try immediately in case SVGs are already present
        initializeMermaidZoom();
    });
}
