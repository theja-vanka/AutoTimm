/**
 * Mermaid Diagram Zoom and Pan Functionality
 * Adds interactive zoom, pan, and fullscreen capabilities to mermaid diagrams
 */

// Wait for mermaid diagrams to be rendered
document.addEventListener('DOMContentLoaded', function() {
    // Initialize after a short delay to ensure mermaid has rendered
    setTimeout(initializeMermaidZoom, 500);
});

function initializeMermaidZoom() {
    const mermaidDiagrams = document.querySelectorAll('.mermaid');

    mermaidDiagrams.forEach(function(diagram, index) {
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

        // Get the SVG element
        const svg = diagram.querySelector('svg');
        if (!svg) return;

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

        // Apply transform
        function applyTransform() {
            svg.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;
            svg.style.transformOrigin = '0 0';
        }

        // Zoom in
        controls.querySelector('.zoom-in').addEventListener('click', function() {
            scale = Math.min(scale * 1.2, 5);
            applyTransform();
        });

        // Zoom out
        controls.querySelector('.zoom-out').addEventListener('click', function() {
            scale = Math.max(scale / 1.2, 0.5);
            applyTransform();
        });

        // Reset zoom
        controls.querySelector('.zoom-reset').addEventListener('click', function() {
            scale = 1;
            translateX = 0;
            translateY = 0;
            applyTransform();
        });

        // Fullscreen toggle
        controls.querySelector('.zoom-fullscreen').addEventListener('click', function() {
            if (!document.fullscreenElement) {
                wrapper.requestFullscreen().catch(err => {
                    console.log('Fullscreen error:', err);
                });
                wrapper.classList.add('fullscreen-active');
            } else {
                document.exitFullscreen();
                wrapper.classList.remove('fullscreen-active');
            }
        });

        // Mouse wheel zoom
        diagram.addEventListener('wheel', function(e) {
            e.preventDefault();
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            scale = Math.min(Math.max(scale * delta, 0.5), 5);
            applyTransform();
        });

        // Pan with mouse drag
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

        // Touch support for mobile
        let touchStartX = 0;
        let touchStartY = 0;
        let initialDistance = 0;
        let initialScale = 1;

        svg.addEventListener('touchstart', function(e) {
            if (e.touches.length === 1) {
                touchStartX = e.touches[0].clientX - translateX;
                touchStartY = e.touches[0].clientY - translateY;
            } else if (e.touches.length === 2) {
                const dx = e.touches[0].clientX - e.touches[1].clientX;
                const dy = e.touches[0].clientY - e.touches[1].clientY;
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
                const dx = e.touches[0].clientX - e.touches[1].clientX;
                const dy = e.touches[0].clientY - e.touches[1].clientY;
                const distance = Math.sqrt(dx * dx + dy * dy);
                scale = Math.min(Math.max(initialScale * (distance / initialDistance), 0.5), 5);
                applyTransform();
            }
        });
    });
}

// Re-initialize when navigating in Material theme (instant loading)
if (typeof document$ !== 'undefined') {
    document$.subscribe(function() {
        setTimeout(initializeMermaidZoom, 500);
    });
}
