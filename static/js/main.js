// Main JavaScript for Knowledge Graph App

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all tooltips
    const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltips.forEach(tooltip => {
        new bootstrap.Tooltip(tooltip);
    });
    
    // Handle input type selection in data processing form
    const inputTypeRadios = document.querySelectorAll('input[name="input_type"]');
    const videoFormGroup = document.getElementById('video-form-group');
    const csvFormGroup = document.getElementById('csv-form-group');
    const urlFormGroup = document.getElementById('url-form-group');
    
    if (inputTypeRadios.length) {
        inputTypeRadios.forEach(radio => {
            radio.addEventListener('change', function() {
                // Hide all form groups
                if (videoFormGroup) videoFormGroup.style.display = 'none';
                if (csvFormGroup) csvFormGroup.style.display = 'none';
                if (urlFormGroup) urlFormGroup.style.display = 'none';
                
                // Show the selected form group
                if (this.value === 'video' && videoFormGroup) {
                    videoFormGroup.style.display = 'block';
                } else if (this.value === 'csv' && csvFormGroup) {
                    csvFormGroup.style.display = 'block';
                } else if (this.value === 'url' && urlFormGroup) {
                    urlFormGroup.style.display = 'block';
                }
            });
        });
        
        // Trigger change event on the checked radio button
        const checkedRadio = document.querySelector('input[name="input_type"]:checked');
        if (checkedRadio) {
            checkedRadio.dispatchEvent(new Event('change'));
        } else if (inputTypeRadios[0]) {
            // Select the first option by default
            inputTypeRadios[0].checked = true;
            inputTypeRadios[0].dispatchEvent(new Event('change'));
        }
    }
    
    // Initialize file input labels to show selected filename
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', function() {
            const label = this.nextElementSibling;
            if (label && this.files.length > 0) {
                label.textContent = this.files[0].name;
            } else if (label) {
                label.textContent = 'Choose file';
            }
        });
    });
    
    // Initialize the graph visualization if on the visualization page
    const graphContainer = document.getElementById('graph-visualization');
    if (graphContainer) {
        const graphId = graphContainer.getAttribute('data-graph-id');
        if (graphId) {
            const graphViz = new KnowledgeGraphVisualizer('graph-visualization');
            graphViz.loadData(graphId);
            
            // Add window resize handler for responsive visualization
            window.addEventListener('resize', function() {
                graphViz.resize();
            });
        }
    }
    
    // Handle deletion confirmations
    const deleteButtons = document.querySelectorAll('.delete-confirm');
    deleteButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            if (!confirm('Are you sure you want to delete this item? This action cannot be undone.')) {
                e.preventDefault();
            }
        });
    });
    
    // Show loading spinner when forms are submitted
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            // Check if the form has a submit button
            const submitBtn = this.querySelector('button[type="submit"]');
            if (submitBtn) {
                // Create and show spinner
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
                submitBtn.disabled = true;
            }
        });
    });
});
