// Enhanced Analysis JavaScript for Lokman-v2
// Handles progress tracking and improved UI interactions

// Enhanced form submission with progress modal
function handleAnalysisSubmission() {
    const form = document.getElementById('uploadForm');
    console.log('Setting up form submission handler for:', form);
    
    if (!form) {
        console.error('uploadForm not found!');
        return;
    }
    
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        console.log('Form submitted!');
        
        const formData = new FormData();
        const files = document.getElementById('files').files;
        console.log('Files selected:', files.length);
        
        if (files.length === 0) {
            alert('Please select files to upload');
            return;
        }
        
        for (let file of files) {
            formData.append('files', file);
        }
        
        const analyzeBtn = document.getElementById('analyzeBtn');
        console.log('Starting analysis with button:', analyzeBtn);
        
        // Show enhanced loading state with progress tracking
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Analyzing...';
        
        const resultsDiv = document.getElementById('results');
        if (resultsDiv) {
            resultsDiv.style.display = 'none';
        }
        
        // Create and show progress modal
        console.log('Showing progress modal...');
        showProgressModal(files);
        
        try {
            console.log('Sending request to /api/analyze...');
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });
            
            console.log('Response received:', response.status);
            const result = await response.json();
            console.log('Analysis result:', result);
            
            // Hide progress modal
            hideProgressModal();
            
            if (result.success) {
                displayEnhancedResults(result);
            } else {
                showError(result.error || 'Analysis failed');
            }
        } catch (error) {
            console.error('Analysis error:', error);
            hideProgressModal();
            showError('Network error: ' + error.message);
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="fas fa-microscope"></i> Analyze CT Scan';
            console.log('Analysis complete, button re-enabled');
        }
    });
}

function showProgressModal(files) {
    // Create progress modal
    const modalHtml = `
        <div class="modal fade" id="progressModal" tabindex="-1" data-bs-backdrop="static" data-bs-keyboard="false">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header bg-primary text-white">
                        <h5 class="modal-title">
                            <i class="fas fa-brain me-2"></i>AI Analysis in Progress
                        </h5>
                    </div>
                    <div class="modal-body">
                        <div class="mb-4">
                            <h6>Overall Progress</h6>
                            <div class="progress mb-2" style="height: 25px;">
                                <div class="progress-bar progress-bar-striped progress-bar-animated" 
                                     id="overallProgress" style="width: 0%"></div>
                            </div>
                            <div class="d-flex justify-content-between">
                                <small id="progressText">Initializing analysis...</small>
                                <small id="progressPercent">0%</small>
                            </div>
                        </div>
                        
                        <div class="mb-4">
                            <h6>Current File: <span id="currentFileName">-</span></h6>
                            <div class="progress mb-2">
                                <div class="progress-bar bg-info" id="fileProgress" style="width: 0%"></div>
                            </div>
                            <small id="currentStage">Preparing...</small>
                        </div>
                        
                        <div class="mb-3">
                            <h6>Processing Pipeline</h6>
                            <div class="d-flex flex-wrap gap-2" id="pipelineStages">
                                <span class="badge bg-secondary">Loading</span>
                                <span class="badge bg-secondary">Preprocessing</span>
                                <span class="badge bg-secondary">Segmentation</span>
                                <span class="badge bg-secondary">Classification</span>
                                <span class="badge bg-secondary">Analysis</span>
                            </div>
                        </div>
                        
                        <div class="row">
                            <div class="col-md-6">
                                <h6>Files Queue</h6>
                                <div class="list-group list-group-flush" id="filesQueue" style="max-height: 200px; overflow-y: auto;">
                                    ${Array.from(files).map((file, index) => `
                                        <div class="list-group-item d-flex justify-content-between align-items-center" id="file-${index}">
                                            <span>${file.name}</span>
                                            <span class="badge bg-secondary">Queued</span>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                            <div class="col-md-6">
                                <h6>Real-time Metrics</h6>
                                <ul class="list-unstyled">
                                    <li><strong>Files Processed:</strong> <span id="processedCount">0</span>/${files.length}</li>
                                    <li><strong>Avg. Processing Time:</strong> <span id="avgTime">-</span></li>
                                    <li><strong>Estimated Remaining:</strong> <span id="estimatedTime">Calculating...</span></li>
                                    <li><strong>Model:</strong> SimpleAttentionUNet3D v2.1</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Remove existing modal if any
    const existingModal = document.getElementById('progressModal');
    if (existingModal) {
        existingModal.remove();
    }
    
    // Add modal to body
    document.body.insertAdjacentHTML('beforeend', modalHtml);
    
    // Show modal
    try {
        const modalElement = document.getElementById('progressModal');
        console.log('Modal element created:', modalElement);
        
        const modal = new bootstrap.Modal(modalElement);
        modal.show();
        console.log('Modal shown successfully');
        
        // Start simulated progress
        simulateAnalysisProgress(files);
    } catch (error) {
        console.error('Error showing modal:', error);
        // Fall back to simple alert if modal fails
        alert('Analysis starting...');
    }
}

function simulateAnalysisProgress(files) {
    const stages = ['Loading', 'Preprocessing', 'Segmentation', 'Classification', 'Analysis'];
    let currentFileIndex = 0;
    let currentStageIndex = 0;
    let overallProgress = 0;
    const startTime = Date.now();
    
    function updateProgress() {
        const file = files[currentFileIndex];
        const totalSteps = files.length * stages.length;
        const currentStep = currentFileIndex * stages.length + currentStageIndex;
        
        // Update overall progress
        overallProgress = Math.min(95, (currentStep / totalSteps) * 100);
        document.getElementById('overallProgress').style.width = overallProgress + '%';
        document.getElementById('progressPercent').textContent = Math.round(overallProgress) + '%';
        
        // Update current file info
        document.getElementById('currentFileName').textContent = file.name;
        document.getElementById('currentStage').textContent = `${stages[currentStageIndex]}...`;
        
        // Update file progress
        const fileProgress = ((currentStageIndex + 1) / stages.length) * 100;
        document.getElementById('fileProgress').style.width = fileProgress + '%';
        
        // Update pipeline stages
        const pipelineElement = document.getElementById('pipelineStages');
        pipelineElement.innerHTML = stages.map((stage, index) => {
            let className = 'badge bg-secondary';
            if (index < currentStageIndex) className = 'badge bg-success';
            else if (index === currentStageIndex) className = 'badge bg-primary';
            return `<span class="${className}">${stage}</span>`;
        }).join(' ');
        
        // Update file status in queue
        const fileElement = document.getElementById(`file-${currentFileIndex}`);
        if (fileElement) {
            const badge = fileElement.querySelector('.badge');
            if (currentStageIndex === stages.length - 1) {
                badge.className = 'badge bg-success';
                badge.textContent = 'Complete';
            } else {
                badge.className = 'badge bg-primary';
                badge.textContent = 'Processing';
            }
        }
        
        // Update metrics
        document.getElementById('processedCount').textContent = 
            currentStageIndex === stages.length - 1 ? currentFileIndex + 1 : currentFileIndex;
        
        const elapsedTime = (Date.now() - startTime) / 1000;
        const avgTimePerFile = currentFileIndex > 0 ? elapsedTime / currentFileIndex : 0;
        document.getElementById('avgTime').textContent = 
            avgTimePerFile > 0 ? `${avgTimePerFile.toFixed(1)}s` : '-';
        
        const remainingFiles = files.length - currentFileIndex - (currentStageIndex === stages.length - 1 ? 1 : 0);
        const estimatedRemaining = remainingFiles * avgTimePerFile;
        document.getElementById('estimatedTime').textContent = 
            estimatedRemaining > 0 ? `${estimatedRemaining.toFixed(0)}s` : 'Almost done!';
        
        // Progress to next stage/file
        currentStageIndex++;
        if (currentStageIndex >= stages.length) {
            currentStageIndex = 0;
            currentFileIndex++;
            
            if (currentFileIndex >= files.length) {
                // Analysis complete
                document.getElementById('overallProgress').style.width = '100%';
                document.getElementById('progressPercent').textContent = '100%';
                document.getElementById('progressText').textContent = 'Analysis complete! Generating report...';
                return;
            }
        }
        
        // Update progress text
        const progressMessages = [
            'Loading DICOM files and extracting metadata...',
            'Preprocessing images and normalizing data...',
            'Performing 3D segmentation with attention mechanism...',
            'Running AI classification and risk assessment...',
            'Generating clinical insights and recommendations...'
        ];
        document.getElementById('progressText').textContent = progressMessages[currentStageIndex];
        
        // Continue simulation
        setTimeout(updateProgress, Math.random() * 800 + 400); // 400-1200ms per stage
    }
    
    updateProgress();
}

function hideProgressModal() {
    console.log('Hiding progress modal...');
    try {
        const modalElement = document.getElementById('progressModal');
        if (modalElement) {
            const modal = bootstrap.Modal.getInstance(modalElement);
            if (modal) {
                modal.hide();
            }
            
            // Remove modal and backdrop after animation
            setTimeout(() => {
                modalElement.remove();
                // Also remove any leftover backdrops
                const backdrops = document.querySelectorAll('.modal-backdrop');
                backdrops.forEach(backdrop => backdrop.remove());
                // Remove modal-open class from body
                document.body.classList.remove('modal-open');
                document.body.style.removeProperty('overflow');
                document.body.style.removeProperty('padding-right');
                console.log('Modal cleaned up successfully');
            }, 300);
        }
    } catch (error) {
        console.error('Error hiding modal:', error);
        // Force cleanup if there's an error
        const backdrops = document.querySelectorAll('.modal-backdrop');
        backdrops.forEach(backdrop => backdrop.remove());
        document.body.classList.remove('modal-open');
        document.body.style.removeProperty('overflow');
        document.body.style.removeProperty('padding-right');
    }
}

function displayEnhancedResults(result) {
    console.log('Displaying enhanced results:', result);
    
    const resultContent = document.getElementById('resultsContent');
    if (!resultContent) {
        console.error('resultsContent element not found!');
        alert('Analysis complete! Check console for results.');
        return;
    }
    
    const summary = result.overall_summary;
    const details = result.detailed_results;
    const fileResults = result.file_results;
    
    let alertClass = 'alert-success';
    let iconClass = 'fa-check-circle';
    let urgencyBadge = 'badge bg-success';
    
    if (summary.risk_level === 'high') {
        alertClass = 'alert-danger';
        iconClass = 'fa-exclamation-triangle';
        urgencyBadge = 'badge bg-danger';
    } else if (summary.risk_level === 'medium') {
        alertClass = 'alert-warning';
        iconClass = 'fa-exclamation-circle';
        urgencyBadge = 'badge bg-warning text-dark';
    }
    
    resultContent.innerHTML = `
        <div class="alert ${alertClass} d-flex align-items-center" role="alert">
            <i class="fas ${iconClass} fa-3x me-4"></i>
            <div class="flex-grow-1">
                <h4 class="alert-heading mb-2">${summary.prediction}</h4>
                <p class="mb-2"><strong>Confidence:</strong> ${(summary.confidence * 100).toFixed(2)}%</p>
                <span class="${urgencyBadge} fs-6">Clinical Priority: ${summary.clinical_urgency.toUpperCase()}</span>
            </div>
        </div>
        
        <!-- Enhanced Metrics Grid -->
        <div class="row mt-4">
            <div class="col-md-3">
                <div class="card bg-light text-center">
                    <div class="card-body">
                        <h3 class="text-primary">${details.files_processed}</h3>
                        <small class="text-muted">Files Analyzed</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-light text-center">
                    <div class="card-body">
                        <h3 class="text-warning">${details.total_abnormal_regions}</h3>
                        <small class="text-muted">Abnormal Regions</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-light text-center">
                    <div class="card-body">
                        <h3 class="text-danger">${details.total_tumor_volume_ml} mL</h3>
                        <small class="text-muted">Total Volume</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-light text-center">
                    <div class="card-body">
                        <h3 class="text-success">${details.processing_time}s</h3>
                        <small class="text-muted">Processing Time</small>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- File-by-File Results -->
        <div class="mt-4">
            <h5><i class="fas fa-file-medical me-2"></i>Detailed File Analysis</h5>
            <div class="accordion" id="fileAccordion">
                ${fileResults.map((file, index) => `
                    <div class="accordion-item">
                        <h2 class="accordion-header" id="heading${index}">
                            <button class="accordion-button ${index === 0 ? '' : 'collapsed'}" type="button" 
                                    data-bs-toggle="collapse" data-bs-target="#collapse${index}">
                                <div class="d-flex justify-content-between w-100 me-3">
                                    <span><strong>${file.filename}</strong></span>
                                    <span class="badge bg-${file.risk_level === 'high' ? 'danger' : file.risk_level === 'medium' ? 'warning' : 'success'}">${file.risk_level.toUpperCase()}</span>
                                </div>
                            </button>
                        </h2>
                        <div id="collapse${index}" class="accordion-collapse collapse ${index === 0 ? 'show' : ''}" 
                             data-bs-parent="#fileAccordion">
                            <div class="accordion-body">
                                <div class="row">
                                    <div class="col-md-6">
                                        <h6>Analysis Results:</h6>
                                        <ul class="list-unstyled">
                                            <li><strong>Prediction:</strong> ${file.prediction}</li>
                                            <li><strong>Confidence:</strong> ${(file.confidence * 100).toFixed(2)}%</li>
                                            <li><strong>Abnormal Regions:</strong> ${file.abnormal_regions}</li>
                                            <li><strong>Tumor Volume:</strong> ${file.tumor_volume_ml} mL</li>
                                        </ul>
                                    </div>
                                    <div class="col-md-6">
                                        <h6>Technical Details:</h6>
                                        <ul class="list-unstyled">
                                            <li><strong>Slices:</strong> ${file.slice_count}</li>
                                            <li><strong>Voxel Size:</strong> ${file.voxel_dimensions}</li>
                                            <li><strong>Risk Level:</strong> ${file.risk_level.toUpperCase()}</li>
                                        </ul>
                                    </div>
                                </div>
                                
                                <!-- Processing Stages -->
                                <div class="mt-3">
                                    <h6>Processing Pipeline:</h6>
                                    <div class="d-flex flex-wrap gap-2">
                                        ${file.processing_stages.map(stage => `
                                            <span class="badge bg-success">
                                                <i class="fas fa-check me-1"></i>${stage.stage.toUpperCase()}
                                            </span>
                                        `).join('')}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
        
        <!-- Medical Visualizations -->
        ${result.visualizations ? `
            <div class="mt-4">
                <h5><i class="fas fa-eye me-2"></i>Medical Visualizations</h5>
                <div class="row">
                    <div class="col-md-6 col-lg-3 mb-3">
                        <div class="card">
                            <div class="card-header bg-primary text-white text-center">
                                <small>Sagittal View</small>
                            </div>
                            <img src="data:image/png;base64,${result.visualizations.sagittal_view}" 
                                 class="card-img-bottom" alt="Sagittal View" style="height: 200px; object-fit: cover;">
                        </div>
                    </div>
                    <div class="col-md-6 col-lg-3 mb-3">
                        <div class="card">
                            <div class="card-header bg-success text-white text-center">
                                <small>Coronal View</small>
                            </div>
                            <img src="data:image/png;base64,${result.visualizations.coronal_view}" 
                                 class="card-img-bottom" alt="Coronal View" style="height: 200px; object-fit: cover;">
                        </div>
                    </div>
                    <div class="col-md-6 col-lg-3 mb-3">
                        <div class="card">
                            <div class="card-header bg-warning text-white text-center">
                                <small>Axial View</small>
                            </div>
                            <img src="data:image/png;base64,${result.visualizations.axial_view}" 
                                 class="card-img-bottom" alt="Axial View" style="height: 200px; object-fit: cover;">
                        </div>
                    </div>
                    <div class="col-md-6 col-lg-3 mb-3">
                        <div class="card">
                            <div class="card-header bg-danger text-white text-center">
                                <small>Segmentation</small>
                            </div>
                            <img src="data:image/png;base64,${result.visualizations.segmentation_overlay}" 
                                 class="card-img-bottom" alt="Segmentation" style="height: 200px; object-fit: cover;">
                        </div>
                    </div>
                </div>
            </div>
        ` : ''}
        
        <!-- Clinical Recommendations -->
        <div class="mt-4">
            <div class="card border-info">
                <div class="card-header bg-info text-white">
                    <h6 class="mb-0"><i class="fas fa-user-md me-2"></i>Clinical Recommendations</h6>
                </div>
                <div class="card-body">
                    <p class="mb-0">${summary.recommendations}</p>
                    <hr>
                    <small class="text-muted">
                        <strong>Model:</strong> ${details.model_version} | 
                        <strong>Analysis Time:</strong> ${new Date(details.analysis_timestamp).toLocaleString()}
                    </small>
                </div>
            </div>
        </div>
    `;
    
    const resultsContainer = document.getElementById('resultsContainer');
    if (resultsContainer) {
        resultsContainer.style.display = 'block';
    }
    
    // Update main metrics if they exist
    const confidenceMetric = document.getElementById('confidenceMetric');
    if (confidenceMetric) {
        confidenceMetric.textContent = (summary.confidence * 100).toFixed(1) + '%';
    }
    
    const filesMetric = document.getElementById('filesMetric');
    if (filesMetric) {
        filesMetric.textContent = details.files_processed;
    }
    
    console.log('Results displayed successfully');
}

function showError(message) {
    console.log('Showing error:', message);
    const resultContent = document.getElementById('resultsContent');
    if (resultContent) {
        resultContent.innerHTML = `
            <div class="alert alert-danger" role="alert">
                <i class="fas fa-exclamation-triangle"></i> Error: ${message}
            </div>
        `;
        
        const resultsContainer = document.getElementById('resultsContainer');
        if (resultsContainer) {
            resultsContainer.style.display = 'block';
        }
    } else {
        alert(`Error: ${message}`);
    }
}

// File display and drag-drop functionality
function setupFileHandling() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('files');
    const fileList = document.getElementById('fileList');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#667eea';
        uploadArea.style.backgroundColor = '#f0f4ff';
    });
    
    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#dee2e6';
        uploadArea.style.backgroundColor = 'transparent';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#dee2e6';
        uploadArea.style.backgroundColor = 'transparent';
        
        const files = e.dataTransfer.files;
        fileInput.files = files;
        displayFiles(files);
    });
    
    fileInput.addEventListener('change', (e) => {
        displayFiles(e.target.files);
    });
    
    function displayFiles(files) {
        console.log('Displaying files:', files.length);
        if (files.length === 0) return;
        
        let html = '<h6>Selected Files:</h6><ul class="list-group">';
        for (let file of files) {
            html += `<li class="list-group-item">${file.name} (${(file.size / 1024).toFixed(2)} KB)</li>`;
        }
        html += '</ul>';
        
        fileList.innerHTML = html;
        analyzeBtn.style.display = 'block';
        console.log('Analyze button shown');
    }
}

// Model Info Function
function showModelInfo() {
    console.log('Showing model info...');
    fetch('/model-info')
    .then(response => response.json())
    .then(data => {
        console.log('Model info data:', data);
        const arch = data.model_architecture || {};
        const status = data.model_status || {};
        const training = data.training_results || {};
        const evaluation = data.evaluation_results || {};
        const logs = data.training_logs || [];
        const files = data.model_files || [];
        const dataset = data.dataset_info || {};
        
        let html = `
            <div class="text-start">
                <div class="row">
                    <div class="col-md-6">
                        <h6 class="text-primary"><i class="fas fa-brain me-2"></i>Model Architecture</h6>
                        <table class="table table-sm">
                            <tr><td><strong>Name:</strong></td><td>${arch.name || 'N/A'}</td></tr>
                            <tr><td><strong>Type:</strong></td><td>${arch.type || 'N/A'}</td></tr>
                            <tr><td><strong>Classes:</strong></td><td>${arch.classes ? arch.classes.join(', ') : 'N/A'}</td></tr>
                            <tr><td><strong>Channels:</strong></td><td>${arch.input_channels || 'N/A'} → ${arch.output_channels || 'N/A'}</td></tr>
                            <tr><td><strong>Base Channels:</strong></td><td>${arch.base_channels || 'N/A'}</td></tr>
                        </table>
                        
                        <h6 class="text-success mt-3"><i class="fas fa-cog me-2"></i>Model Status</h6>
                        <table class="table table-sm">
                            <tr><td><strong>Loaded:</strong></td><td>${status.loaded ? '✅ Yes' : '❌ No'}</td></tr>
                            <tr><td><strong>Device:</strong></td><td>${status.device || 'N/A'}</td></tr>
                            <tr><td><strong>Device Name:</strong></td><td>${status.device_name || 'N/A'}</td></tr>
                            ${status.compute_capability ? `<tr><td><strong>Compute Capability:</strong></td><td>${status.compute_capability}</td></tr>` : ''}
                            ${status.memory_tier ? `<tr><td><strong>Memory Tier:</strong></td><td>${status.memory_tier}</td></tr>` : ''}
                            <tr><td><strong>GPU Memory Total:</strong></td><td>${status.gpu_memory_total || 'N/A'}</td></tr>
                            <tr><td><strong>GPU Memory Used:</strong></td><td>${status.gpu_memory_allocated || 'N/A'}</td></tr>
                            <tr><td><strong>GPU Memory Cached:</strong></td><td>${status.gpu_memory_cached || 'N/A'}</td></tr>
                            <tr><td><strong>Parameters:</strong></td><td>${status.parameters || 'N/A'}</td></tr>
                            <tr><td><strong>Mixed Precision:</strong></td><td>${status.mixed_precision || 'N/A'}</td></tr>
                            <tr><td><strong>TensorCore Support:</strong></td><td>${status.tensorcore_support || 'N/A'}</td></tr>
                            <tr><td><strong>Optimizations:</strong></td><td>${status.optimizations_enabled || 'N/A'}</td></tr>
                        </table>
                    </div>
                    
                    <div class="col-md-6">
                        <h6 class="text-warning"><i class="fas fa-chart-line me-2"></i>Training Results</h6>
                        <table class="table table-sm">`;
        
        if (Object.keys(training).length > 0) {
            html += `
                <tr><td><strong>Final Epoch:</strong></td><td>${training.final_epoch || 'N/A'}</td></tr>
                <tr><td><strong>Final Loss:</strong></td><td>${training.final_test_loss || 'N/A'}</td></tr>
                <tr><td><strong>Final Accuracy:</strong></td><td>${training.final_test_accuracy || 'N/A'}</td></tr>
                <tr><td><strong>Best Val Accuracy:</strong></td><td>${training.best_val_accuracy || 'N/A'}</td></tr>
                <tr><td><strong>Training Time:</strong></td><td>${training.training_time || 'N/A'}</td></tr>
                <tr><td><strong>Optimizer:</strong></td><td>${training.optimizer || 'N/A'}</td></tr>
                <tr><td><strong>Learning Rate:</strong></td><td>${training.learning_rate || 'N/A'}</td></tr>
                <tr><td><strong>Batch Size:</strong></td><td>${training.batch_size || 'N/A'}</td></tr>`;
        } else {
            html += `<tr><td colspan="2" class="text-muted">No training results available</td></tr>`;
        }
        
        html += `</table>
                        
                        <h6 class="text-info mt-3"><i class="fas fa-clipboard-check me-2"></i>Evaluation Results</h6>
                        <table class="table table-sm">`;
        
        if (Object.keys(evaluation).length > 0) {
            html += `
                <tr><td><strong>Test Accuracy:</strong></td><td>${evaluation.test_accuracy || 'N/A'}</td></tr>
                <tr><td><strong>Val Accuracy:</strong></td><td>${evaluation.validation_accuracy || 'N/A'}</td></tr>
                <tr><td><strong>Precision:</strong></td><td>${evaluation.precision || 'N/A'}</td></tr>
                <tr><td><strong>Recall:</strong></td><td>${evaluation.recall || 'N/A'}</td></tr>
                <tr><td><strong>F1 Score:</strong></td><td>${evaluation.f1_score || 'N/A'}</td></tr>
                <tr><td><strong>Test Samples:</strong></td><td>${evaluation.test_samples || 'N/A'}</td></tr>
                <tr><td><strong>Evaluation Time:</strong></td><td>${evaluation.evaluation_time || 'N/A'}</td></tr>`;
        } else {
            html += `<tr><td colspan="2" class="text-muted">No evaluation results available</td></tr>`;
        }
        
        html += `</table>
                    </div>
                </div>
                
                <div class="row mt-3">
                    <div class="col-md-6">
                        <h6 class="text-secondary"><i class="fas fa-database me-2"></i>Dataset Info</h6>
                        <table class="table table-sm">`;
        
        if (dataset.total_samples) {
            html += `
                <tr><td><strong>Total Samples:</strong></td><td>${dataset.total_samples}</td></tr>
                <tr><td><strong>Data Splits:</strong></td><td>Available</td></tr>
                <tr><td><strong>Processed Data:</strong></td><td>Available</td></tr>`;
        } else {
            html += `<tr><td colspan="2" class="text-muted">${dataset.status || 'Dataset info not available'}</td></tr>`;
        }
        
        html += `</table>
                    </div>
                    
                    <div class="col-md-6">
                        <h6 class="text-dark"><i class="fas fa-file me-2"></i>Model Files</h6>`;
        
        if (files.length > 0) {
            html += `<div style="max-height: 150px; overflow-y: auto;">`;
            files.forEach(file => {
                const date = new Date(file.modified * 1000).toLocaleDateString();
                html += `<small><strong>${file.name}</strong><br>Size: ${file.size_mb} MB | Modified: ${date}</small><br>`;
            });
            html += `</div>`;
        } else {
            html += `<small class="text-muted">No model files found</small>`;
        }
        
        html += `</div>
                </div>`;
        
        if (logs.length > 0) {
            html += `
                <div class="mt-3">
                    <h6 class="text-muted"><i class="fas fa-terminal me-2"></i>Recent Training Log</h6>
                    <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; max-height: 200px; overflow-y: auto; font-family: monospace; font-size: 0.8rem;">`;
            logs.forEach(log => {
                html += `${log}<br>`;
            });
            html += `</div>
                </div>`;
        }
        
        html += `</div>`;
        
        Swal.fire({
            title: '<i class="fas fa-info-circle text-primary"></i> Lokman-v2 Model Information',
            html: html,
            width: '90%',
            showConfirmButton: true,
            confirmButtonText: 'Close',
            confirmButtonColor: '#667eea',
            customClass: {
                popup: 'model-info-popup'
            }
        });
    })
    .catch(error => {
        console.error('Model info error:', error);
        Swal.fire({
            icon: 'error',
            title: 'Error',
            text: 'Could not load model information',
            confirmButtonColor: '#e74c3c'
        });
    });
}

// Initialize enhanced functionality when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing Lokman-v2 enhanced analysis...');
    
    setupFileHandling();
    handleAnalysisSubmission();
    
    // Make showModelInfo available globally
    window.showModelInfo = showModelInfo;
    
    // Health check
    fetch('/api/health')
    .then(response => response.json())
    .then(data => {
        console.log('Server health:', data);
    })
    .catch(error => {
        console.error('Health check failed:', error);
    });
    
    console.log('Lokman-v2 initialization complete!');
});