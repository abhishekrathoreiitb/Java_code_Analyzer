/**
 * Java RAG Analyzer - Professional Edition
 * Main JavaScript Application Logic
 */

// Application State
let appState = {
    sessionId: null,
    selectedModel: null,
    isProcessing: false,
    currentQuery: null,
    results: []
};

// Model Configurations
const modelConfigs = {
    openai: {
        name: 'OpenAI GPT',
        versions: ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'],
        requiresKey: true,
        endpoint: false
    },
    anthropic: {
        name: 'Anthropic Claude',
        versions: ['claude-3-5-sonnet-20241022', 'claude-3-haiku-20240307'],
        requiresKey: true,
        endpoint: false
    },
    google: {
        name: 'Google Gemini',
        versions: ['gemini-1.5-pro', 'gemini-1.5-flash'],
        requiresKey: true,
        endpoint: false
    },
    local: {
        name: 'Local Model',
        versions: ['llama3', 'codellama', 'mistral'],
        requiresKey: false,
        endpoint: true
    }
};

// Initialize Application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    loadSavedConfiguration();
});

/**
 * Initialize the application with default states
 */
function initializeApp() {
    updateConnectionStatus(false);
    updateAIStatus('No Model Selected');
    updateCodebaseStatus('No Codebase Loaded');
}

/**
 * Setup all event listeners for interactive elements
 */
function setupEventListeners() {
    // File upload
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');

    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // Temperature slider
    const tempSlider = document.getElementById('temperature');
    const tempValue = document.getElementById('tempValue');
    tempSlider.addEventListener('input', () => {
        tempValue.textContent = tempSlider.value;
    });

    // Query input
    const queryInput = document.getElementById('queryInput');
    queryInput.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            performAnalysis();
        }
    });
}

// ============================================================================
// TAB MANAGEMENT
// ============================================================================

/**
 * Switch between different tabs - FIXED VERSION
 * @param {string} tabName - Name of the tab to switch to
 * @param {HTMLElement} buttonElement - Button element that triggered the switch (optional)
 */
function switchTab(tabName, buttonElement = null) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(tabName + '-tab').classList.add('active');
    
    // Activate the correct button
    if (buttonElement) {
        // If called from click event, use the passed button element
        buttonElement.classList.add('active');
    } else {
        // If called programmatically, find the button by its onclick attribute
        const buttons = document.querySelectorAll('.tab-btn');
        buttons.forEach(btn => {
            const onclick = btn.getAttribute('onclick');
            if (onclick && onclick.includes(`'${tabName}'`)) {
                btn.classList.add('active');
            }
        });
    }
}

// ============================================================================
// MODEL SELECTION AND CONFIGURATION
// ============================================================================

/**
 * Select AI model and show configuration options
 * @param {string} modelType - Type of model (openai, anthropic, google, local)
 */
function selectModel(modelType) {
    // Clear previous selection
    document.querySelectorAll('.model-card').forEach(card => {
        card.classList.remove('selected');
    });

    // Select new model
    document.getElementById('model-' + modelType).classList.add('selected');
    appState.selectedModel = modelType;

    // Show configuration
    const configDiv = document.getElementById('model-config');
    const versionSelect = document.getElementById('modelVersion');
    const endpointConfig = document.getElementById('endpoint-config');

    configDiv.style.display = 'block';

    // Populate model versions
    versionSelect.innerHTML = '';
    modelConfigs[modelType].versions.forEach(version => {
        const option = document.createElement('option');
        option.value = version;
        option.textContent = version;
        versionSelect.appendChild(option);
    });

    // Show/hide endpoint configuration
    if (modelConfigs[modelType].endpoint) {
        endpointConfig.style.display = 'block';
    } else {
        endpointConfig.style.display = 'none';
    }

    updateAIStatus(modelConfigs[modelType].name + ' Selected');
}

/**
 * Save AI model configuration and create session
 */
async function saveConfiguration() {
    const apiKey = document.getElementById('apiKey').value.trim();
    const modelVersion = document.getElementById('modelVersion').value;
    const apiEndpoint = document.getElementById('apiEndpoint').value;

    if (!appState.selectedModel) {
        showNotification('Please select a model first', 'error');
        return;
    }

    if (modelConfigs[appState.selectedModel].requiresKey && !apiKey) {
        showNotification('API key is required for this model', 'error');
        return;
    }

    try {
        const response = await fetch('/api/sessions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                provider: appState.selectedModel,
                model: modelVersion,
                api_key: apiKey,
                endpoint: apiEndpoint
            })
        });

        if (response.ok) {
            const data = await response.json();
            appState.sessionId = data.session_id;
            
            updateConnectionStatus(true);
            updateAIStatus(modelConfigs[appState.selectedModel].name + ' Connected');
            
            showNotification('Configuration saved successfully!', 'success');
            
            // Save to localStorage
            localStorage.setItem('rag_config', JSON.stringify({
                provider: appState.selectedModel,
                model: modelVersion,
                endpoint: apiEndpoint
            }));
            
            // Switch to upload tab
            switchTab('upload');
        } else {
            throw new Error('Failed to create session');
        }
    } catch (error) {
        console.error('Configuration error:', error);
        showNotification('Configuration failed: ' + error.message, 'error');
    }
}

/**
 * Load previously saved configuration from localStorage
 */
function loadSavedConfiguration() {
    const saved = localStorage.getItem('rag_config');
    if (saved) {
        try {
            const config = JSON.parse(saved);
            selectModel(config.provider);
            document.getElementById('modelVersion').value = config.model;
            if (config.endpoint) {
                document.getElementById('apiEndpoint').value = config.endpoint;
            }
        } catch (error) {
            console.error('Failed to load saved configuration:', error);
        }
    }
}

// ============================================================================
// FILE UPLOAD MANAGEMENT
// ============================================================================

/**
 * Trigger file selection dialog
 */
function triggerFileSelect() {
    document.getElementById('fileInput').click();
}

/**
 * Handle file selection from input
 * @param {Event} event - File input change event
 */
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        displayFileInfo(file);
    }
}

/**
 * Handle drag over event
 * @param {Event} event - Drag over event
 */
function handleDragOver(event) {
    event.preventDefault();
    event.currentTarget.classList.add('dragover');
}

/**
 * Handle drag leave event
 * @param {Event} event - Drag leave event
 */
function handleDragLeave(event) {
    event.currentTarget.classList.remove('dragover');
}

/**
 * Handle file drop event
 * @param {Event} event - Drop event
 */
function handleDrop(event) {
    event.preventDefault();
    event.currentTarget.classList.remove('dragover');
    
    const file = event.dataTransfer.files[0];
    if (file && file.name.endsWith('.zip')) {
        document.getElementById('fileInput').files = event.dataTransfer.files;
        displayFileInfo(file);
    } else {
        showNotification('Please drop a ZIP file', 'error');
    }
}

/**
 * Display information about selected file
 * @param {File} file - Selected file object
 */
function displayFileInfo(file) {
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileSize').textContent = formatFileSize(file.size);
    document.getElementById('fileInfo').style.display = 'block';
}

/**
 * Clear selected file
 */
function clearFile() {
    document.getElementById('fileInput').value = '';
    document.getElementById('fileInfo').style.display = 'none';
    updateCodebaseStatus('No Codebase Loaded');
}

// ============================================================================
// CODEBASE PROCESSING
// ============================================================================

/**
 * Process uploaded codebase
 */
async function processCodebase() {
    if (!appState.sessionId) {
        showNotification('Please configure AI model first', 'error');
        switchTab('setup');
        return;
    }

    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        showNotification('Please select a file first', 'error');
        return;
    }

    appState.isProcessing = true;
    document.getElementById('processBtn').disabled = true;
    document.getElementById('progressContainer').style.display = 'block';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`/api/sessions/${appState.sessionId}/upload`, {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            showNotification('Upload successful! Processing...', 'success');
            pollProcessingStatus();
        } else {
            throw new Error('Upload failed');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showNotification('Upload failed: ' + error.message, 'error');
        appState.isProcessing = false;
        document.getElementById('processBtn').disabled = false;
        document.getElementById('progressContainer').style.display = 'none';
    }
}

/**
 * Poll processing status with regular intervals
 */
async function pollProcessingStatus() {
    if (!appState.isProcessing) return;

    try {
        const response = await fetch(`/api/sessions/${appState.sessionId}/status`);
        const data = await response.json();

        updateProgress(data.progress || 0, data.message || 'Processing...');

        if (data.status === 'completed') {
            appState.isProcessing = false;
            document.getElementById('processBtn').disabled = false;
            
            showNotification('Processing completed successfully!', 'success');
            updateCodebaseStatus('Codebase Loaded & Analyzed');
            
            if (data.stats) {
                displayStats(data.stats);
            }
            
            // Switch to analyze tab
            setTimeout(() => switchTab('analyze'), 1000);
            
        } else if (data.status === 'error') {
            appState.isProcessing = false;
            document.getElementById('processBtn').disabled = false;
            showNotification('Processing failed: ' + data.message, 'error');
            
        } else {
            setTimeout(pollProcessingStatus, 1000);
        }
    } catch (error) {
        console.error('Status check error:', error);
        setTimeout(pollProcessingStatus, 2000);
    }
}

/**
 * Update progress bar and text
 * @param {number} percentage - Progress percentage (0-100)
 * @param {string} message - Progress message
 */
function updateProgress(percentage, message) {
    document.getElementById('progressFill').style.width = percentage + '%';
    document.getElementById('progressText').textContent = message;
}

/**
 * Display codebase statistics
 * @param {Object} stats - Statistics object from server
 */
function displayStats(stats) {
    const statsGrid = document.getElementById('statsGrid');
    const statsContainer = document.getElementById('statsContainer');
    
    statsGrid.innerHTML = `
        <div class="stat-card">
            <div class="stat-value">${stats.total_files || 0}</div>
            <div class="stat-label">Java Files</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${(stats.total_lines || 0).toLocaleString()}</div>
            <div class="stat-label">Lines of Code</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${stats.total_classes || 0}</div>
            <div class="stat-label">Classes</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${stats.total_methods || 0}</div>
            <div class="stat-label">Methods</div>
        </div>
    `;
    
    if (stats.spring_features && stats.spring_features.length > 0) {
        statsGrid.innerHTML += `
            <div class="stat-card">
                <div class="stat-value">${stats.spring_features.length}</div>
                <div class="stat-label">Spring Features</div>
            </div>
        `;
    }
    
    if (stats.issues_found && stats.issues_found.length > 0) {
        statsGrid.innerHTML += `
            <div class="stat-card" style="background: linear-gradient(135deg, #ff6b6b, #ee5a52);">
                <div class="stat-value">${stats.issues_found.length}</div>
                <div class="stat-label">Issues Found</div>
            </div>
        `;
    }
    
    statsContainer.style.display = 'block';
}

// ============================================================================
// ANALYSIS MANAGEMENT
// ============================================================================

/**
 * Set query text and switch to analyze tab
 * @param {string} query - Query text to set
 */
function setQuery(query) {
    document.getElementById('queryInput').value = query;
    switchTab('analyze');
}

/**
 * Perform code analysis based on query
 */
async function performAnalysis() {
    const query = document.getElementById('queryInput').value.trim();
    const depth = document.getElementById('analysisDepth').value;
    const count = parseInt(document.getElementById('resultsCount').value);

    if (!query) {
        showNotification('Please enter a query', 'error');
        return;
    }

    if (!appState.sessionId) {
        showNotification('Please configure AI model and upload codebase first', 'error');
        return;
    }

    const analyzeBtn = document.getElementById('analyzeBtn');
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<div class="loading"></div>Analyzing...';

    try {
        const response = await fetch(`/api/sessions/${appState.sessionId}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                top_k: count,
                depth: depth
            })
        });

        if (response.ok) {
            const result = await response.json();
            displayResults(query, result);
            switchTab('results');
        } else {
            throw new Error('Analysis failed');
        }
    } catch (error) {
        console.error('Analysis error:', error);
        showNotification('Analysis failed: ' + error.message, 'error');
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-search"></i>Analyze Code';
    }
}

/**
 * Display analysis results
 * @param {string} query - Original query
 * @param {Object} result - Analysis result from server
 */
function displayResults(query, result) {
    const container = document.getElementById('resultsContainer');
    const timestamp = new Date().toLocaleString();
    
    const resultHtml = `
        <div class="result-card">
            <div class="result-header">
                <h3><i class="fas fa-search"></i> Query: ${query}</h3>
                <small><i class="fas fa-clock"></i> ${timestamp}</small>
            </div>
            <div class="result-content">${result.answer}</div>
            
            ${result.sources && result.sources.length > 0 ? `
                <div class="sources-section">
                    <h4><i class="fas fa-code"></i> Source Files Referenced</h4>
                    ${result.sources.map(source => `
                        <div class="source-item">
                            <strong>${source.file_path ? source.file_path.split('/').pop() : 'Unknown File'}</strong>
                            <br><small>Classes: ${source.classes ? source.classes.length : 0} | 
                            Methods: ${source.methods ? source.methods.length : 0} | 
                            Type: ${source.file_type || 'Unknown'}</small>
                        </div>
                    `).join('')}
                </div>
            ` : ''}
        </div>
    `;
    
    container.innerHTML = resultHtml;
    appState.results.unshift({ query, result, timestamp });
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Update connection status indicator
 * @param {boolean} connected - Connection status
 */
function updateConnectionStatus(connected) {
    const status = document.getElementById('connectionStatus');
    const text = document.getElementById('connectionText');
    
    if (connected) {
        status.classList.add('connected');
        text.textContent = 'Connected';
    } else {
        status.classList.remove('connected');
        text.textContent = 'Not Connected';
    }
}

/**
 * Update AI model status text
 * @param {string} status - Status text to display
 */
function updateAIStatus(status) {
    document.getElementById('aiModelStatus').textContent = status;
}

/**
 * Update codebase status text
 * @param {string} status - Status text to display
 */
function updateCodebaseStatus(status) {
    document.getElementById('codebaseStatus').textContent = status;
}

/**
 * Format file size in human readable format
 * @param {number} bytes - File size in bytes
 * @returns {string} Formatted file size
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Show notification to user
 * @param {string} message - Message to display
 * @param {string} type - Notification type (info, success, error, warning)
 */
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        ${message}
        <button onclick="this.parentElement.remove()" style="float: right; background: none; border: none; color: inherit; font-size: 18px; cursor: pointer;">&times;</button>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => notification.classList.add('show'), 100);
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}

// ============================================================================
// EXPORT FUNCTIONS
// ============================================================================

/**
 * Export analysis results as JSON file
 */
function exportResults() {
    const data = {
        results: appState.results,
        exportTime: new Date().toISOString(),
        version: '1.0.0'
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `java-rag-results-${Date.now()}.json`;
    a.click();
    
    URL.revokeObjectURL(url);
    showNotification('Results exported successfully!', 'success');
}

/**
 * Export current configuration as JSON file
 */
function exportConfiguration() {
    const config = {
        model: appState.selectedModel,
        settings: {
            temperature: document.getElementById('temperature').value,
            maxTokens: document.getElementById('maxTokens').value,
            contextSize: document.getElementById('contextSize').value,
            enableCache: document.getElementById('enableCache').checked,
            enableLogging: document.getElementById('enableLogging').checked,
            enableMetrics: document.getElementById('enableMetrics').checked
        }
    };
    
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `java-rag-config-${Date.now()}.json`;
    a.click();
    
    URL.revokeObjectURL(url);
    showNotification('Configuration exported successfully!', 'success');
}

/**
 * Clear all application data
 */
function clearAllData() {
    if (confirm('Are you sure you want to clear all data? This cannot be undone.')) {
        localStorage.clear();
        appState = {
            sessionId: null,
            selectedModel: null,
            isProcessing: false,
            currentQuery: null,
            results: []
        };
        
        // Reset UI
        document.getElementById('resultsContainer').innerHTML = `
            <div style="text-align: center; padding: 60px 20px; color: #6c757d;">
                <i class="fas fa-search" style="font-size: 4rem; margin-bottom: 20px;"></i>
                <h3>No Analysis Results Yet</h3>
                <p>Upload a codebase and run an analysis to see results here.</p>
            </div>
        `;
        
        updateConnectionStatus(false);
        updateAIStatus('No Model Selected');
        updateCodebaseStatus('No Codebase Loaded');
        
        showNotification('All data cleared successfully!', 'success');
        switchTab('setup');
    }
}