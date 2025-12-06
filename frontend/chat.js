/**
 * ChatGPT-Style Chat Interface with Sticky Header
 * Clean, simple, and robust chat functionality
 */

class ChatInterface {
    constructor() {
        this.isLoading = false;
        this.messages = [];
        this.currentRequestController = null; // For tracking the current request
        this.isCancelled = false; // Track if current request was cancelled
        this.streamingMessageId = null; // Track streaming message for token-by-token rendering
        this.scrollScheduled = false; // Track if scroll is scheduled for throttling
        this.isStreaming = false; // Track if we're actively streaming content
        this.lastScrollTime = 0; // Track last scroll time for debouncing
        this.lastScrollLogTime = 0;
        this.lastMarkdownRenderTime = 0; // Track last markdown render time for throttling
        this.tokensSinceLastRender = 0; // Track tokens since last markdown render
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadChatHistory();
        // Ensure initial button state is correct
        this.updateButtonStates();
    }

    setupEventListeners() {
        const chatInput = document.getElementById('chatInput');
        const sendButton = document.getElementById('sendChatButton');
        const clearButton = document.getElementById('clearChatBtn');
        const newChatButton = document.getElementById('newChatNavBtn'); // Using the correct ID from navbar
        const historyButton = document.getElementById('chatHistoryNavBtn'); // Using the correct ID from navbar  
        const statsButton = document.getElementById('chatStatsBtn');
        const exportButton = document.getElementById('chatExportBtn');

        if (!chatInput || !sendButton) {
            return;
        }

        // Set up initial send button event listener
        sendButton.onclick = () => {
            this.sendMessage();
        };

        // Send message on Enter key and handle ticker autocomplete navigation
        chatInput.addEventListener('keydown', (e) => {
            // Handle ticker autocomplete navigation first
            if (this.handleTickerAutocompleteKeydown(e)) {
                return; // Don't process other keys if autocomplete handled it
            }
            
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auto-resize textarea and handle ticker autocomplete
        chatInput.addEventListener('input', () => {
            chatInput.style.height = 'auto';
            chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
            
            // Handle ticker autocomplete
            this.handleTickerAutocomplete();
        });

        // Setup compact mode selector
        // Mode selector removed - always use agent mode

        // New chat
        if (newChatButton) {
            newChatButton.addEventListener('click', () => this.startNewChat());
        }

        // Clear chat (optional button)
        if (clearButton) {
            clearButton.addEventListener('click', () => this.clearChat());
        } else {
        }

        // Chat history modal
        if (historyButton) {
            historyButton.addEventListener('click', () => this.showHistoryModal());
        }

        // Chat statistics modal
        if (statsButton) {
            statsButton.addEventListener('click', () => this.showStatsModal());
        }

        // Export chat modal
        if (exportButton) {
            exportButton.addEventListener('click', () => this.showExportModal());
        }

        // Setup modal event listeners
        this.setupModalEventListeners();

    }

    setupModalEventListeners() {
        // These will be set up when modals are opened since the elements don't exist yet
    }

    setupModeSelector(toggleBtnId, dropdownId, currentModeId) {
        const toggleBtn = document.getElementById(toggleBtnId);
        const dropdown = document.getElementById(dropdownId);
        const currentMode = document.getElementById(currentModeId);
        
        if (!toggleBtn || !dropdown) {
            return;
        }

        // Toggle dropdown
        toggleBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            
            const isOpen = dropdown.classList.contains('show');
            dropdown.classList.toggle('show');
            toggleBtn.setAttribute('aria-expanded', !isOpen);
        });

        // Handle mode selection
        const modeItems = dropdown.querySelectorAll('.mode-item');
        modeItems.forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                
                // Skip if disabled
                if (item.classList.contains('mode-disabled')) {
                    return;
                }
                
                const mode = item.getAttribute('data-mode');
                
                // Update UI
                modeItems.forEach(i => i.classList.remove('active'));
                item.classList.add('active');
                
                // Update button
                const icon = mode === 'agent' ? 'fa-magnifying-glass-chart' : 'fa-bolt';
                const text = 'Agent'; // Always agent mode
                
                const iconEl = toggleBtn.querySelector('i:first-child');
                if (iconEl) iconEl.className = `fas ${icon}`;
                
                if (currentMode) currentMode.textContent = text;
                
                // Close dropdown
                dropdown.classList.remove('show');
                toggleBtn.setAttribute('aria-expanded', 'false');
                
            });
        });

        // Close on outside click
        document.addEventListener('click', (e) => {
            if (!toggleBtn.contains(e.target) && !dropdown.contains(e.target)) {
                dropdown.classList.remove('show');
                toggleBtn.setAttribute('aria-expanded', 'false');
            }
        });

        
        // Verify initial state
        setTimeout(() => {
            const activeItem = document.querySelector('#modeDropdown .mode-item.active');
        }, 100);
    }

    getCurrentMode() {
        const activeItem = document.querySelector('#modeDropdown .mode-item.active');
        const mode = 'agent'; // Always agent mode
        
        
        return mode;
    }

    startNewChat() {
        // Use the global startNewChat function for consistency
        if (typeof window.startNewChat === 'function') {
            window.startNewChat();
        } else {
            // Fallback implementation
            
            // Clear the current conversation ID to start fresh
            window.currentConversationId = null;
            
        // Clear the current chat without backend call
        const messagesContainer = document.getElementById('chatMessages');
        if (messagesContainer) {
            messagesContainer.innerHTML = `
                <div class="chat-message assistant">
                    <div class="chat-content">
                        <div class="chat-avatar assistant">
                            <i class="fas fa-chart-line"></i>
                        </div>
                        <div class="message-text-wrapper">
                            <div class="chat-bubble">
                                <div class="welcome-content">
                                    <h2>Welcome to StrataLens</h2>
                                    <p>Your AI analyst for public equity markets.</p>
                                    
                                    <h3 class="quick-suggestions-heading">Quick Suggestions</h3>
                                    <div class="quick-examples">
                                        <button class="example-btn" onclick="document.getElementById('chatInput').value = 'Compare $MSFT and $GOOGL cloud segment'; document.getElementById('chatInput').focus();">
                                            Compare $MSFT and $GOOGL cloud segment
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Clear local messages array
        this.messages = [];
        
        // Focus on input
        const chatInput = document.getElementById('chatInput');
        if (chatInput) {
            chatInput.focus();
        }
        
        }
    }

    showHistoryModal() {
        // Use the global chat history modal function
        if (typeof window.openChatHistoryModal === 'function') {
            window.openChatHistoryModal();
        } else {
        }
    }

    hideHistoryModal() {
        // Use the global function
        if (typeof window.closeChatHistoryModal === 'function') {
            window.closeChatHistoryModal();
        } else {
        }
    }

    showStatsModal() {
        const modal = document.getElementById('chatStatsModal');
        if (!modal) {
            return;
        }

        modal.classList.remove('hidden');
        modal.classList.add('flex');

        // Set up modal event listeners
        const closeBtn = document.getElementById('closeChatStatsModalBtn');
        if (closeBtn) {
            closeBtn.onclick = () => this.hideStatsModal();
        }

        this.loadStatsData();
    }

    hideStatsModal() {
        const modal = document.getElementById('chatStatsModal');
        if (modal) {
            modal.classList.add('hidden');
            modal.classList.remove('flex');
        }
    }

    showExportModal() {
        const modal = document.getElementById('chatExportModal');
        if (!modal) {
            return;
        }

        modal.classList.remove('hidden');
        modal.classList.add('flex');

        // Set up modal event listeners
        const closeBtn = document.getElementById('closeChatExportModalBtn');
        const cancelBtn = document.getElementById('cancelChatExportBtn');
        const exportBtn = document.getElementById('exportChatBtn');

        if (closeBtn) {
            closeBtn.onclick = () => this.hideExportModal();
        }

        if (cancelBtn) {
            cancelBtn.onclick = () => this.hideExportModal();
        }

        if (exportBtn) {
            exportBtn.onclick = (e) => this.performExport(e);
        }
    }

    hideExportModal() {
        const modal = document.getElementById('chatExportModal');
        if (modal) {
            modal.classList.add('hidden');
            modal.classList.remove('flex');
        }
    }

    async loadStatsData() {
        const loading = document.getElementById('chatStatsLoading');
        const content = document.getElementById('chatStatsContent');

        if (loading) loading.classList.remove('hidden');
        if (content) content.classList.add('hidden');

        try {
            const response = await fetch(`${CONFIG.apiBaseUrl}/chat/stats`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('authToken')}`
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.success) {
                this.renderStats(data);
            } else {
                throw new Error(data.error || 'Failed to load statistics');
            }

        } catch (error) {
            if (content) {
                content.innerHTML = `
                    <div class="text-center py-8">
                        <i class="fas fa-exclamation-triangle text-red-500 text-2xl mb-2"></i>
                        <p class="text-text-secondary">Failed to load statistics</p>
                        <button class="btn btn-primary mt-2" onclick="chatInterface.loadStatsData()">
                            <i class="fas fa-redo mr-2"></i>Try Again
                        </button>
                    </div>
                `;
                content.classList.remove('hidden');
            }
        } finally {
            if (loading) loading.classList.add('hidden');
        }
    }

    renderStats(data) {
        const content = document.getElementById('chatStatsContent');
        if (!content) return;

        // Update basic stats
        const totalMessages = document.getElementById('totalChatMessages');
        const activeDays = document.getElementById('activeChatDays');
        const avgQuestionLength = document.getElementById('avgQuestionLength');

        if (totalMessages) totalMessages.textContent = data.stats.total_messages;
        if (activeDays) activeDays.textContent = data.stats.active_days;
        if (avgQuestionLength) avgQuestionLength.textContent = Math.round(data.stats.avg_question_length);

        // Render top conversations
        const topConversations = document.getElementById('topConversations');
        if (topConversations && data.top_conversations) {
            topConversations.innerHTML = '';
            
            data.top_conversations.slice(0, 5).forEach((conversation, index) => {
                const item = document.createElement('div');
                item.className = 'conversation-item';
                
                const date = new Date(conversation.created_at).toLocaleDateString();
                
                item.innerHTML = `
                    <div class="conversation-preview">${this.escapeHtml(conversation.topic_preview)}</div>
                    <div class="conversation-meta">
                        <span>${date}</span>
                        <span>${conversation.response_length} chars</span>
                    </div>
                `;
                
                topConversations.appendChild(item);
            });
        }

        content.classList.remove('hidden');
    }

    async performExport(e) {
        e.preventDefault();
        
        const formatSelect = document.getElementById('chatExportFormat');
        const dateFromInput = document.getElementById('chatExportDateFrom');
        const dateToInput = document.getElementById('chatExportDateTo');
        const includeCitationsInput = document.getElementById('chatExportIncludeCitations');
        const exportBtn = document.getElementById('exportChatBtn');
        const exportBtnText = document.getElementById('exportChatBtnText');
        const exportSpinner = document.getElementById('exportChatSpinner');

        if (!formatSelect) return;

        // Show loading state
        if (exportBtnText) exportBtnText.textContent = 'Exporting...';
        if (exportSpinner) exportSpinner.classList.remove('hidden');
        if (exportBtn) exportBtn.disabled = true;

        try {
            const requestBody = {
                format: formatSelect.value,
                include_citations: includeCitationsInput ? includeCitationsInput.checked : true
            };

            if (dateFromInput && dateFromInput.value) {
                requestBody.date_from = dateFromInput.value;
            }

            if (dateToInput && dateToInput.value) {
                requestBody.date_to = dateToInput.value;
            }

            const response = await fetch(`${CONFIG.apiBaseUrl}/chat/export`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('authToken')}`
                },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.success) {
                this.downloadExportedData(data, formatSelect.value);
                this.hideExportModal();
            } else {
                throw new Error(data.error || 'Export failed');
            }

        } catch (error) {
            if (typeof showToast === 'function') {
                showToast(`Export failed: ${error.message}`, 'error');
            }
        } finally {
            // Reset loading state
            if (exportBtnText) exportBtnText.textContent = 'Export';
            if (exportSpinner) exportSpinner.classList.add('hidden');
            if (exportBtn) exportBtn.disabled = false;
        }
    }

    downloadExportedData(data, format) {
        let content, filename, mimeType;

        const dateStr = new Date().toISOString().split('T')[0];

        switch (format) {
            case 'json':
                content = JSON.stringify(data.data, null, 2);
                filename = `chat-history-${dateStr}.json`;
                mimeType = 'application/json';
                break;
            case 'csv':
                content = data.content;
                filename = `chat-history-${dateStr}.csv`;
                mimeType = 'text/csv';
                break;
            case 'txt':
                content = data.content;
                filename = `chat-history-${dateStr}.txt`;
                mimeType = 'text/plain';
                break;
            default:
                throw new Error('Unsupported format');
        }

        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        link.click();
        
        URL.revokeObjectURL(url);
    }

    async sendMessage() {
        const chatInput = document.getElementById('chatInput');
        const message = chatInput.value.trim();


        if (!message || this.isLoading) {
            return;
        }

        // Check character limit
        if (message.length > 4000) {
            this.addMessage('assistant', 
                'Your message is too long! Please keep it under 4000 characters. ' +
                `Current length: ${message.length} characters.`, [], 'error');
            return;
        }

        // Set loading state
        this.isLoading = true;
        this.isCancelled = false; // Reset cancellation flag
        this.isStreaming = true; // Set streaming flag for optimized scrolling
        this.updateButtonStates();

        // Clear input
        chatInput.value = '';
        chatInput.style.height = 'auto';

        // Add user message
        this.addMessage('user', message);

        // DON'T show typing indicator - we use reasoning panel now
        // this.showTypingIndicator();
        
        // Show reasoning panel IMMEDIATELY (before sending request)
        this.createProgressStepsContainer();

        // Send to backend
        try {
            // Check if WebSocket is available for streaming
            if (window.wsClient && window.wsClient.isConnected()) {
                await this.sendViaWebSocket(message);
            } else {
                const response = await this.sendToBackend(message);
                
                // Check if request was cancelled before displaying response
                if (this.isCancelled) {
                    return;
                }
                
                this.hideTypingIndicator();
                // Message is already added by handleJSONResponse, no need to add again
            }
        } catch (error) {
            // Check if request was cancelled before showing error
            if (this.isCancelled) {
                return;
            }
            
            this.hideTypingIndicator();
            
            // Handle demo limit specifically
            if (error.message.startsWith('DEMO_LIMIT:')) {
                this.addMessage('assistant',
                    `Your free requests for today have been used. Please try again tomorrow.`, [], 'info');
            } else {
                this.addMessage('assistant', `Sorry, I encountered an error: ${error.message}`, [], 'error');
            }
        } finally {
            // Reset loading state and clear request controller
            this.isLoading = false;
            this.isCancelled = false; // Reset cancellation flag
            // DON'T reset isStreaming here - it should be reset when streaming actually completes
            this.currentRequestController = null;
            this.updateButtonStates();
        }
    }

    async stopCurrentQuery() {
        
        if (!this.isLoading) {
            return;
        }
        
        // Prevent multiple simultaneous cancellation attempts
        if (this.isCancelled) {
            return;
        }
        
        this.isCancelled = true;
        
        // Close EventSource if active
        if (this.currentEventSource) {
            this.currentEventSource.close();
            this.currentEventSource = null;
        }
        
        // Remove progress steps immediately (no delay for cancellation)
        const panel = document.getElementById('chat-inline-reasoning-panel');
        if (panel && panel.parentNode) {
            panel.parentNode.removeChild(panel);
        }
        
        // Immediately show cancellation to user - don't wait for backend
        this.hideTypingIndicator();
        this.addMessage('assistant', 'Request cancelled by user.', [], 'cancelled');
        
        // Reset loading state immediately for better UX
        this.isLoading = false;
        this.isCancelled = true;
        this.isStreaming = false; // Reset streaming flag
        this.currentRequestController = null;
        this.updateButtonStates();

        // Call backend cancellation in the background (don't await it)
        this.cancelBackendRequest();
    }
    
    async cancelBackendRequest() {
        try {
            const response = await fetch(`${CONFIG.apiBaseUrl}/chat/cancel`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('authToken')}`
                }
            });


            if (response.ok) {
                const result = await response.json();
                if (result.success) {
                } else {
                }
            } else {
                try {
                    const errorText = await response.text();
                } catch (e) {
                }
            }
        } catch (error) {
        }
    }

    updateButtonStates() {
        const sendButton = document.getElementById('sendChatButton');
        const chatInput = document.getElementById('chatInput');


        if (this.isLoading) {
            // Convert to stop button
            if (sendButton) {
                sendButton.classList.remove('send-button');
                sendButton.classList.add('stop-button');
                sendButton.innerHTML = '<i class="fas fa-stop"></i>';
                sendButton.title = 'Stop current query';
                // Update the onclick handler
                sendButton.onclick = () => {
                    this.stopCurrentQuery();
                };
            } else {
            }
            if (chatInput) {
                chatInput.disabled = true;
            }
        } else {
            // Convert back to send button
            if (sendButton) {
                sendButton.classList.remove('stop-button');
                sendButton.classList.add('send-button');
                sendButton.innerHTML = '<i class="fas fa-paper-plane"></i>';
                sendButton.title = 'Send message';
                sendButton.disabled = false; // Re-enable the button
                // Update the onclick handler
                sendButton.onclick = () => {
                    this.sendMessage();
                };
            } else {
            }
            if (chatInput) {
                chatInput.disabled = false;
            }
        }
    }


    async sendViaWebSocket(message) {
        return new Promise((resolve, reject) => {
            if (!window.wsClient || !window.wsClient.isConnected()) {
                reject(new Error('WebSocket not connected'));
                return;
            }
            
            // Set up one-time event listeners for this request
            const handleStream = (data) => {
                // The streaming is handled by the global WebSocket handlers
            };
            
            const handleProgress = (data) => {
                // The progress is handled by the global WebSocket handlers
            };
            
            const handleResult = (data) => {
                this.hideTypingIndicator();
                
                // Clean up event listeners
                window.wsClient.off('chat_stream', handleStream);
                window.wsClient.off('chat_progress', handleProgress);
                window.wsClient.off('chat_result', handleResult);
                window.wsClient.off('chat_error', handleError);
                
                resolve(data);
            };
            
            const handleError = (data) => {
                this.hideTypingIndicator();
                
                // Clean up event listeners
                window.wsClient.off('chat_stream', handleStream);
                window.wsClient.off('chat_progress', handleProgress);
                window.wsClient.off('chat_result', handleResult);
                window.wsClient.off('chat_error', handleError);
                
                reject(new Error(data.message || 'Chat error'));
            };
            
            // Add event listeners
            window.wsClient.on('chat_stream', handleStream);
            window.wsClient.on('chat_progress', handleProgress);
            window.wsClient.on('chat_result', handleResult);
            window.wsClient.on('chat_error', handleError);
            
            // Get selected mode (backend determines iterations)
            const mode = this.getCurrentMode();
            
            
            // Send the chat message via WebSocket
            const chatMessage = {
                type: 'chat_message',
                message: message,
                comprehensive: true,
                mode: mode
            };
            
            window.wsClient.send(chatMessage);
        });
    }

    async sendToBackend(message) {
        // Check if user is authenticated
        const token = localStorage.getItem('authToken');
        const isAuthenticated = token && localStorage.getItem('currentUser');
        
        // Use streaming endpoints for both authenticated and demo users
        return await this.sendToBackendWithStreaming(message, isAuthenticated);
    }

    async sendToBackendWithStreaming(message, isAuthenticated) {
        return new Promise(async (resolve, reject) => {
            let progressData = {
                analysis: null,
                search: null,
                generation: null
            };
            
            try {
                // Get selected mode (backend determines iterations)
                const mode = this.getCurrentMode();
                
                
                // Build streaming endpoint and request body
                let streamUrl;
                let requestBody;
                let headers = {
                    'Content-Type': 'application/json'
                };
                
                if (isAuthenticated) {
                    const token = localStorage.getItem('authToken');
                    headers['Authorization'] = `Bearer ${token}`;
                    streamUrl = `${CONFIG.apiBaseUrl}/chat/message/stream-v2`;
                    requestBody = {
                        message: message,
                        comprehensive: true,
                        mode: mode,
                        conversation_id: window.currentConversationId || null
                    };
                } else {
                    // Get session ID for demo users
                    const sessionId = window.SessionManager ? 
                        window.SessionManager.getOrCreateSessionId() : 
                        `session_${Date.now()}`;
                    
                    streamUrl = `${CONFIG.apiBaseUrl}/chat/landing/demo/stream-v2`;
                    requestBody = {
                        message: message,
                        comprehensive: true,
                        mode: mode,
                        session_id: sessionId,
                        conversation_id: window.currentConversationId || null
                    };
                }
                
                
                // Progress panel already created above - don't create again
                
                // Use fetch with POST for streaming
                const response = await fetch(streamUrl, {
                    method: 'POST',
                    headers: headers,
                    body: JSON.stringify(requestBody)
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                // Create a reader for the response stream
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                
                // Process streaming response
                const processStream = async () => {
                    try {
                        while (true) {
                            const { done, value } = await reader.read();
                            
                            if (done) {
                                break;
                            }
                            
                            // Decode the chunk and add to buffer
                            buffer += decoder.decode(value, { stream: true });
                            
                            // Process complete lines (SSE format: "data: {...}\n\n")
                            const lines = buffer.split('\n\n');
                            buffer = lines.pop() || ''; // Keep incomplete line in buffer
                            
                            for (const line of lines) {
                                if (!line.trim()) continue;
                                
                                // Remove "data: " prefix if present
                                const jsonStr = line.startsWith('data: ') ? line.substring(6) : line;
                                
                                try {
                                    const data = JSON.parse(jsonStr);
                        
                                    // Catch-all filter: Skip any event with evaluation reasoning text
                                    if (data.message && typeof data.message === 'string') {
                                        const msg = data.message;
                                        // Skip if it contains evaluation reasoning patterns
                                        // Check for common evaluation phrases
                                        const hasEvaluationPattern = (
                                            (msg.includes('The answer') || msg.includes('The response')) &&
                                            (msg.includes('lacks') || msg.includes('gaps') || msg.includes('missing') || 
                                             msg.includes('does not provide') || msg.includes('omits') || msg.includes('it omits') ||
                                             msg.includes('prevent the answer') || msg.includes('gaps prevent') ||
                                             msg.includes('omits critical') || msg.includes('omits key') ||
                                             msg.includes('lacks direct') || msg.includes('lacks explicit'))
                                        ) || (
                                            // Also catch patterns that start with evaluation language
                                            msg.includes('omits critical') || msg.includes('omits key') ||
                                            msg.includes('lacks direct') || msg.includes('lacks explicit') ||
                                            (msg.includes('omits') && (msg.includes('quantitative') || msg.includes('details'))) ||
                                            (msg.includes('gaps prevent') || msg.includes('prevent a full'))
                                        );
                                        
                                        if (hasEvaluationPattern) {
                                            // This is evaluation reasoning, skip this event entirely
                                            console.log('🚫 Filtered evaluation reasoning:', msg.substring(0, 100));
                                            continue;
                                        }
                                    }
                                    // Also check data fields for evaluation reasoning
                                    if (data.data) {
                                        if (data.data.reasoning && typeof data.data.reasoning === 'string') {
                                            const reasoning = data.data.reasoning;
                                            if (reasoning.includes('The answer') && 
                                                (reasoning.includes('lacks') || reasoning.includes('gaps') || 
                                                 reasoning.includes('missing') || reasoning.includes('omits'))) {
                                                // Skip this event
                                                continue;
                                            }
                                        }
                                        if (data.data.evaluation_summary && typeof data.data.evaluation_summary === 'string') {
                                            // Skip evaluation summary
                                            continue;
                                        }
                                        if (data.data.iteration_reasoning && typeof data.data.iteration_reasoning === 'string') {
                                            const iterReasoning = data.data.iteration_reasoning;
                                            if (iterReasoning.includes('The answer') && 
                                                (iterReasoning.includes('lacks') || iterReasoning.includes('gaps'))) {
                                                // Skip this event
                                                continue;
                                            }
                                        }
                                    }
                        
                        if (data.type === 'progress') {
                            // Handle progress updates
                            this.handleProgressEvent(data);
                        } else if (data.type === 'analysis') {
                            // Handle analysis updates
                            this.handleAnalysisEvent(data);
                            progressData.analysis = data.data;
                        } else if (data.type === 'search') {
                            // Handle search updates
                            this.handleSearchEvent(data);
                            progressData.search = data.data;
                        } else if (data.type === 'iteration_start') {
                            // Handle iteration start
                            this.handleIterationStartEvent(data);
                        } else if (data.type === 'iteration_eval') {
                            // Handle iteration evaluation
                            this.handleIterationEvalEvent(data);
                        } else if (data.type === 'agent_decision') {
                            // Handle agent's iteration decision and reasoning
                            this.handleAgentDecisionEvent(data);
                        } else if (data.type === 'master_thinking') {
                            // Handle master agent thinking and reasoning
                            this.handleMasterThinkingEvent(data);
                        } else if (data.type === 'iteration_followup') {
                            // Handle iteration follow-up question
                            this.handleIterationFollowupEvent(data);
                        } else if (data.type === 'iteration_search') {
                            // Handle iteration search results
                            this.handleIterationSearchEvent(data);
                        } else if (data.type === 'iteration_news_search') {
                            // Handle iteration news search
                            this.handleIterationNewsSearchEvent(data);
                        } else if (data.type === 'iteration_complete') {
                            // Handle iteration completion (early stop)
                            this.handleIterationCompleteEvent(data);
                        } else if (data.type === 'iteration_final') {
                            // Handle final answer generation after iterations
                            this.handleIterationFinalEvent(data);
                        } else if (data.type === 'token') {
                            // Handle token streaming - append characters in real-time
                            if (!this.streamingMessageId) {
                                // Create streaming message container on first token
                                this.streamingMessageId = 'streaming-' + Date.now();
                                this.createStreamingMessage(this.streamingMessageId);
                                
                                const createdMsg = document.getElementById(this.streamingMessageId);
                                const bubble = document.getElementById(`${this.streamingMessageId}-bubble`);
                            }
                            // Append token to streaming message
                            this.appendTokenToMessage(this.streamingMessageId, data.content);
                        } else if (data.type === 'result') {
                            // Handle final result
                            
                            const resultData = data.data;
                            const responseData = resultData.response || resultData;  // Handle both nested and flat structures
                            
                            // Update conversation ID from streaming response if provided
                            if (data.conversation_id) {
                                window.currentConversationId = data.conversation_id;
                            }
                            
                            // Update progress bar to 100%
                            const progressBar = document.getElementById('chat-inline-progress-bar');
                            if (progressBar) {
                                progressBar.style.width = '100%';
                            }
                            
                            // Mark generation as complete
                            this.updateChatPanelStatus('complete', 'Response generated');
                            
                            // Update the "Generating response" step to completed (if it exists)
                            const generationStep = document.getElementById('chat-step-generation');
                            if (generationStep) {
                                this.updateChatReasoningStep('generation', 'Response generated', 'completed');
                            } else {
                                // If it doesn't exist, add it as completed
                            this.addChatReasoningStep('generation', 'Response generated', 'completed');
                            }
                            
                            // Process citations to match expected format
                            const rawCitations = responseData.citations || resultData.citations || [];
                            console.log('📎 Raw citations received:', rawCitations);
                            console.log('📎 Citations count:', rawCitations.length);
                            console.log('📎 News citations:', rawCitations.filter(c => c.type === 'news'));

                            const citations = this.processCitations(
                                rawCitations,
                                resultData.chunks || []
                            );

                            console.log('📎 Processed citations:', citations);
                            console.log('📎 Processed news citations:', citations.filter(c => c.type === 'news'));
                            
                            // Extract reasoning trace HTML from the panel before removing it
                            const reasoningSteps = document.getElementById('chat-inline-reasoning-steps');
                            let reasoningHTML = '';
                            if (reasoningSteps) {
                                reasoningHTML = reasoningSteps.innerHTML;
                            }
                            
                            // If we were streaming tokens, finalize that message with citations
                            if (this.streamingMessageId) {
                                
                                // Move the streaming message to the main chat container first
                                this.moveStreamingMessageToMainChat(this.streamingMessageId);
                                
                                // Then finalize it with citations and reasoning
                                this.finalizeStreamingMessage(this.streamingMessageId, citations, reasoningHTML);
                                
                                this.streamingMessageId = null;
                            } else {
                                // No token streaming, add complete message
                                const answer = responseData.answer || resultData.answer || 'No answer generated';
                                this.addMessage('assistant', answer, citations, 'normal', reasoningHTML);
                            }

                            // Remove the standalone reasoning panel after finalizing/migrating content
                            const panel = document.getElementById('chat-inline-reasoning-panel');
                            if (panel && panel.parentNode) {
                                panel.parentNode.removeChild(panel);
                            }
                            
                            // Reset streaming flag now that streaming is complete
                            this.isStreaming = false;
                            
                            // Close stream
                            await reader.cancel();
                            
                            resolve({
                                success: true,
                                answer: responseData.answer || resultData.answer,
                                citations: citations,
                                timing: resultData.timing,
                                analysis: resultData.analysis,
                                conversation_id: data.conversation_id
                            });
                        } else if (data.type === 'error') {
                            // Handle errors
                            
                            // Clean up streaming message if exists
                            if (this.streamingMessageId) {
                                const streamingMsg = document.getElementById(this.streamingMessageId);
                                if (streamingMsg && streamingMsg.parentNode) {
                                    streamingMsg.parentNode.removeChild(streamingMsg);
                                }
                                this.streamingMessageId = null;
                            }
                            
                            // Remove panel immediately on error
                            const panel = document.getElementById('chat-inline-reasoning-panel');
                            if (panel && panel.parentNode) {
                                panel.parentNode.removeChild(panel);
                            }
                            
                            // Reset streaming flag on error
                            this.isStreaming = false;
                            
                            // Check for rate limit errors
                            if (data.error === 'RATE_LIMIT_EXCEEDED' || data.error === 'DEMO_LIMIT_EXCEEDED') {
                                await reader.cancel();
                                
                                if (!isAuthenticated) {
                                    reject(new Error(`DEMO_LIMIT: ${data.message}`));
                                } else {
                                    reject(new Error(data.message));
                                }
                                return;
                            } else {
                                await reader.cancel();
                                reject(new Error(data.message || 'An error occurred during processing'));
                                return;
                            }
                        }
                                } catch (parseError) {
                                }
                            }
                        }
                    } catch (streamError) {
                    
                    // Reset streaming flag on error
                    this.isStreaming = false;
                    
                    // Clean up streaming message if exists
                    if (this.streamingMessageId) {
                        const streamingMsg = document.getElementById(this.streamingMessageId);
                        if (streamingMsg && streamingMsg.parentNode) {
                            streamingMsg.parentNode.removeChild(streamingMsg);
                        }
                        this.streamingMessageId = null;
                    }
                    
                    // Remove panel immediately on connection error
                    const panel = document.getElementById('chat-inline-reasoning-panel');
                    if (panel && panel.parentNode) {
                        panel.parentNode.removeChild(panel);
                    }
                    
                    await reader.cancel();
                    reject(new Error('Stream error. Please try again.'));
                    }
                };
                
                // Start processing the stream
                processStream();
                
            } catch (error) {
                
                // Reset streaming flag on setup error
                this.isStreaming = false;
                
                // Remove panel immediately on setup error
                const panel = document.getElementById('chat-inline-reasoning-panel');
                if (panel && panel.parentNode) {
                    panel.parentNode.removeChild(panel);
                }
                
                this.addMessage('assistant', `Error: ${error.message}`, [], 'error');
                reject(error);
            }
        });
    }

    createProgressStepsContainer() {
        // Hide any old typing indicator
        this.hideTypingIndicator();
        
        // Remove any existing inline reasoning panel
        const existingPanel = document.getElementById('chat-inline-reasoning-panel');
        if (existingPanel) {
            existingPanel.remove();
        }
        
        // Create a NEW inline reasoning panel AS A MESSAGE
        const messagesContainer = document.getElementById('chatMessages');
        if (!messagesContainer) return;
        
        const panel = document.createElement('div');
        panel.id = 'chat-inline-reasoning-panel';
        panel.className = 'chat-message reasoning-panel assistant';
        panel.innerHTML = `
            <div class="chat-content">
                <div class="chat-avatar assistant">
                    <i class="fas fa-chart-line"></i>
                </div>
                <div class="message-text-wrapper" style="width: 100%;">
                    <div id="chat-inline-reasoning-box" class="bg-slate-50 p-4 rounded-lg border border-slate-200">
                        <div class="flex items-center justify-between mb-3">
                            <div class="flex items-center space-x-2">
                                <div class="w-1.5 h-1.5 rounded-full bg-blue-500" id="chat-panel-indicator"></div>
                                <p id="chat-panel-status" class="text-sm font-medium text-slate-700">Processing query...</p>
                            </div>
                        </div>
                        
                        <!-- Progress Bar -->
                        <div id="chat-progress-bar-container" class="mb-3">
                            <div class="bg-slate-200 h-1 rounded-full overflow-hidden">
                                <div id="chat-inline-progress-bar" class="bg-blue-500 h-1 rounded-full transition-all duration-300" style="width: 10%"></div>
                            </div>
                        </div>
                        
                        <!-- Reasoning Steps Container -->
                        <div id="chat-inline-reasoning-steps" class="space-y-1 overflow-y-auto">
                        </div>
                    </div>
                    <!-- Streaming Content Container (will be filled when tokens arrive) -->
                    <div id="chat-inline-streaming-container" class="mt-3"></div>
                </div>
            </div>
        `;
        
        // Append to messages container (will appear right after user's message)
        messagesContainer.appendChild(panel);
        
        this.scrollToBottom();
    }

    removeProgressStepsContainer() {
        // DON'T remove the panel - keep it visible like screener
        // Just mark it as complete
        const panel = document.getElementById('chat-inline-reasoning-panel');
        if (panel) {
            this.updateChatPanelStatus('completed', 'Complete');
            const icon = document.getElementById('chat-panel-icon');
            if (icon) {
                icon.className = 'fas fa-check-circle text-green-600';
            }
            
            // Hide progress bar
            const progressBarContainer = document.getElementById('chat-progress-bar-container');
            if (progressBarContainer) {
                progressBarContainer.style.display = 'none';
            }
        }
    }

    updateChatPanelStatus(status, message) {
        const statusEl = document.getElementById('chat-panel-status');
        const indicatorEl = document.getElementById('chat-panel-indicator');
        
        if (statusEl) {
            statusEl.textContent = message;
        }
        
        if (indicatorEl) {
            switch (status) {
                case 'processing':
                    indicatorEl.className = 'w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse';
                    break;
                case 'complete':
                    indicatorEl.className = 'w-1.5 h-1.5 rounded-full bg-green-500';
                    break;
                case 'error':
                    indicatorEl.className = 'w-1.5 h-1.5 rounded-full bg-red-500';
                    break;
            }
        }
    }

    addChatReasoningStep(stepId, message, status = 'active') {
        const container = document.getElementById('chat-inline-reasoning-steps');
        if (!container) return;
        
        // Hide initial progress bar when first step appears
        const progressSection = document.getElementById('chat-progress-bar-container');
        if (progressSection && status !== 'active') {
            progressSection.classList.add('hidden');
        }
        
        // Check if this step already exists
        let stepEl = document.getElementById(`chat-step-${stepId}`);
        
        if (status === 'active') {
            // Remove any existing active steps ONLY
            const existingActive = container.querySelectorAll('.reasoning-step.active');
            existingActive.forEach(el => el.remove());
            
            // If step already exists as completed, don't recreate it
            if (stepEl && (stepEl.classList.contains('completed') || stepEl.classList.contains('failed'))) {
                return;
            }
            
            // Create or update active step
            if (!stepEl) {
                stepEl = document.createElement('div');
                stepEl.id = `chat-step-${stepId}`;
                container.appendChild(stepEl);
            }
            
            stepEl.className = `reasoning-step ${status}`;
            stepEl.innerHTML = `
                <div class="flex items-start space-x-3">
                    <div class="step-icon w-5 h-5 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                        <div class="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></div>
                    </div>
                    <div class="flex-1">
                        <div class="step-message text-sm text-slate-700 font-medium">${message}</div>
                        <div class="step-details text-xs text-slate-500 mt-1"></div>
                    </div>
                </div>
            `;
        } else {
            // For completed/failed, check if step exists and update it
            if (stepEl) {
                this.updateChatReasoningStep(stepId, message, status);
            } else {
                // Create new completed step with dot style
                const stepDiv = document.createElement('div');
                stepDiv.id = `chat-step-${stepId}`;
                stepDiv.className = `reasoning-step ${status}`;
                
                stepDiv.innerHTML = `
                    <div class="flex items-start space-x-2.5 text-xs">
                        <div class="w-1 h-1 rounded-full bg-slate-400 mt-1.5 flex-shrink-0"></div>
                        <div class="flex-1 text-slate-600 leading-relaxed">
                            ${message}
                        </div>
                    </div>
                `;
                
                container.appendChild(stepDiv);
            }
        }
        
        // Scroll to show new step
        container.scrollTop = container.scrollHeight;
    }

    updateChatReasoningStep(stepId, message, status) {
        const stepEl = document.getElementById(`chat-step-${stepId}`);
        if (!stepEl) return;
        
        stepEl.className = `reasoning-step ${status}`;
        
        // Update with simple dot style
        if (status === 'completed') {
            stepEl.innerHTML = `
                <div class="flex items-start space-x-2.5 text-xs">
                    <div class="w-1 h-1 rounded-full bg-slate-400 mt-1.5 flex-shrink-0"></div>
                    <div class="flex-1 text-slate-600 leading-relaxed">
                        ${message}
                    </div>
                </div>
            `;
        } else if (status === 'active') {
            stepEl.innerHTML = `
                <div class="flex items-start space-x-3">
                    <div class="step-icon w-5 h-5 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                        <div class="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></div>
                    </div>
                    <div class="flex-1">
                        <div class="step-message text-sm text-slate-700 font-medium">${message}</div>
                    </div>
                </div>
            `;
        }
    }

    addChatReasoningInfo(stepId, message) {
        const stepEl = document.getElementById(`chat-step-${stepId}`);
        if (!stepEl) return;
        
        const detailsEl = stepEl.querySelector('.step-details');
        if (detailsEl) {
            const info = document.createElement('div');
            info.className = 'text-xs text-blue-600 mt-1 pl-2 border-l-2 border-blue-200';
            info.textContent = message;
            detailsEl.appendChild(info);
        }
    }

    handleProgressEvent(data) {
        // Update progress bar (use inline version)
        const progressBar = document.getElementById('chat-inline-progress-bar');
        if (progressBar) {
            let width = 10;
            if (data.step === 'analysis') width = 30;
            else if (data.step === 'search') width = 60;
            else if (data.step === 'generation') width = 85;
            progressBar.style.width = width + '%';
        }
        
        // Update status bar
        this.updateChatPanelStatus('processing', data.message);
        
        // Only add step if it's not "Generating response..." (that should only show at the end)
        // For agent mode, we want to show iteration steps, not the generic "Generating response" at the top
        if (data.message && data.message.includes('Generating response')) {
            // Don't add this as a step - it will be added at the end when actually generating
            return;
        }
        
        // Add or update step for other progress messages
        this.addChatReasoningStep(data.step, data.message, 'active');
        
        this.scrollToBottom();
    }

    handleAnalysisEvent(data) {
        // Update status
        this.updateChatPanelStatus('processing', 'Searching transcripts...');
        
        // Show analysis with actual reason from question analyzer - KEEP IT VISIBLE
        const container = document.getElementById('chat-inline-reasoning-steps');
        if (container) {
            // Use reasoning from question analyzer as the main message
            const reasoning = data.data?.reasoning || data.message || 'Question analysis complete';
            
            // Build details HTML showing tickers and target quarters
            let detailsHTML = '';
            if (data.data) {
                const details = [];
                if (data.data.tickers && data.data.tickers.length > 0) {
                    details.push(`<div class="text-xs mt-1.5"><span class="font-medium text-slate-600">Companies:</span> <span class="text-blue-600 font-semibold">${data.data.tickers.join(', ')}</span></div>`);
                }
                
                // Show all target quarters
                if (data.data.target_quarters && data.data.target_quarters.length > 0) {
                    const quarterList = data.data.target_quarters
                        .map(q => q.replace('_', ' ').toUpperCase())  // Format: "2025_q2" -> "2025 Q2"
                        .join(', ');
                    details.push(`<div class="text-xs mt-0.5"><span class="font-medium text-slate-600">Target Quarters:</span> <span class="text-slate-600">${quarterList}</span></div>`);
                } else if (data.data.quarter_context) {
                    const periodText = data.data.quarter_context === 'multiple' ? 'Multiple periods' : 
                                     data.data.quarter_context === 'latest' ? 'Latest quarter' :
                                     data.data.quarter_context === 'specific' ? 'Specific quarter' :
                                     data.data.quarter_context;
                    details.push(`<div class="text-xs mt-0.5"><span class="font-medium text-slate-600">Period:</span> <span class="text-slate-600">${periodText}</span></div>`);
                }
                
                if (details.length > 0) {
                    detailsHTML = `<div class="mt-1">${details.join('')}</div>`;
                }
            }
            
            // ADD the analysis step (don't replace - build up the trace)
            const analysisStep = document.createElement('div');
            analysisStep.className = 'reasoning-step completed mb-1';
            analysisStep.innerHTML = `
                <div class="flex items-start space-x-2.5 text-xs">
                    <div class="w-1 h-1 rounded-full bg-slate-400 mt-1.5 flex-shrink-0"></div>
                    <div class="flex-1 text-slate-600 leading-relaxed">
                        ${reasoning}
                        ${detailsHTML}
                    </div>
                </div>
            `;
            container.appendChild(analysisStep);
        }
        
        // Update progress bar
        const progressBar = document.getElementById('chat-inline-progress-bar');
        if (progressBar) {
            progressBar.style.width = '40%';
        }
        
        this.scrollToBottom();
    }

    handleSearchEvent(data) {
        // Don't update status to "Generating response" here - that should only show at the end
        // Just show the search results in the trace
        
        // ADD search results to the trace (don't replace - build up)
        const container = document.getElementById('chat-inline-reasoning-steps');
        if (container) {
            const message = data.message || 'Found relevant transcripts';
            
            // Build transcript details if available
            let transcriptDetailsHTML = '';
            if (data.data && data.data.transcripts) {
                const transcriptLines = [];
                for (const [ticker, quarters] of Object.entries(data.data.transcripts)) {
                    transcriptLines.push(`<div class="text-xs mt-0.5"><span class="font-semibold text-blue-600">${ticker}:</span> <span class="text-slate-600">${quarters.join(', ')}</span></div>`);
                }
                if (transcriptLines.length > 0) {
                    transcriptDetailsHTML = `<div class="mt-1.5">${transcriptLines.join('')}</div>`;
                }
            }
            
            const searchStep = document.createElement('div');
            searchStep.className = 'reasoning-step completed mb-1';
            searchStep.innerHTML = `
                <div class="flex items-start space-x-2.5 text-xs">
                    <div class="w-1 h-1 rounded-full bg-slate-400 mt-1.5 flex-shrink-0"></div>
                    <div class="flex-1 text-slate-600 leading-relaxed">
                        ${message}
                        ${transcriptDetailsHTML}
                    </div>
                </div>
            `;
            container.appendChild(searchStep);
        }
        
        // Update progress bar
        const progressBar = document.getElementById('chat-inline-progress-bar');
        if (progressBar) {
            progressBar.style.width = '70%';
        }
        
        this.scrollToBottom();
    }

    handleIterationStartEvent(data) {
        const iterationData = data.data;
        const container = document.getElementById('chat-inline-reasoning-steps');
        if (!container) return;

        // Store current iteration for context
        this.currentIteration = iterationData.iteration;

        // Don't create iteration sections - the trace will flow naturally
        this.scrollToBottom();
    }

    handleIterationEvalEvent(data) {
        // Skip displaying detailed evaluation reasoning text
        // Only show action traces, not evaluation details
        // The evaluation happens internally but we don't show the reasoning to users
        return;
    }

    handleAgentDecisionEvent(data) {
        // Skip displaying detailed evaluation reasoning
        // Only show action traces, not evaluation details
        // The decision happens internally but we don't show the reasoning to users
        return;
    }

    handleMasterThinkingEvent(data) {
        const thinkingData = data.data;
        const container = document.getElementById('chat-inline-reasoning-steps');
        if (!container) return;

        // Filter out evaluation reasoning text - only show action-oriented thinking
        const message = data.message || '';
        
        // Skip if it's evaluation reasoning (contains phrases like "The answer", "lacks", "gaps", etc.)
        if (message.includes('The answer') && 
            (message.includes('lacks') || message.includes('gaps') || message.includes('missing') || 
             message.includes('does not provide') || message.includes('omits'))) {
            // This is evaluation reasoning, skip it
            return;
        }

        // Show master agent thinking in professional format with dot (only for action-oriented messages)
        const thinkingStep = document.createElement('div');
        thinkingStep.className = 'reasoning-step completed mb-1';
        thinkingStep.innerHTML = `
            <div class="flex items-start space-x-2.5 text-xs">
                <div class="w-1 h-1 rounded-full bg-slate-400 mt-1.5 flex-shrink-0"></div>
                <div class="flex-1 text-slate-600 leading-relaxed">
                    ${message}
                </div>
            </div>
        `;
        container.appendChild(thinkingStep);
        this.scrollToBottom();
    }

    handleIterationFollowupEvent(data) {
        const iterationData = data.data;
        const container = document.getElementById('chat-inline-reasoning-steps');
        if (!container) return;

        // Backend now includes all search questions in conversational format
        // Format: "Searching: [question]" for each question, one per line
        const searchMessage = data.message || 'Searching...';
        
        // Split by newlines and format each line with a bullet point
        const lines = searchMessage.split('\n').filter(line => line.trim());
        const formattedLines = lines.map(line => {
            // Each line is "Searching: [question]" - style it nicely
            return `<div class="mb-1">${line}</div>`;
        }).join('');

        // Present follow-up as a professional search action
        const followupStep = document.createElement('div');
        followupStep.className = 'reasoning-step completed mb-2';
        followupStep.innerHTML = `
            <div class="flex items-start space-x-2.5 text-xs">
                <div class="w-1 h-1 rounded-full bg-slate-400 mt-1.5 flex-shrink-0"></div>
                <div class="flex-1 text-slate-600 leading-relaxed">
                    ${formattedLines}
                </div>
            </div>
        `;
        container.appendChild(followupStep);
        this.scrollToBottom();
    }

    handleIterationSearchEvent(data) {
        const iterationData = data.data;
        const container = document.getElementById('chat-inline-reasoning-steps');
        if (!container) return;

        // Create professional language for search results
        const newChunks = iterationData.new_chunks_count;
        const sources = iterationData.sources && iterationData.sources.length > 0 
            ? ` from ${iterationData.sources.slice(0, 2).join(', ')}${iterationData.sources.length > 2 ? ' and others' : ''}` 
            : '';
        
        let searchText = '';
        if (newChunks > 0) {
            searchText = `Found ${newChunks} additional source${newChunks > 1 ? 's' : ''}${sources}`;
        } else {
            searchText = `Compiling response from available sources`;
        }

        const searchStep = document.createElement('div');
        searchStep.className = 'reasoning-step completed mb-1';
        searchStep.innerHTML = `
            <div class="flex items-start space-x-2.5 text-xs">
                <div class="w-1 h-1 rounded-full bg-slate-400 mt-1.5 flex-shrink-0"></div>
                <div class="flex-1 text-slate-600 leading-relaxed">
                    ${searchText}
                </div>
            </div>
        `;
        container.appendChild(searchStep);

        this.scrollToBottom();
    }

    handleIterationNewsSearchEvent(data) {
        const iterationData = data.data;
        const container = document.getElementById('chat-inline-reasoning-steps');
        if (!container) return;

        const query = iterationData.query || 'latest news';
        
        // Create a separate line for each web search with a web icon
        const newsStep = document.createElement('div');
        newsStep.className = 'reasoning-step completed mb-1';
        newsStep.innerHTML = `
            <div class="flex items-start space-x-2.5 text-xs">
                <div class="flex items-center justify-center w-4 h-4 mt-0.5 flex-shrink-0">
                    <i class="fas fa-globe text-blue-500 text-xs"></i>
                </div>
                <div class="flex-1 text-slate-600 leading-relaxed">
                    Searching web: "${query}"
                </div>
            </div>
        `;
        container.appendChild(newsStep);

        this.scrollToBottom();
    }

    handleIterationCompleteEvent(data) {
        const iterationData = data.data;
        const container = document.getElementById('chat-inline-reasoning-steps');
        if (!container) return;

        // Filter out evaluation reasoning - only show action-oriented completion messages
        const reason = iterationData.reason || '';
        
        // Skip if it's evaluation reasoning (contains phrases like "The answer", "lacks", "gaps", etc.)
        if (reason.includes('The answer') && 
            (reason.includes('lacks') || reason.includes('gaps') || reason.includes('missing') || 
             reason.includes('does not provide') || reason.includes('omits'))) {
            // This is evaluation reasoning, skip it
            return;
        }

        // Show completion with professional format and dot (only for action-oriented messages)
        const completeStep = document.createElement('div');
        completeStep.className = 'reasoning-step completed mb-1';
        completeStep.innerHTML = `
            <div class="flex items-start space-x-2.5 text-xs">
                <div class="w-1 h-1 rounded-full bg-slate-400 mt-1.5 flex-shrink-0"></div>
                <div class="flex-1 text-slate-600 leading-relaxed">
                    ${reason}
                </div>
            </div>
        `;
        container.appendChild(completeStep);

        this.scrollToBottom();
    }

    handleIterationFinalEvent(data) {
        const iterationData = data.data;
        const container = document.getElementById('chat-inline-reasoning-steps');
        if (!container) return;

        // Update status to show we're generating the final response
        this.updateChatPanelStatus('processing', 'Generating response...');

        // Add "Generating response..." as the final step using the shared reasoning step helper
        // This ensures it will automatically switch to a completed style when streaming finishes
        const message = data.message || 'Preparing comprehensive response';
        this.addChatReasoningStep('generation', message, 'active');

        this.scrollToBottom();
    }

    createStreamingMessage(messageId) {
        // Create an empty assistant message that tokens will be appended to
        // Prefer rendering inside the inline reasoning panel if available
        const inlinePanel = document.getElementById('chat-inline-reasoning-panel');

        // Reset markdown rendering counters for new message
        this.lastMarkdownRenderTime = 0;
        this.tokensSinceLastRender = 0;

        if (inlinePanel) {
            // Place streaming bubble inside the same panel box for unified styling
            const streamingHost = document.getElementById('chat-inline-streaming-container')
                || inlinePanel.querySelector('.message-text-wrapper');

            if (!streamingHost) return;

            const messageDiv = document.createElement('div');
            messageDiv.id = messageId;
            // Keep minimal wrapper since avatar and wrapper already exist in panel
            messageDiv.className = '';

            const bubble = document.createElement('div');
            // Use same typography as streaming markdown elsewhere but keep it consistent with panel
            bubble.className = 'chat-bubble';
            bubble.id = `${messageId}-bubble`;
            bubble.innerHTML = '<span class="typing-cursor">|</span>';

            messageDiv.appendChild(bubble);
            streamingHost.appendChild(messageDiv);

            // Scroll into view
            this.scrollToBottom(true);
            requestAnimationFrame(() => this.scrollToBottom());
            return;
        }

        // Fallback: create a separate assistant message if no inline panel exists
        const messagesContainer = document.getElementById('chatMessages');
        if (!messagesContainer) return;

        const messageDiv = document.createElement('div');
        messageDiv.id = messageId;
        messageDiv.className = 'chat-message assistant';

        const messageContent = document.createElement('div');
        messageContent.className = 'chat-content';

        const avatar = document.createElement('div');
        avatar.className = 'chat-avatar assistant';
        avatar.innerHTML = '<i class="fas fa-chart-line"></i>';

        const textWrapper = document.createElement('div');
        textWrapper.className = 'message-text-wrapper';

        const bubble = document.createElement('div');
        bubble.className = 'chat-bubble';
        bubble.id = `${messageId}-bubble`;
        bubble.innerHTML = '<span class="typing-cursor">|</span>';

        textWrapper.appendChild(bubble);
        messageContent.appendChild(avatar);
        messageContent.appendChild(textWrapper);
        messageDiv.appendChild(messageContent);
        messagesContainer.appendChild(messageDiv);

        this.scrollToBottom(true);
        requestAnimationFrame(() => this.scrollToBottom());
    }

    appendTokenToMessage(messageId, token) {
        // Append token to the streaming message - render markdown in real-time
        const bubble = document.getElementById(`${messageId}-bubble`);
        if (!bubble) {
            return;
        }
        
        // Update stored raw text
        const currentText = (bubble.getAttribute('data-raw-text') || '') + token;
        bubble.setAttribute('data-raw-text', currentText);
        
        // Get or create content container and cursor
        let contentDiv = bubble.querySelector('.streaming-content');
        let cursor = bubble.querySelector('.typing-cursor');
        
        if (!contentDiv) {
            // First token - set up the structure
            bubble.textContent = ''; // Clear initial cursor
            
            contentDiv = document.createElement('div');
            contentDiv.className = 'streaming-content prose prose-sm max-w-none';
            contentDiv.style.display = 'inline';
            bubble.appendChild(contentDiv);
            
            // Add cursor after content
            cursor = document.createElement('span');
            cursor.className = 'typing-cursor';
            cursor.textContent = '|';
            bubble.appendChild(cursor);
        }
        
        // Throttle markdown rendering - only re-render every 100ms or every 20 tokens
        const now = Date.now();
        if (!this.lastMarkdownRenderTime) {
            this.lastMarkdownRenderTime = 0;
            this.tokensSinceLastRender = 0;
        }
        
        this.tokensSinceLastRender++;
        const shouldRender = (now - this.lastMarkdownRenderTime > 100) || (this.tokensSinceLastRender >= 20);
        
        if (shouldRender) {
            // Render markdown
            try {
                contentDiv.innerHTML = marked.parse(currentText);
                this.lastMarkdownRenderTime = now;
                this.tokensSinceLastRender = 0;
                
                // Force scroll after markdown render since content height changed
                requestAnimationFrame(() => {
                    this.scrollToBottom();
                });
            } catch (e) {
                // Fallback to plain text if markdown parsing fails
                contentDiv.textContent = currentText;
            }
        }
        
        
        // Also do regular throttled autoscroll between markdown renders
        if (!this.lastScrollTime || now - this.lastScrollTime > 100) {
            this.lastScrollTime = now;
            requestAnimationFrame(() => {
                this.scrollToBottom();
            });
        }
    }

    finalizeStreamingMessage(messageId, citations, reasoningHTML) {
        // Finalize the streaming message with citations and reasoning trace
        
        const messageDiv = document.getElementById(messageId);
        if (!messageDiv) {
            return;
        }
        
        const bubble = document.getElementById(`${messageId}-bubble`);
        if (bubble) {
            // Remove typing cursor
            const cursor = bubble.querySelector('.typing-cursor');
            if (cursor) {
                cursor.remove();
            }
            
            // Get the raw text and ensure final markdown rendering
            const finalText = bubble.getAttribute('data-raw-text') || bubble.textContent || bubble.innerText;
            
            // Check if we have streaming content div (markdown streaming) or just text
            const contentDiv = bubble.querySelector('.streaming-content');
            
            if (contentDiv) {
                // Already rendered as markdown during streaming, just do final render to ensure completeness
                try {
                    contentDiv.innerHTML = marked.parse(finalText);
                } catch (e) {
                    contentDiv.textContent = finalText;
                }
            } else {
                // No streaming content, render markdown directly to bubble
                try {
                    bubble.innerHTML = marked.parse(finalText);
                } catch (e) {
                    bubble.textContent = finalText;
                }
            }
        }
        
        // Resolve the correct message host: either inside the streaming message or, for inline panel path,
        // the panel's message-text-wrapper
        let textWrapper = messageDiv.querySelector('.message-text-wrapper');
        
        if (!textWrapper) {
            const inlinePanel = document.getElementById('chat-inline-reasoning-panel');
            if (inlinePanel) {
                textWrapper = inlinePanel.querySelector('.message-text-wrapper');
            }
        }
        
        // If still no textWrapper found, create one for the streaming message
        if (!textWrapper) {
            // Create a proper text wrapper for the streaming message
            textWrapper = document.createElement('div');
            textWrapper.className = 'message-text-wrapper';
            
            // Move the bubble content into the text wrapper
            const bubble = document.getElementById(`${messageId}-bubble`);
            if (bubble && bubble.parentNode) {
                // Move the bubble from its current parent to the text wrapper
                bubble.parentNode.removeChild(bubble);
                textWrapper.appendChild(bubble);
                
                // Add the text wrapper to the message div
                messageDiv.appendChild(textWrapper);
            }
        }
        
        
        // Add reasoning trace at the top if provided
        if (reasoningHTML) {
            const reasoningContainer = document.createElement('div');
            reasoningContainer.className = 'reasoning-trace-container collapsed mb-4';
            reasoningContainer.innerHTML = `
                <div class="reasoning-trace-header bg-slate-50 p-3 rounded-lg border border-slate-200 cursor-pointer" onclick="this.parentElement.classList.toggle('collapsed')">
                    <div class="flex items-center justify-between">
                        <div class="flex items-center space-x-2">
                            <div class="w-1.5 h-1.5 rounded-full bg-slate-400"></div>
                            <span class="text-sm font-medium text-slate-700">Reasoning Trace</span>
                            <span class="text-xs text-slate-500">(Click to expand)</span>
                        </div>
                        <i class="fas fa-chevron-down text-slate-500 toggle-icon"></i>
                    </div>
                </div>
                <div class="reasoning-trace-content pl-4 border-l-2 border-slate-200 ml-4 mt-2 space-y-1" style="display: none;">
                    ${reasoningHTML}
                </div>
            `;
            
            // For inline panel case, insert after the reasoning box, not before everything
            const inlinePanel = document.getElementById('chat-inline-reasoning-panel');
            if (inlinePanel && textWrapper === inlinePanel.querySelector('.message-text-wrapper')) {
                // Insert reasoning trace after the reasoning box but before streaming container
                const reasoningBox = document.getElementById('chat-inline-reasoning-box');
                const streamingContainer = document.getElementById('chat-inline-streaming-container');
                if (reasoningBox && streamingContainer) {
                    textWrapper.insertBefore(reasoningContainer, streamingContainer);
                } else {
                    textWrapper.appendChild(reasoningContainer);
                }
            } else {
                textWrapper.insertBefore(reasoningContainer, textWrapper.firstChild);
            }
        }
        
        // Add citations at the end
        if (citations && citations.length > 0) {
            // Separate transcript, 10-K, and news citations
            const transcriptCitations = citations.filter(c => !c.type || c.type === 'transcript');
            const tenKCitations = citations.filter(c => c.type === '10-K');
            const newsCitations = citations.filter(c => c.type === 'news');
            
            const totalCount = citations.length;
            
            const citationsDiv = document.createElement('div');
            citationsDiv.className = 'citations collapsed';
            
            const citationsHeader = document.createElement('div');
            citationsHeader.className = 'citations-header';
            citationsHeader.innerHTML = `
                <i class="fas fa-link"></i> 
                <span class="sources-count">${totalCount} source${totalCount > 1 ? 's' : ''}</span>
                <button class="toggle-sources-btn" onclick="toggleSources(this)">
                    <i class="fas fa-chevron-down"></i>
                </button>
            `;
            citationsDiv.appendChild(citationsHeader);
            
            const citationsContent = document.createElement('div');
            citationsContent.className = 'citations-content';
            citationsContent.style.display = 'none';
            
            // Add transcript citations section
            if (transcriptCitations.length > 0) {
                const transcriptSection = document.createElement('div');
                transcriptSection.className = 'citations-section';
                transcriptSection.innerHTML = `<div class="citations-section-header">📄 Document Sources (${transcriptCitations.length})</div>`;
                
                transcriptCitations.forEach((citation, index) => {
                    const citationDiv = document.createElement('div');
                    citationDiv.className = 'citation';
                    
                    const company = citation.company || citation.ticker || 'Unknown Company';
                    const quarter = citation.quarter || 'Unknown Quarter';
                    const chunkText = citation.chunk_text || 'No preview available';
                    const formattedQuarter = quarter.toString().replace('_', ' ');
                    const transcriptAvailable = citation.transcript_available === true;
                    const forceShowTranscript = true;
                    
                    citationDiv.innerHTML = `
                        <div class="citation-header">
                            <div class="citation-header-left">
                                <span class="citation-company">${this.escapeHtml(company)}</span>
                                <span class="citation-quarter">${this.escapeHtml(formattedQuarter)}</span>
                                ${transcriptAvailable || forceShowTranscript ? '<span class="transcript-badge">📄 Transcript</span>' : ''}
                            </div>
                            ${transcriptAvailable || forceShowTranscript ? `<button class="view-transcript-btn" onclick="viewTranscript('${company}', '${this.escapeHtml(quarter)}', ${index})">
                                <i class="fas fa-file-alt mr-1"></i>View Full Transcript
                            </button>` : ''}
                        </div>
                        <div class="citation-preview">${this.escapeHtml(chunkText.substring(0, 150))}${chunkText.length > 150 ? '...' : ''}</div>
                        <div class="citation-actions">
                            <button class="citation-expand-btn" onclick="toggleCitation(${index}, this)">
                                <i class="fas fa-expand-alt"></i> Show More
                            </button>
                        </div>
                    `;
                    
                    transcriptSection.appendChild(citationDiv);
                });
                
                citationsContent.appendChild(transcriptSection);
            }

            // Add 10-K citations section
            if (tenKCitations.length > 0) {
                const tenKSection = document.createElement('div');
                tenKSection.className = 'citations-section';
                tenKSection.innerHTML = `<div class="citations-section-header">📑 10-K SEC Filings (${tenKCitations.length})</div>`;

                tenKCitations.forEach((citation, index) => {
                    const citationDiv = document.createElement('div');
                    citationDiv.className = 'citation tenk-citation';

                    const ticker = citation.ticker || 'Unknown';
                    const fiscalYear = citation.fiscal_year || 'Unknown';
                    const section = citation.section || 'Unknown Section';
                    const marker = citation.marker || `[10K${index + 1}]`;
                    const chunkType = citation.chunk_type || 'text';
                    const path = citation.path || '';

                    citationDiv.innerHTML = `
                        <div class="citation-header">
                            <div class="citation-header-left">
                                <span class="citation-marker">${this.escapeHtml(marker)}</span>
                                <span class="citation-company">${this.escapeHtml(ticker)}</span>
                                <span class="citation-fiscal-year">FY${this.escapeHtml(fiscalYear)}</span>
                                ${chunkType === 'table' ? '<span class="table-badge">📊 Table</span>' : ''}
                            </div>
                        </div>
                        <div class="citation-section-info">
                            <span class="text-xs text-slate-600">${this.escapeHtml(section)}</span>
                        </div>
                        ${path ? `<div class="citation-path">
                            <span class="text-xs text-slate-500">${this.escapeHtml(path)}</span>
                        </div>` : ''}
                    `;

                    tenKSection.appendChild(citationDiv);
                });

                citationsContent.appendChild(tenKSection);
            }

            // Add news citations section
            if (newsCitations.length > 0) {
                const newsSection = document.createElement('div');
                newsSection.className = 'citations-section';
                newsSection.innerHTML = `<div class="citations-section-header">🌐 Web Sources (${newsCitations.length})</div>`;
                
                newsCitations.forEach((citation, index) => {
                    const citationDiv = document.createElement('div');
                    citationDiv.className = 'citation news-citation';
                    
                    const title = citation.title || 'No title';
                    const url = citation.url || '#';
                    const marker = citation.marker || `[N${index + 1}]`;
                    const publishedDate = citation.published_date || '';
                    
                    citationDiv.innerHTML = `
                        <div class="citation-header">
                            <div class="citation-header-left">
                                <span class="citation-marker">${this.escapeHtml(marker)}</span>
                                <span class="citation-title">${this.escapeHtml(title)}</span>
                                ${publishedDate ? `<span class="citation-date">${this.escapeHtml(publishedDate)}</span>` : ''}
                            </div>
                            <a href="${this.escapeHtml(url)}" target="_blank" rel="noopener noreferrer" class="view-article-btn">
                                <i class="fas fa-external-link-alt mr-1"></i>View Article
                            </a>
                        </div>
                        <div class="citation-url">
                            <a href="${this.escapeHtml(url)}" target="_blank" rel="noopener noreferrer" class="text-blue-600 hover:underline text-xs">
                                ${this.escapeHtml(url)}
                            </a>
                        </div>
                    `;
                    
                    newsSection.appendChild(citationDiv);
                });
                
                citationsContent.appendChild(newsSection);
            }
            
            citationsDiv.appendChild(citationsContent);
            textWrapper.appendChild(citationsDiv);
            
            // Store citations globally
            window.lastCitations = citations;
        }
        
        this.scrollToBottom();
    }

    moveStreamingMessageToMainChat(messageId) {
        // Move the streaming message from the inline panel to the main chat container
        const messageDiv = document.getElementById(messageId);
        if (!messageDiv) return;

        const messagesContainer = document.getElementById('chatMessages');
        if (!messagesContainer) return;

        // Check if the message is currently in the inline panel
        const inlinePanel = document.getElementById('chat-inline-reasoning-panel');
        if (inlinePanel && inlinePanel.contains(messageDiv)) {
            // Remove from inline panel
            messageDiv.remove();
            
            // Ensure the message has proper structure for main chat
            if (!messageDiv.classList.contains('chat-message')) {
                messageDiv.classList.add('chat-message', 'assistant');
            }
            
            // Add proper chat content wrapper if missing
            if (!messageDiv.querySelector('.chat-content')) {
                const chatContent = document.createElement('div');
                chatContent.className = 'chat-content';
                
                // Move all children to the chat content wrapper
                while (messageDiv.firstChild) {
                    chatContent.appendChild(messageDiv.firstChild);
                }
                
                // Add avatar
                const avatar = document.createElement('div');
                avatar.className = 'chat-avatar assistant';
                avatar.innerHTML = '<i class="fas fa-chart-line"></i>';
                chatContent.insertBefore(avatar, chatContent.firstChild);
                
                messageDiv.appendChild(chatContent);
            }
            
            // Ensure the message has a proper text wrapper for citations and reasoning
            const chatContent = messageDiv.querySelector('.chat-content');
            if (chatContent && !chatContent.querySelector('.message-text-wrapper')) {
                const textWrapper = document.createElement('div');
                textWrapper.className = 'message-text-wrapper';
                
                // Move the bubble content into the text wrapper
                const bubble = messageDiv.querySelector('.chat-bubble');
                if (bubble) {
                    bubble.parentNode.removeChild(bubble);
                    textWrapper.appendChild(bubble);
                    chatContent.appendChild(textWrapper);
                }
            }
            
            // Add to main chat container
            messagesContainer.appendChild(messageDiv);
            
            // Scroll to show the new message
            this.scrollToBottom();
        }
    }

    processCitations(citations, chunks) {
        // If citations are already in the right format, return them
        // Check if ANY citation has the proper structure (not just the first one)
        if (citations && citations.length > 0) {
            const hasProperStructure = citations.some(c =>
                c.company || c.type || c.title || c.url
            );
            if (hasProperStructure) {
                return citations;
            }
        }
        
        // Otherwise, process chunks to create citations
        const processedCitations = [];

        for (let i = 0; i < chunks.length; i++) {
            const chunk = chunks[i];
            processedCitations.push({
                company: chunk.ticker || 'Unknown',
                quarter: chunk.year && chunk.quarter ? `${chunk.year}_Q${chunk.quarter}` : 'Unknown',
                chunk_id: chunk.citation || i.toString(),
                chunk_text: chunk.chunk_text || '',
                relevance_score: chunk.similarity || 0,
                source_file: chunk.source_file || null,
                transcript_available: false
            });
        }
        
        return processedCitations;
    }
    
    async handleSSEStream(response) {
        return new Promise((resolve, reject) => {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            
            // Create streaming message element
            let streamingMessage = document.getElementById('streaming-chat-message');
            if (!streamingMessage) {
                streamingMessage = document.createElement('div');
                streamingMessage.id = 'streaming-chat-message';
                streamingMessage.className = 'chat-message assistant';
                
                const messageContent = document.createElement('div');
                messageContent.className = 'chat-content';
                
                const messageText = document.createElement('div');
                messageText.className = 'chat-bubble';
                messageText.innerHTML = '';
                
                messageContent.appendChild(messageText);
                streamingMessage.appendChild(messageContent);
                
                const messagesContainer = document.getElementById('chatMessages');
                if (messagesContainer) {
                    messagesContainer.appendChild(streamingMessage);
                    if (typeof scrollToBottom === 'function') {
                        scrollToBottom();
                    }
                }
            }
            
            // Create progress indicator
            let progressIndicator = document.getElementById('chat-progress-indicator');
            if (!progressIndicator) {
                progressIndicator = document.createElement('div');
                progressIndicator.id = 'chat-progress-indicator';
                progressIndicator.className = 'chat-progress-indicator';
                
                const progressBar = document.createElement('div');
                progressBar.className = 'progress-bar';
                
                const progressFill = document.createElement('div');
                progressFill.className = 'progress-fill';
                
                const progressText = document.createElement('div');
                progressText.className = 'progress-text';
                
                progressBar.appendChild(progressFill);
                progressIndicator.appendChild(progressBar);
                progressIndicator.appendChild(progressText);
                
                const messagesContainer = document.getElementById('chatMessages');
                if (messagesContainer) {
                    messagesContainer.appendChild(progressIndicator);
                    if (typeof scrollToBottom === 'function') {
                        scrollToBottom();
                    }
                }
            }
            
            function readStream() {
                reader.read().then(({ done, value }) => {
                    if (done) {
                        // Hide progress indicator
                        if (progressIndicator && progressIndicator.parentNode) {
                            progressIndicator.parentNode.removeChild(progressIndicator);
                        }
                        resolve({ success: true, answer: 'Stream completed' });
                        return;
                    }
                    
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop(); // Keep incomplete line in buffer
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                
                                if (data.type === 'progress') {
                                    // Update progress indicator
                                    const progressFill = progressIndicator.querySelector('.progress-fill');
                                    const progressText = progressIndicator.querySelector('.progress-text');
                                    
                                    if (progressFill) {
                                        const percentage = Math.round(data.progress * 100);
                                        progressFill.style.width = `${percentage}%`;
                                    }
                                    
                                    if (progressText) {
                                        progressText.textContent = data.message || 'Processing...';
                                    }
                                    
                                    if (typeof scrollToBottom === 'function') {
                                        scrollToBottom();
                                    }
                                } else if (data.type === 'stream') {
                                    // Handle streaming content
                                    const messageText = streamingMessage.querySelector('.chat-bubble');
                                    if (messageText && data.content) {
                                        const currentContent = messageText.textContent || '';
                                        const newContent = currentContent + data.content;
                                        
                                        if (typeof marked !== 'undefined') {
                                            try {
                                                marked.setOptions({
                                                    breaks: true,
                                                    gfm: true,
                                                    headerIds: false,
                                                    mangle: false
                                                });
                                                messageText.innerHTML = marked.parse(newContent);
                                            } catch (error) {
                                                messageText.innerHTML = newContent.replace(/</g, '&lt;').replace(/>/g, '&gt;');
                                            }
                                        } else {
                                            messageText.innerHTML = newContent.replace(/</g, '&lt;').replace(/>/g, '&gt;');
                                        }
                                        
                                        // Use throttled scrolling during streaming (scrollToBottom uses isStreaming flag)
                                        if (typeof scrollToBottom === 'function') {
                                            scrollToBottom();
                                        }
                                    } else {
                                    }
                                } else if (data.type === 'result') {
                                    // Final result received
                                    if (progressIndicator && progressIndicator.parentNode) {
                                        progressIndicator.parentNode.removeChild(progressIndicator);
                                    }
                                    
                                    // Remove streaming message and add final message
                                    if (streamingMessage && streamingMessage.parentNode) {
                                        streamingMessage.parentNode.removeChild(streamingMessage);
                                    }
                                    
                                    resolve(data);
                                    return;
                                } else if (data.type === 'error') {
                                    // Error received
                                    if (progressIndicator && progressIndicator.parentNode) {
                                        progressIndicator.parentNode.removeChild(progressIndicator);
                                    }
                                    
                                    if (streamingMessage && streamingMessage.parentNode) {
                                        streamingMessage.parentNode.removeChild(streamingMessage);
                                    }
                                    
                                    reject(new Error(data.message || 'Streaming error'));
                                    return;
                                }
                            } catch (e) {
                            }
                        }
                    }
                    
                    readStream();
                }).catch(reject);
            }
            
            readStream();
        });
    }
    
    async handleJSONResponse(response) {
        try {
            const data = await response.json();
            
            if (data.success) {
                // Add the complete response to chat
                this.addMessage('assistant', data.answer, data.citations || []);
                
                // Update conversation ID if provided
                if (data.conversation_id) {
                    window.currentConversationId = data.conversation_id;
                }
                
                return data;
            } else {
                throw new Error(data.error || 'Unknown error occurred');
            }
        } catch (error) {
            throw error;
        }
    }
    
    async sendToLegacyBackend(message) {
        // Check if user is authenticated
        const token = localStorage.getItem('authToken');
        const isAuthenticated = token && localStorage.getItem('currentUser');
        
        // Use appropriate endpoint based on authentication status
        const endpoint = isAuthenticated ? '/chat/message' : '/chat/landing/demo';
        const headers = {
            'Content-Type': 'application/json'
        };
        
        // Only add Authorization header for authenticated requests
        if (isAuthenticated) {
            headers['Authorization'] = `Bearer ${token}`;
        }
        
        
        // Build request body with conversation_id for thread continuity
        const requestBody = {
            message: message,
            comprehensive: true,
            conversation_id: window.currentConversationId || null
        };
        
        // Add session_id for demo users
        if (!isAuthenticated) {
            requestBody.session_id = window.SessionManager ? 
                window.SessionManager.getOrCreateSessionId() : 
                `session_${Date.now()}`;
        }
        
        
        const response = await fetch(`${CONFIG.apiBaseUrl}${endpoint}`, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            const errorMessage = errorData.detail || response.statusText;
            throw new Error(`HTTP ${response.status}: ${errorMessage}`);
        }

        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || 'Unknown error occurred');
        }
        
        // Update conversation ID from response if provided
        if (data.conversation_id) {
            window.currentConversationId = data.conversation_id;
        }

        return data;
    }

    addMessage(sender, content, citations = [], type = 'normal', reasoningHTML = '') {
        const messagesContainer = document.getElementById('chatMessages');
        if (!messagesContainer) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${sender}`;

        const messageContent = document.createElement('div');
        messageContent.className = 'chat-content';

        // Add avatar for all messages in Claude style
        const avatar = document.createElement('div');
        avatar.className = `chat-avatar ${sender}`;
        if (sender === 'user') {
            avatar.innerHTML = '<i class="fas fa-user"></i>';
        } else {
            avatar.innerHTML = '<i class="fas fa-chart-line"></i>';
        }

        // Create a wrapper for the text content and citations
        const textWrapper = document.createElement('div');
        textWrapper.className = 'message-text-wrapper';
        
        // Add reasoning trace at the TOP if provided (for assistant messages)
        if (sender === 'assistant' && reasoningHTML) {
            const reasoningContainer = document.createElement('div');
            reasoningContainer.className = 'reasoning-trace-container collapsed mb-4';
            reasoningContainer.innerHTML = `
                <div class="reasoning-trace-header bg-slate-50 p-3 rounded-lg border border-slate-200 cursor-pointer" onclick="this.parentElement.classList.toggle('collapsed')">
                    <div class="flex items-center justify-between">
                        <div class="flex items-center space-x-2">
                            <div class="w-1.5 h-1.5 rounded-full bg-slate-400"></div>
                            <span class="text-sm font-medium text-slate-700">Reasoning Trace</span>
                            <span class="text-xs text-slate-500">(Click to expand)</span>
                        </div>
                        <i class="fas fa-chevron-down text-slate-500 toggle-icon"></i>
                    </div>
                </div>
                <div class="reasoning-trace-content pl-4 border-l-2 border-slate-200 ml-4 mt-2 space-y-1" style="display: none;">
                    ${reasoningHTML}
                </div>
            `;
            textWrapper.appendChild(reasoningContainer);
        }
        
        const messageText = document.createElement('div');
        messageText.className = 'chat-bubble';
        
        // Render markdown for assistant messages, escape HTML for user messages
        if (sender === 'assistant' && typeof marked !== 'undefined') {
            try {
                // Configure marked.js for better rendering
                marked.setOptions({
                    breaks: true,
                    gfm: true,
                    headerIds: false,
                    mangle: false
                });
                messageText.innerHTML = marked.parse(content);
            } catch (error) {
                messageText.innerHTML = this.escapeHtml(content);
            }
        } else {
            messageText.innerHTML = this.escapeHtml(content);
        }
        
        textWrapper.appendChild(messageText);

        // Add citations below the message text (not in the flex layout)
        if (citations && citations.length > 0) {
            // Separate transcript, 10-K, and news citations
            const transcriptCitations = citations.filter(c => !c.type || c.type === 'transcript');
            const tenKCitations = citations.filter(c => c.type === '10-K');
            const newsCitations = citations.filter(c => c.type === 'news');
            
            const totalCount = citations.length;
            
            const citationsDiv = document.createElement('div');
            citationsDiv.className = 'citations collapsed';
            
            const citationsHeader = document.createElement('div');
            citationsHeader.className = 'citations-header';
            citationsHeader.innerHTML = `
                <i class="fas fa-link"></i> 
                <span class="sources-count">${totalCount} source${totalCount > 1 ? 's' : ''}</span>
                <button class="toggle-sources-btn" onclick="toggleSources(this)">
                    <i class="fas fa-chevron-down"></i>
                </button>
            `;
            citationsDiv.appendChild(citationsHeader);
            
            const citationsContent = document.createElement('div');
            citationsContent.className = 'citations-content';
            citationsContent.style.display = 'none'; // Hide sources by default
            
            // Add transcript citations section
            if (transcriptCitations.length > 0) {
                const transcriptSection = document.createElement('div');
                transcriptSection.className = 'citations-section';
                transcriptSection.innerHTML = `<div class="citations-section-header">📄 Document Sources (${transcriptCitations.length})</div>`;
                
                transcriptCitations.forEach((citation, index) => {
                    const citationDiv = document.createElement('div');
                    citationDiv.className = 'citation';
                    
                    // Extract proper company and quarter info
                    const company = citation.company || citation.ticker || 'Unknown Company';
                    const quarter = citation.quarter || 'Unknown Quarter';
                    const chunkText = citation.chunk_text || 'No preview available';
                    
                    // Format quarter properly (e.g., "2025_Q2" -> "2025 Q2")
                    const formattedQuarter = quarter.toString().replace('_', ' ');
                    
                    // Check if transcript is available
                    const transcriptAvailable = citation.transcript_available === true;
                    
                    // FORCE show transcript button for testing - remove this after confirming database has data
                    const forceShowTranscript = true;
                    
                    citationDiv.innerHTML = `
                        <div class="citation-header">
                            <div class="citation-header-left">
                                <span class="citation-company">${this.escapeHtml(company)}</span>
                                <span class="citation-quarter">${this.escapeHtml(formattedQuarter)}</span>
                                ${transcriptAvailable || forceShowTranscript ? '<span class="transcript-badge">📄 Transcript</span>' : ''}
                            </div>
                            ${transcriptAvailable || forceShowTranscript ? `<button class="view-transcript-btn" onclick="viewTranscript('${company}', '${this.escapeHtml(quarter)}', ${index})">
                                <i class="fas fa-file-alt mr-1"></i>View Full Transcript
                            </button>` : ''}
                        </div>
                        <div class="citation-preview">${this.escapeHtml(chunkText.substring(0, 150))}${chunkText.length > 150 ? '...' : ''}</div>
                        <div class="citation-actions">
                            <button class="citation-expand-btn" onclick="toggleCitation(${index}, this)">
                                <i class="fas fa-expand-alt"></i> Show More
                            </button>
                        </div>
                    `;
                    
                    transcriptSection.appendChild(citationDiv);
                });
                
                citationsContent.appendChild(transcriptSection);
            }

            // Add 10-K citations section
            if (tenKCitations.length > 0) {
                const tenKSection = document.createElement('div');
                tenKSection.className = 'citations-section';
                tenKSection.innerHTML = `<div class="citations-section-header">📑 10-K SEC Filings (${tenKCitations.length})</div>`;

                tenKCitations.forEach((citation, index) => {
                    const citationDiv = document.createElement('div');
                    citationDiv.className = 'citation tenk-citation';

                    const ticker = citation.ticker || 'Unknown';
                    const fiscalYear = citation.fiscal_year || 'Unknown';
                    const section = citation.section || 'Unknown Section';
                    const marker = citation.marker || `[10K${index + 1}]`;
                    const chunkType = citation.chunk_type || 'text';
                    const path = citation.path || '';

                    citationDiv.innerHTML = `
                        <div class="citation-header">
                            <div class="citation-header-left">
                                <span class="citation-marker">${this.escapeHtml(marker)}</span>
                                <span class="citation-company">${this.escapeHtml(ticker)}</span>
                                <span class="citation-fiscal-year">FY${this.escapeHtml(fiscalYear)}</span>
                                ${chunkType === 'table' ? '<span class="table-badge">📊 Table</span>' : ''}
                            </div>
                        </div>
                        <div class="citation-section-info">
                            <span class="text-xs text-slate-600">${this.escapeHtml(section)}</span>
                        </div>
                        ${path ? `<div class="citation-path">
                            <span class="text-xs text-slate-500">${this.escapeHtml(path)}</span>
                        </div>` : ''}
                    `;

                    tenKSection.appendChild(citationDiv);
                });

                citationsContent.appendChild(tenKSection);
            }

            // Add news citations section
            if (newsCitations.length > 0) {
                const newsSection = document.createElement('div');
                newsSection.className = 'citations-section';
                newsSection.innerHTML = `<div class="citations-section-header">🌐 Web Sources (${newsCitations.length})</div>`;
                
                newsCitations.forEach((citation, index) => {
                    const citationDiv = document.createElement('div');
                    citationDiv.className = 'citation news-citation';
                    
                    const title = citation.title || 'No title';
                    const url = citation.url || '#';
                    const marker = citation.marker || `[N${index + 1}]`;
                    const publishedDate = citation.published_date || '';
                    
                    citationDiv.innerHTML = `
                        <div class="citation-header">
                            <div class="citation-header-left">
                                <span class="citation-marker">${this.escapeHtml(marker)}</span>
                                <span class="citation-title">${this.escapeHtml(title)}</span>
                                ${publishedDate ? `<span class="citation-date">${this.escapeHtml(publishedDate)}</span>` : ''}
                            </div>
                            <a href="${this.escapeHtml(url)}" target="_blank" rel="noopener noreferrer" class="view-article-btn">
                                <i class="fas fa-external-link-alt mr-1"></i>View Article
                            </a>
                        </div>
                        <div class="citation-url">
                            <a href="${this.escapeHtml(url)}" target="_blank" rel="noopener noreferrer" class="text-blue-600 hover:underline text-xs">
                                ${this.escapeHtml(url)}
                            </a>
                        </div>
                    `;
                    
                    newsSection.appendChild(citationDiv);
                });
                
                citationsContent.appendChild(newsSection);
            }
            
            citationsDiv.appendChild(citationsContent);
            textWrapper.appendChild(citationsDiv);
            
            // Store citations globally for expand functionality
            window.lastCitations = citations;
        }

        // Claude-style structure: avatar first, then text wrapper (which contains text + citations)
        messageContent.appendChild(avatar);
        messageContent.appendChild(textWrapper);
        
        messageDiv.appendChild(messageContent);
        messagesContainer.appendChild(messageDiv);

        // Scroll to bottom after message is added - use multiple strategies for reliability
        // Immediate scroll
        this.scrollToBottom(true);
        
        // Smooth scroll after render
        requestAnimationFrame(() => {
            this.scrollToBottom();
            // Additional scroll after a short delay for any lazy-loaded content
            setTimeout(() => {
                this.scrollToBottom();
            }, 100);
        });

        // Store message
        this.messages.push({ sender, content, citations, type, timestamp: new Date() });
    }

    showTypingIndicator() {
        const messagesContainer = document.getElementById('chatMessages');
        if (!messagesContainer) return;

        // Simple processing message
        const processingMessage = "Processing";

        const typingDiv = document.createElement('div');
        typingDiv.className = 'chat-message assistant typing-indicator';
        typingDiv.id = 'typing-indicator';

        typingDiv.innerHTML = `
            <div class="chat-content">
                <div class="chat-avatar assistant">
                    <i class="fas fa-chart-line"></i>
                </div>
                <div class="message-text-wrapper">
                    <div class="chat-bubble">
                        <div class="chat-typing-indicator">
                            <div class="typing-spinner">
                                <div class="spinner-ring"></div>
                            </div>
                            <span class="typing-text">${processingMessage}</span>
                        </div>
                    </div>
                </div>
            </div>
        `;

        messagesContainer.appendChild(typingDiv);

        // Scroll to bottom using chat section scrollbar
        setTimeout(() => {
            const chatSection = document.getElementById('chatSection');
            if (chatSection) {
                chatSection.scrollTo({
                    top: chatSection.scrollHeight,
                    behavior: 'smooth'
                });
            }
        }, 100);
    }

    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    showCancellationInProgress() {
        // Update the typing indicator to show cancellation
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            const typingContent = typingIndicator.querySelector('.chat-typing-indicator span');
            if (typingContent) {
                typingContent.textContent = 'Cancelling request...';
            }
        }
        
        // Also disable the stop button to prevent multiple clicks
        const sendButton = document.getElementById('sendChatButton');
        if (sendButton) {
            sendButton.disabled = true;
            sendButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
            sendButton.title = 'Cancelling...';
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    async loadChatHistory() {
        // Initialize with a fresh conversation thread
        
        // Clear any existing conversation ID
        window.currentConversationId = null;
        
        // Add welcome message for new conversation
        this.addWelcomeMessage();
    }
    
    addWelcomeMessage() {
        // Check if welcome message already exists
        const existingWelcome = document.querySelector('#chatMessages .chat-message.assistant');
        if (existingWelcome) {
            return; // Already has welcome message
        }
        
        // Add welcome message
        const welcomeHtml = `
            <div class="chat-message assistant">
                <div class="chat-content">
                    <div class="chat-avatar assistant">
                        <i class="fas fa-chart-line"></i>
                    </div>
                    <div class="message-text-wrapper">
                        <div class="chat-bubble">
                            <div class="welcome-content">
                                <h2>Welcome to StrataLens</h2>
                                <p>Your AI analyst for public equity markets.</p>
                                
                                <h3 class="quick-suggestions-heading">Quick Suggestions</h3>
                                <div class="quick-examples">
                                    <button class="example-btn" onclick="document.getElementById('chatInput').value = 'Compile $INTC management commentary on foundry business in last 3 quarters'; document.getElementById('chatInput').focus();">
                                        $INTC foundry business commentary
                                    </button>
                                    <button class="example-btn" onclick="document.getElementById('chatInput').value = 'Compare $MSFT and $GOOGL cloud segment'; document.getElementById('chatInput').focus();">
                                        Compare $MSFT and $GOOGL cloud segment
                                    </button>
                                    <button class="example-btn" onclick="document.getElementById('chatInput').value = 'Compile $META AI capex commentary in last 3 quarters'; document.getElementById('chatInput').focus();">
                                        $META AI capex commentary
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
                    const messagesContainer = document.getElementById('chatMessages');
        if (messagesContainer) {
            messagesContainer.insertAdjacentHTML('afterbegin', welcomeHtml);
        }
    }
    
    // REMOVED restoreConversation - using global conversation thread functions
    
    formatDate(dateString) {
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', { 
            year: 'numeric', 
            month: 'short', 
            day: 'numeric' 
        });
    }

    scrollToBottom(instant = false) {
        // Only autoscroll during streaming to show the streaming effect
        if (!this.isStreaming) {
            return;
        }
        
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) {
            return;
        }
        
        // Check if user has scrolled up manually (more tolerant threshold for markdown rendering)
        // Increased from 100px to 200px to account for content height changes during markdown rendering
        const distanceFromBottom = chatMessages.scrollHeight - chatMessages.scrollTop - chatMessages.clientHeight;
        const isNearBottom = distanceFromBottom < 200;
        
        // Only check occasionally to avoid spam (every 500ms)
        const now = Date.now();
        if (!this.lastScrollLogTime || now - this.lastScrollLogTime > 500) {
            this.lastScrollLogTime = now;
        }
        
        // Only auto-scroll if user hasn't scrolled away
        if (isNearBottom) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }

    async clearChat() {
        if (confirm('Are you sure you want to clear the chat history?')) {
            try {
                const response = await fetch(`${CONFIG.apiBaseUrl}/chat/clear`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${localStorage.getItem('authToken')}`
                    },
                    body: JSON.stringify({ confirm: true })
                });

                if (response.ok) {
                    // Clear UI
                    const messagesContainer = document.getElementById('chatMessages');
                    if (!messagesContainer) {
                        return;
                    }
                    messagesContainer.innerHTML = `
                        <div class="chat-message assistant">
                            <div class="chat-content">
                                <div class="chat-bubble">
                                    <p>StrataLens Financial Intelligence is ready to assist with:</p>
                                    <ul>
                                        <li>• Comprehensive financial analysis and benchmarking</li>
                                        <li>• Market trends and sector performance insights</li>
                                        <li>• Earnings analysis and forecast modeling</li>
                                        <li>• Investment research and due diligence support</li>
                                    </ul>
                                    <p>Please submit your financial research inquiry.</p>
                                </div>
                            </div>
                        </div>
                    `;
                    this.messages = [];
                }
            } catch (error) {
                if (typeof showToast === 'function') {
                    showToast('Failed to clear chat history', 'error');
                }
            }
        }
    }

    exportChat() {
        if (this.messages.length === 0) {
            if (typeof showToast === 'function') {
                showToast('No messages to export', 'warning');
            }
            return;
        }

        const chatData = this.messages.map(msg => ({
            sender: msg.sender,
            content: msg.content,
            timestamp: msg.timestamp
        }));

        const dataStr = JSON.stringify(chatData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = `chat-export-${new Date().toISOString().split('T')[0]}.json`;
        link.click();
        
        URL.revokeObjectURL(url);
    }

    // Ticker Autocomplete Methods
    handleTickerAutocomplete() {
        const chatInput = document.getElementById('chatInput');
        if (!chatInput) return;

        const text = chatInput.value;
        const cursorPosition = chatInput.selectionStart;
        
        // Find the current word being typed (looking for $ prefix)
        const beforeCursor = text.substring(0, cursorPosition);
        
        // Look for $ followed by letters/numbers (ticker pattern)
        const tickerMatch = beforeCursor.match(/\$([A-Za-z0-9]*)$/);
        
        if (tickerMatch) {
            const tickerPrefix = tickerMatch[1].toUpperCase();
            if (tickerPrefix.length >= 1) {
                this.fetchTickerSuggestions(tickerPrefix);
            } else {
                this.hideTickerSuggestions();
            }
        } else {
            this.hideTickerSuggestions();
        }
    }

    async fetchTickerSuggestions(tickerPrefix) {
        try {
            const params = new URLSearchParams({
                query: tickerPrefix,
                limit: 8
            });

            const config = window.STRATALENS_CONFIG || { apiBaseUrl: 'http://localhost:8000' };
            const response = await fetch(`${config.apiBaseUrl}/companies/public/search?${params.toString()}`);

            const result = await response.json();

            if (response.ok && result.companies) {
                this.showTickerSuggestions(result.companies);
            } else {
                this.hideTickerSuggestions();
            }

        } catch (error) {
            this.hideTickerSuggestions();
        }
    }

    showTickerSuggestions(companies) {
        // Remove existing suggestions
        this.hideTickerSuggestions();

        if (!companies || companies.length === 0) return;

        const chatInput = document.getElementById('chatInput');
        if (!chatInput) return;

        // Create suggestions dropdown
        const suggestionsDiv = document.createElement('div');
        suggestionsDiv.id = 'tickerSuggestions';
        suggestionsDiv.className = 'absolute z-50 bg-white border border-gray-200 rounded-lg shadow-lg max-h-60 overflow-y-auto';
        
        // Position the dropdown below the input
        const inputRect = chatInput.getBoundingClientRect();
        suggestionsDiv.style.position = 'fixed';
        suggestionsDiv.style.top = `${inputRect.bottom + 5}px`;
        suggestionsDiv.style.left = `${inputRect.left}px`;
        suggestionsDiv.style.width = `${Math.max(inputRect.width, 300)}px`;

        // Add suggestion items
        companies.forEach((company, index) => {
            const suggestionItem = document.createElement('div');
            suggestionItem.className = 'p-3 hover:bg-gray-50 cursor-pointer border-b border-gray-100 last:border-b-0';
            suggestionItem.innerHTML = `
                <div class="flex items-center justify-between">
                    <div class="flex items-center gap-3">
                        <div class="flex flex-col">
                            <div class="flex items-center gap-2">
                                <span class="font-semibold text-gray-800">$${company.symbol}</span>
                                <span class="text-sm text-gray-500">${company.exchangeShortName || ''}</span>
                            </div>
                            <span class="text-sm text-gray-600 line-clamp-1">${company.companyName}</span>
                        </div>
                    </div>
                    <div class="flex flex-col items-end text-xs text-gray-400">
                        ${company.marketCap ? `<span>${this.formatMarketCap(company.marketCap)}</span>` : ''}
                        ${company.sector ? `<span>${company.sector}</span>` : ''}
                    </div>
                </div>
            `;
            
            suggestionItem.addEventListener('click', () => {
                this.selectTickerSuggestion(company.symbol);
            });

            suggestionsDiv.appendChild(suggestionItem);
        });

        document.body.appendChild(suggestionsDiv);

        // Add click outside to close
        setTimeout(() => {
            document.addEventListener('click', this.handleOutsideClick.bind(this));
        }, 100);
    }

    hideTickerSuggestions() {
        const suggestionsDiv = document.getElementById('tickerSuggestions');
        if (suggestionsDiv) {
            suggestionsDiv.remove();
        }
    }

    selectTickerSuggestion(ticker) {
        const chatInput = document.getElementById('chatInput');
        if (!chatInput) return;

        const text = chatInput.value;
        const cursorPosition = chatInput.selectionStart;
        
        // Find the current ticker being typed
        const beforeCursor = text.substring(0, cursorPosition);
        const tickerMatch = beforeCursor.match(/\$([A-Za-z0-9]*)$/);
        
        if (tickerMatch) {
            const startPos = cursorPosition - tickerMatch[0].length;
            const newText = text.substring(0, startPos) + `$${ticker} ` + text.substring(cursorPosition);
            
            chatInput.value = newText;
            
            // Set cursor position after the inserted ticker
            const newCursorPos = startPos + ticker.length + 2; // +2 for $ and space
            chatInput.setSelectionRange(newCursorPos, newCursorPos);
            
            // Trigger input event to resize textarea
            chatInput.dispatchEvent(new Event('input'));
        }

        this.hideTickerSuggestions();
    }

    handleOutsideClick(event) {
        const suggestionsDiv = document.getElementById('tickerSuggestions');
        const chatInput = document.getElementById('chatInput');
        
        if (suggestionsDiv && !suggestionsDiv.contains(event.target) && 
            chatInput && !chatInput.contains(event.target)) {
            this.hideTickerSuggestions();
            document.removeEventListener('click', this.handleOutsideClick.bind(this));
        }
    }

    formatMarketCap(marketCap) {
        if (!marketCap) return '';
        
        if (marketCap >= 1e12) {
            return `$${(marketCap / 1e12).toFixed(1)}T`;
        } else if (marketCap >= 1e9) {
            return `$${(marketCap / 1e9).toFixed(1)}B`;
        } else if (marketCap >= 1e6) {
            return `$${(marketCap / 1e6).toFixed(1)}M`;
        } else {
            return `$${marketCap.toFixed(0)}`;
        }
    }

    handleTickerAutocompleteKeydown(e) {
        const suggestionsDiv = document.getElementById('tickerSuggestions');
        if (!suggestionsDiv || suggestionsDiv.style.display === 'none') {
            return false; // No suggestions visible, don't handle
        }

        const suggestions = suggestionsDiv.querySelectorAll('.p-3');
        let selectedIndex = -1;

        // Find currently selected suggestion
        suggestions.forEach((suggestion, index) => {
            if (suggestion.classList.contains('bg-blue-50')) {
                selectedIndex = index;
            }
        });

        switch (e.key) {
            case 'ArrowDown':
                e.preventDefault();
                selectedIndex = Math.min(selectedIndex + 1, suggestions.length - 1);
                this.highlightSuggestion(suggestions, selectedIndex);
                return true;

            case 'ArrowUp':
                e.preventDefault();
                selectedIndex = Math.max(selectedIndex - 1, 0);
                this.highlightSuggestion(suggestions, selectedIndex);
                return true;

            case 'Enter':
                e.preventDefault();
                if (selectedIndex >= 0) {
                    const selectedSuggestion = suggestions[selectedIndex];
                    const ticker = selectedSuggestion.querySelector('.font-semibold').textContent.replace('$', '');
                    this.selectTickerSuggestion(ticker);
                }
                return true;

            case 'Escape':
                e.preventDefault();
                this.hideTickerSuggestions();
                return true;

            default:
                return false;
        }
    }

    highlightSuggestion(suggestions, index) {
        suggestions.forEach((suggestion, i) => {
            if (i === index) {
                suggestion.classList.add('bg-blue-50');
                suggestion.scrollIntoView({ block: 'nearest' });
            } else {
                suggestion.classList.remove('bg-blue-50');
            }
        });
    }
}

// Chat History Modal Management (for history modal accessed from chat header)
class ChatHistoryManager {
    constructor() {
        this.currentPage = 1;
        this.filters = {};
    }

    setupEventListeners() {
        // Modal-based history search controls - set up when modal is opened
        const searchBtn = document.getElementById('chatHistorySearchBtn');
        const clearBtn = document.getElementById('chatHistoryClearBtn');
        const prevBtn = document.getElementById('chatHistoryPrevBtn');
        const nextBtn = document.getElementById('chatHistoryNextBtn');

        if (searchBtn) {
            searchBtn.onclick = () => this.searchHistory();
        }

        if (clearBtn) {
            clearBtn.onclick = () => this.clearFilters();
        }

        if (prevBtn) {
            prevBtn.onclick = () => this.loadPage(this.currentPage - 1);
        }

        if (nextBtn) {
            nextBtn.onclick = () => this.loadPage(this.currentPage + 1);
        }

    }

    async loadPage(page = 1) {
        const loading = document.getElementById('chatHistoryLoading');
        const list = document.getElementById('chatHistoryList');

        if (loading) loading.classList.remove('hidden');
        if (list) list.innerHTML = '';

        try {
            const params = new URLSearchParams({
                limit: '20',
                offset: String((page - 1) * 20)
            });

            // Add filters if present
            if (this.filters.search) {
                params.append('search', this.filters.search);
            }
            if (this.filters.date_from) {
                params.append('date_from', this.filters.date_from);
            }
            if (this.filters.date_to) {
                params.append('date_to', this.filters.date_to);
            }

            const response = await fetch(`${CONFIG.apiBaseUrl}/chat/history?${params}`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('authToken')}`
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.success) {
                this.renderHistory(data.messages);
                this.updatePagination(data.total_count, page);
                this.currentPage = page;
            } else {
                throw new Error(data.error || 'Failed to load chat history');
            }

        } catch (error) {
            if (list) {
                list.innerHTML = `
                    <div class="text-center py-8">
                        <i class="fas fa-exclamation-triangle text-red-500 text-2xl mb-2"></i>
                        <p class="text-text-secondary">Failed to load chat history</p>
                        <button class="btn btn-primary mt-2" onclick="chatHistoryManager.loadPage(${page})">
                            <i class="fas fa-redo mr-2"></i>Try Again
                        </button>
                    </div>
                `;
            }
        } finally {
            if (loading) loading.classList.add('hidden');
        }
    }

    renderHistory(messages) {
        const list = document.getElementById('chatHistoryList');
        if (!list) return;

        if (messages.length === 0) {
            list.innerHTML = `
                <div class="text-center py-8">
                    <i class="fas fa-comments text-text-tertiary text-4xl mb-4"></i>
                    <h3 class="text-lg font-semibold text-text-primary mb-2">No Messages Found</h3>
                    <p class="text-text-secondary">Start a conversation to see your chat history here.</p>
                </div>
            `;
            return;
        }

        list.innerHTML = '';

        messages.forEach(message => {
            const item = document.createElement('div');
            item.className = 'chat-history-item';
            
            const citationsCount = message.citations ? message.citations.length : 0;
            const formattedDate = new Date(message.created_at).toLocaleString();

            item.innerHTML = `
                <div class="chat-history-header">
                    <div class="flex-1">
                        <div class="chat-history-question">${this.escapeHtml(message.user_message)}</div>
                    </div>
                    <div class="chat-history-meta">
                        <div class="chat-history-date">${formattedDate}</div>
                        <div class="chat-history-id">${message.id.substring(0, 8)}</div>
                    </div>
                </div>
                <div class="chat-history-answer">${this.escapeHtml(message.assistant_response.substring(0, 200))}${message.assistant_response.length > 200 ? '...' : ''}</div>
                ${citationsCount > 0 ? `
                    <div class="chat-history-citations">
                        <span class="chat-history-citations-count">
                            <i class="fas fa-link mr-1"></i>${citationsCount} citation${citationsCount > 1 ? 's' : ''}
                        </span>
                    </div>
                ` : ''}
                <div class="chat-history-actions">
                    <button class="chat-history-action-btn" onclick="chatHistoryManager.viewMessage('${message.id}')">
                        <i class="fas fa-eye mr-1"></i>View Full
                    </button>
                    <button class="chat-history-action-btn" onclick="chatHistoryManager.reuseMessage('${this.escapeHtml(message.user_message)}')">
                        <i class="fas fa-redo mr-1"></i>Ask Again
                    </button>
                </div>
            `;

            list.appendChild(item);
        });
    }

    async searchHistory() {
        const searchInput = document.getElementById('chatHistorySearch');
        const dateFromInput = document.getElementById('chatHistoryDateFrom');
        const dateToInput = document.getElementById('chatHistoryDateTo');

        this.filters = {};

        if (searchInput && searchInput.value.trim()) {
            this.filters.search = searchInput.value.trim();
        }

        if (dateFromInput && dateFromInput.value) {
            this.filters.date_from = dateFromInput.value;
        }

        if (dateToInput && dateToInput.value) {
            this.filters.date_to = dateToInput.value;
        }

        await this.loadPage(1);
    }

    clearFilters() {
        const searchInput = document.getElementById('chatHistorySearch');
        const dateFromInput = document.getElementById('chatHistoryDateFrom');
        const dateToInput = document.getElementById('chatHistoryDateTo');

        if (searchInput) searchInput.value = '';
        if (dateFromInput) dateFromInput.value = '';
        if (dateToInput) dateToInput.value = '';

        this.filters = {};
        this.loadPage(1);
    }

    async viewMessage(chatId) {
        try {
            const response = await fetch(`${CONFIG.apiBaseUrl}/chat/history/${chatId}`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('authToken')}`
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.success) {
                this.showMessageModal(data.message);
            } else {
                throw new Error(data.error || 'Failed to load message');
            }

        } catch (error) {
            if (typeof showToast === 'function') {
                showToast(`Failed to load message: ${error.message}`, 'error');
            }
        }
    }

    reuseMessage(message) {
        // Close history modal and switch to chat
        const historyModal = document.getElementById('chatHistoryModal');
        if (historyModal) {
            historyModal.classList.add('hidden');
            historyModal.classList.remove('flex');
        }

        // Fill chat input
        setTimeout(() => {
            const chatInput = document.getElementById('chatInput');
            if (chatInput) {
                chatInput.value = message;
                chatInput.focus();
                chatInput.style.height = 'auto';
                chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
            }
        }, 100);
    }

    updatePagination(totalCount, currentPage) {
        const pagination = document.getElementById('chatHistoryPagination');
        const paginationInfo = document.getElementById('chatHistoryPaginationInfo');
        const prevBtn = document.getElementById('chatHistoryPrevBtn');
        const nextBtn = document.getElementById('chatHistoryNextBtn');

        if (!pagination) return;

        const totalPages = Math.ceil(totalCount / 20);
        const startItem = (currentPage - 1) * 20 + 1;
        const endItem = Math.min(currentPage * 20, totalCount);

        if (paginationInfo) {
            paginationInfo.textContent = `Showing ${startItem}-${endItem} of ${totalCount} messages`;
        }

        if (prevBtn) {
            prevBtn.disabled = currentPage <= 1;
        }

        if (nextBtn) {
            nextBtn.disabled = currentPage >= totalPages;
        }

        if (totalPages > 1) {
            pagination.classList.remove('hidden');
        } else {
            pagination.classList.add('hidden');
        }
    }

    showMessageModal(message) {
        // Create a modal to display the full chat message
        const modal = document.createElement('div');
        modal.className = 'fixed inset-0 bg-black/60 z-[70] flex items-center justify-center p-4';
        modal.id = 'chatMessageModal';

        const citationsCount = message.citations ? message.citations.length : 0;
        const formattedDate = new Date(message.created_at).toLocaleString();

        modal.innerHTML = `
            <div class="card max-w-4xl w-full max-h-[90vh] bg-bg-secondary dark:bg-bg-primary">
                <div class="px-6 py-4 border-b border-border-primary">
                    <div class="flex items-center justify-between">
                        <h3 class="text-lg font-semibold flex items-center gap-2">
                            <i class="fas fa-comment text-accent-primary"></i>
                            Chat Message
                        </h3>
                        <button onclick="this.closest('#chatMessageModal').remove()" class="w-8 h-8 flex items-center justify-center text-text-tertiary hover:text-text-primary rounded-full hover:bg-slate-100 dark:hover:bg-slate-700/50">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                </div>
                <div class="p-6 overflow-y-auto max-h-[70vh]">
                    <div class="space-y-4">
                        <div>
                            <div class="flex justify-between items-start mb-2">
                                <h4 class="font-semibold text-text-primary">Your Question</h4>
                                <span class="text-sm text-text-tertiary">${formattedDate}</span>
                            </div>
                            <div class="bg-bg-tertiary p-3 rounded-lg">
                                <p class="text-text-primary">${this.escapeHtml(message.user_message)}</p>
                            </div>
                        </div>
                        
                        <div>
                            <h4 class="font-semibold text-text-primary mb-2">AI Response</h4>
                            <div class="bg-bg-primary p-4 rounded-lg border border-border-primary">
                                <div class="prose prose-sm max-w-none">${typeof marked !== 'undefined' ? marked.parse(message.assistant_response) : this.escapeHtml(message.assistant_response)}</div>
                            </div>
                        </div>
                        
                        ${citationsCount > 0 ? `
                            <div>
                                <h4 class="font-semibold text-text-primary mb-2">
                                    <i class="fas fa-link mr-1"></i>Citations (${citationsCount})
                                </h4>
                                <div class="space-y-2">
                                    ${message.citations.map((citation, index) => `
                                        <div class="bg-bg-tertiary p-3 rounded-lg border border-border-primary">
                                            <div class="flex items-center gap-2 mb-2">
                                                <span class="citation-company">${this.escapeHtml(citation.company)}</span>
                                                <span class="citation-quarter">${this.escapeHtml(citation.quarter)}</span>
                                                ${citation.transcript_available ? '<span class="transcript-available-badge">📄 Transcript Available</span>' : ''}
                                            </div>
                                            <p class="text-sm text-text-secondary">${this.escapeHtml(citation.chunk_text.substring(0, 200))}${citation.chunk_text.length > 200 ? '...' : ''}</p>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        ` : ''}
                    </div>
                </div>
                <div class="px-6 py-4 border-t border-border-primary bg-bg-tertiary">
                    <div class="flex justify-end gap-2">
                        <button onclick="chatHistoryManager.reuseMessage('${this.escapeHtml(message.user_message)}')" class="btn btn-secondary">
                            <i class="fas fa-redo mr-2"></i>Ask Again
                        </button>
                        <button onclick="this.closest('#chatMessageModal').remove()" class="btn btn-primary">
                            Close
                        </button>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Quick message insertion function
function insertQuickMessage(message) {
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.value = message;
        chatInput.focus();
        chatInput.style.height = 'auto';
        chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
    }
}

// Toggle citation expansion
function toggleCitation(index, button) {
    const citationDiv = button.closest('.citation');
    const preview = citationDiv.querySelector('.citation-preview');
    const isExpanded = citationDiv.classList.contains('expanded');
    
    if (isExpanded) {
        // Collapse
        citationDiv.classList.remove('expanded');
        button.innerHTML = '<i class="fas fa-expand-alt"></i> Show More';
        
        // Truncate text
        const citations = window.lastCitations || [];
        if (citations[index] && citations[index].chunk_text) {
            const chunkText = citations[index].chunk_text;
            preview.textContent = chunkText.substring(0, 150) + (chunkText.length > 150 ? '...' : '');
        }
    } else {
        // Expand
        citationDiv.classList.add('expanded');
        button.innerHTML = '<i class="fas fa-compress-alt"></i> Show Less';
        
        // Show full text
        const citations = window.lastCitations || [];
        if (citations[index] && citations[index].chunk_text) {
            preview.textContent = citations[index].chunk_text;
        }
    }
}

// Get relevant chunks for a specific company and quarter
function getRelevantChunksForCompany(company, quarter) {
    if (!window.lastCitations || window.lastCitations.length === 0) {
        return [];
    }
    
    // Convert quarter format for comparison
    const quarterForComparison = quarter.replace(' ', '_').toLowerCase();
    
    return window.lastCitations
        .filter(citation => citation.company === company && citation.quarter === quarterForComparison)
        .map(citation => ({
            chunk_text: citation.chunk_text,
            chunk_id: citation.chunk_id || '',
            relevance_score: citation.relevance_score || 0
        }));
}

// Simple function to view transcript - calls the direct API endpoint
function viewTranscript(company, quarter, citationIndex) {
    
    // Check if company is valid
    if (!company || company === 'Unknown') {
        if (typeof showToast === 'function') {
            showToast(`Cannot view transcript: Company is "${company}". Please try a different source.`, 'warning');
        }
        return;
    }
    
    try {
        // Parse quarter to extract year and quarter number
        // Try multiple formats
        let quarterMatch = quarter.match(/(\d{4})_Q(\d)/);
        if (!quarterMatch) {
            // Try format like "2025 Q1" or "2025_Q1"
            quarterMatch = quarter.match(/(\d{4})[_\s]Q(\d)/);
        }
        if (!quarterMatch) {
            // Try format like "2025_q1" (lowercase)
            quarterMatch = quarter.match(/(\d{4})_q(\d)/);
        }
        if (!quarterMatch) {
            // If we only have a quarter number (like "4"), assume current year
            if (/^\d+$/.test(quarter)) {
                const currentYear = new Date().getFullYear();
                const quarterNum = parseInt(quarter);
                quarterMatch = [quarter, currentYear.toString(), quarterNum.toString()];
            } else {
                if (typeof showToast === 'function') {
                    showToast(`Invalid quarter format: "${quarter}". Expected format like "2025_Q1" or just "1" for current year.`, 'warning');
                }
                return;
            }
        }
        
        const year = parseInt(quarterMatch[1]);
        const quarterNum = parseInt(quarterMatch[2]);
        
        // Show loading message
        
        // Determine if user is authenticated or in demo mode
        const authToken = localStorage.getItem('authToken');
        const isDemoMode = !authToken || window.location.pathname.includes('landing');
        
        // Use appropriate endpoint based on authentication status
        const endpoint = isDemoMode 
            ? `${CONFIG.apiBaseUrl}/demo/transcript/${company}/${year}/${quarterNum}`
            : `${CONFIG.apiBaseUrl}/transcript/${company}/${year}/${quarterNum}`;
        
        // Prepare headers
        const headers = {};
        if (!isDemoMode && authToken) {
            headers['Authorization'] = `Bearer ${authToken}`;
        }
        
        // Fetch transcript from API
        fetch(endpoint, { headers })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.success && data.transcript_text) {
                // Get relevant chunks for highlighting
                const relevantChunks = getRelevantChunksForCompany(company, quarter);
                
                // Open transcript in a modal with highlighting
                showTranscriptModal(data, company, quarter, relevantChunks);
            } else {
                throw new Error('Transcript data not available');
            }
        })
        .catch(error => {
            
            // Check if it's a 404 error with helpful message
            if (error.message.includes('404')) {
                if (error.message.includes('Chunks are available for search')) {
                    if (typeof showToast === 'function') {
                        showToast(`Transcript Viewing Unavailable: The complete transcript for ${company} ${quarter} is not available for viewing, but you can still ask questions about this company's earnings.`, 'warning');
                    }
                } else {
                    if (typeof showToast === 'function') {
                        showToast(`Transcript Not Found: No transcript data is available for ${company} ${quarter}.`, 'warning');
                    }
                }
            } else {
                if (typeof showToast === 'function') {
                    showToast(`Failed to load ${company} transcript: ${error.message}`, 'error');
                }
            }
        });
        
    } catch (error) {
        if (typeof showToast === 'function') {
            showToast('Failed to view transcript', 'error');
        }
    }
}

// View complete transcript - simple API call
async function viewCompleteTranscript(company, quarter, citationIndex) {
    
    // Prevent multiple simultaneous calls - check immediately
    const callId = `${company}-${quarter}-${Date.now()}`;
    if (window.transcriptLoading) {
        return;
    }
    
    window.transcriptLoading = callId;
    
    try {
        // Get the citation data from multiple sources
        let citations = window.lastCitations || [];
        let citation = citations[citationIndex];
        
        
        // If not found in lastCitations, try to find it in the current conversation
        if (!citation) {
            
            // Try to get citations from the current conversation
            const currentConversation = window.currentConversationId;
            if (currentConversation) {
                // Look for citations in the current conversation messages
                const messages = document.querySelectorAll('.chat-bubble');
                for (let message of messages) {
                    const citationsDiv = message.querySelector('.citations');
                    if (citationsDiv) {
                        const citationItems = citationsDiv.querySelectorAll('.citation-item');
                        for (let item of citationItems) {
                            const citationCompany = item.querySelector('.citation-company')?.textContent;
                            const citationQuarter = item.querySelector('.citation-quarter')?.textContent;
                            
                            if (citationCompany === company && citationQuarter === quarter) {
                                // Found the citation, create a mock citation object
                                citation = {
                                    company: citationCompany,
                                    quarter: citationQuarter,
                                    transcript_available: true,
                                    chunk_text: item.querySelector('.citation-preview')?.textContent || ''
                                };
                                break;
                            }
                        }
                        if (citation) break;
                    }
                }
            }
        }
        
        if (!citation) {
            // Create a minimal citation object if we can't find the original
            citation = {
                company: company,
                quarter: quarter,
                transcript_available: true,
                chunk_text: 'Citation details not available'
            };
        }
        
        // Check if transcript is available
        if (!citation.transcript_available || citation.transcript_available !== true) {
            if (typeof showToast === 'function') {
                showToast('Transcript is not available for this citation. Please try a different source.', 'warning');
            }
            window.transcriptLoading = false;
            return;
        }
        
        
        // Parse quarter format (e.g., "2025_Q2" -> year: 2025, quarter: 2)
        const quarterStr = citation.quarter.toString();
        
        const quarterParts = quarterStr.replace('_', ' ').split(' ');
        
        if (quarterParts.length < 2) {
            throw new Error(`Invalid quarter format: ${quarterStr}`);
        }
        
        const year = parseInt(quarterParts[0]);
        const quarterNum = parseInt(quarterParts[1].replace('Q', ''));
        
        
        if (isNaN(year) || isNaN(quarterNum)) {
            throw new Error(`Could not parse quarter: ${quarterStr}`);
        }
        
        // Show loading state
        const button = event?.target?.closest('.transcript-view-btn');
        
        if (button) {
            const originalText = button.innerHTML;
            button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
            button.disabled = true;
            
            try {
                
                // Determine base URL
                const baseUrl = window.location.protocol === 'file:' ? 'http://localhost:8000' : '';
                
                // Get all relevant chunks from all citations for this company/quarter
                // Convert quarter format from "2025 Q1" to "2025_q1" for comparison
                const quarterForComparison = quarter.replace(' ', '_').toLowerCase();
                
                // Try to get relevant chunks from multiple sources
                let relevantChunks = [];
                
                // First, try from window.lastCitations
                if (window.lastCitations && window.lastCitations.length > 0) {
                    relevantChunks = window.lastCitations
                        .filter(c => c.company === company && c.quarter === quarterForComparison)
                        .map(c => ({
                            chunk_text: c.chunk_text,
                            chunk_id: c.chunk_id || '',
                            relevance_score: c.relevance_score || 0
                        }));
                }
                
                // If no chunks found, try to extract from DOM
                if (relevantChunks.length === 0) {
                    const messages = document.querySelectorAll('.chat-bubble');
                    for (let message of messages) {
                        const citationsDiv = message.querySelector('.citations');
                        if (citationsDiv) {
                            const citationItems = citationsDiv.querySelectorAll('.citation-item');
                            for (let item of citationItems) {
                                const citationCompany = item.querySelector('.citation-company')?.textContent;
                                const citationQuarter = item.querySelector('.citation-quarter')?.textContent;
                                
                                if (citationCompany === company && citationQuarter === quarter) {
                                    const chunkText = item.querySelector('.citation-preview')?.textContent || '';
                                    if (chunkText) {
                                        relevantChunks.push({
                                            chunk_text: chunkText,
                                            chunk_id: `dom-${relevantChunks.length}`,
                                            relevance_score: 0.8
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
                
                // If still no chunks, create a basic one from the citation
                if (relevantChunks.length === 0 && citation.chunk_text) {
                    relevantChunks = [{
                        chunk_text: citation.chunk_text,
                        chunk_id: 'fallback-0',
                        relevance_score: 0.5
                    }];
                }
                
                // Final fallback - create a minimal chunk if we still have nothing
                if (relevantChunks.length === 0) {
                    relevantChunks = [{
                        chunk_text: `Transcript content for ${company} ${quarter}`,
                        chunk_id: 'minimal-0',
                        relevance_score: 0.3
                    }];
                }
                
                
                
                // Use the highlighted transcript endpoint
                const requestBody = {
                    ticker: company,
                    year: year,
                    quarter: quarterNum,
                    relevant_chunks: relevantChunks
                };
                
                
                const response = await fetch(`${baseUrl}/transcript/with-highlights`, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('authToken')}`,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(requestBody)
                });
                
                
                if (!response.ok) {
                    throw new Error(`Failed to fetch transcript: ${response.statusText}`);
                }
                
                const transcriptData = await response.json();
                
                
                if (!transcriptData.success) {
                    throw new Error('Failed to load transcript');
                }
                
                // Show transcript modal
                showTranscriptModal(transcriptData, company, quarter);
                
            } finally {
                // Restore button
                button.innerHTML = originalText;
                button.disabled = false;
                window.transcriptLoading = false;
            }
        } else {
            // No button found, proceed without button state management
            
            try {
                
                // Determine base URL
                const baseUrl = window.location.protocol === 'file:' ? 'http://localhost:8000' : '';
                
                // Get all relevant chunks from all citations for this company/quarter
                // Convert quarter format from "2025 Q1" to "2025_q1" for comparison
                const quarterForComparison = quarter.replace(' ', '_').toLowerCase();
                
                // Try to get relevant chunks from multiple sources
                let relevantChunks = [];
                
                // First, try from window.lastCitations
                if (window.lastCitations && window.lastCitations.length > 0) {
                    relevantChunks = window.lastCitations
                        .filter(c => c.company === company && c.quarter === quarterForComparison)
                        .map(c => ({
                            chunk_text: c.chunk_text,
                            chunk_id: c.chunk_id || '',
                            relevance_score: c.relevance_score || 0
                        }));
                }
                
                // If no chunks found, try to extract from DOM
                if (relevantChunks.length === 0) {
                    const messages = document.querySelectorAll('.chat-bubble');
                    for (let message of messages) {
                        const citationsDiv = message.querySelector('.citations');
                        if (citationsDiv) {
                            const citationItems = citationsDiv.querySelectorAll('.citation-item');
                            for (let item of citationItems) {
                                const citationCompany = item.querySelector('.citation-company')?.textContent;
                                const citationQuarter = item.querySelector('.citation-quarter')?.textContent;
                                
                                if (citationCompany === company && citationQuarter === quarter) {
                                    const chunkText = item.querySelector('.citation-preview')?.textContent || '';
                                    if (chunkText) {
                                        relevantChunks.push({
                                            chunk_text: chunkText,
                                            chunk_id: `dom-${relevantChunks.length}`,
                                            relevance_score: 0.8
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
                
                // If still no chunks, create a basic one from the citation
                if (relevantChunks.length === 0 && citation.chunk_text) {
                    relevantChunks = [{
                        chunk_text: citation.chunk_text,
                        chunk_id: 'fallback-0',
                        relevance_score: 0.5
                    }];
                }
                
                // Final fallback - create a minimal chunk if we still have nothing
                if (relevantChunks.length === 0) {
                    relevantChunks = [{
                        chunk_text: `Transcript content for ${company} ${quarter}`,
                        chunk_id: 'minimal-0',
                        relevance_score: 0.3
                    }];
                }
                
                
                
                // Use the highlighted transcript endpoint
                const requestBody = {
                    ticker: company,
                    year: year,
                    quarter: quarterNum,
                    relevant_chunks: relevantChunks
                };
                
                
                const response = await fetch(`${baseUrl}/transcript/with-highlights`, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('authToken')}`,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(requestBody)
                });
                
                
                if (!response.ok) {
                    throw new Error(`Failed to fetch transcript: ${response.statusText}`);
                }
                
                const transcriptData = await response.json();
                
                
                if (!transcriptData.success) {
                    throw new Error('Failed to load transcript');
                }
                
                // Show transcript modal
                showTranscriptModal(transcriptData, company, quarter);
                
            } catch (error) {
                if (typeof showToast === 'function') {
                    showToast(`Failed to load transcript: ${error.message}`, 'error');
                }
            } finally {
                window.transcriptLoading = false;
            }
        }
        
    } catch (error) {
        if (typeof showToast === 'function') {
            showToast(`Failed to load transcript: ${error.message}`, 'error');
        }
        window.transcriptLoading = false;
    }
}

// Format transcript with proper speaker separation and chunk highlighting
function formatTranscriptWithSpeakers(transcriptText, relevantChunks = []) {
    if (!transcriptText) return 'No transcript available';
    
    // Enhanced speaker patterns to better match earnings call transcripts
    const speakerPatterns = [
        // Pattern for full names like "Kenneth J. Dorell:" or "Mark Elliot Zuckerberg:"
        /^([A-Z][a-zA-Z\s]+[A-Za-z]):\s/gm,
        // Pattern for names with middle initials like "Kenneth J. Dorell:"
        /^([A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+):\s/gm,
        // Pattern for names with periods like "Mr. John Pitzer:"
        /^([A-Z][a-z]*\.?\s*[A-Za-z\s]+[A-Za-z]):\s/gm,
        // Pattern for "Operator:" style single words
        /^([A-Z][a-z]+):\s/gm,
        // Pattern for names with hyphens like "Lip-Bu Tan:"
        /^([A-Za-z]+-[A-Za-z\s]+[A-Za-z]):\s/gm,
    ];
    
    let formattedText = transcriptText;
    let hasSpeakers = false;
    
    // Apply formatting for each speaker pattern
    speakerPatterns.forEach(pattern => {
        formattedText = formattedText.replace(pattern, (match, speaker) => {
            const cleanSpeaker = speaker.trim();
            // Skip if it's too short, contains numbers, or is a common word
            const commonWords = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'all', 'this', 'that'];
            if (cleanSpeaker.length < 3 || /\d/.test(cleanSpeaker) || commonWords.includes(cleanSpeaker.toLowerCase())) {
                return match;
            }
            hasSpeakers = true;
            return `<div class="speaker-section mb-4"><div class="speaker-name font-semibold text-accent-primary mb-2">${cleanSpeaker}:</div>`;
        });
    });
    
    // Close any unclosed speaker sections and wrap content
    if (hasSpeakers) {
        formattedText = formattedText.replace(/(<div class="speaker-section mb-4"><div class="speaker-name font-semibold text-accent-primary mb-2">[^<]*<\/div>)([^<]*?)(?=<div class="speaker-section mb-4">|$)/gs, 
            (match, speakerTag, content) => {
                // Clean up the content and wrap it properly
                const cleanContent = content.trim().replace(/\n\s*\n/g, '\n');
                return speakerTag + `<div class="speaker-content text-text-primary leading-relaxed">${cleanContent}</div></div>`;
            });
    } else {
        // If no speakers were detected, wrap the entire content
        formattedText = `<div class="speaker-section mb-4"><div class="speaker-content text-text-primary leading-relaxed">${formattedText}</div></div>`;
    }
    
    // Apply chunk highlighting if relevant chunks are provided
    if (relevantChunks && relevantChunks.length > 0) {
        formattedText = highlightRelevantChunks(formattedText, relevantChunks);
    }
    
    return formattedText;
}

// Highlight relevant chunks in the transcript
function highlightRelevantChunks(formattedText, relevantChunks) {
    let highlightedText = formattedText;
    
    relevantChunks.forEach((chunk, index) => {
        if (chunk.chunk_text && chunk.chunk_text.trim().length > 0) {
            // Escape special regex characters in the chunk text
            const escapedChunkText = chunk.chunk_text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            
            // Create a regex that matches the chunk text (case-insensitive, flexible whitespace)
            const chunkRegex = new RegExp(escapedChunkText.replace(/\s+/g, '\\s+'), 'gi');
            
            // Replace matches with highlighted version
            highlightedText = highlightedText.replace(chunkRegex, (match) => {
                const relevanceScore = chunk.relevance_score || 0;
                const highlightIntensity = Math.min(Math.max(relevanceScore * 100, 20), 100); // Scale to 20-100%
                
                return `<mark class="chunk-highlight" style="background-color: rgba(59, 130, 246, ${highlightIntensity / 100}); padding: 2px 4px; border-radius: 3px; transition: all 0.2s ease;" title="Relevance: ${(relevanceScore * 100).toFixed(1)}%">${match}</mark>`;
            });
        }
    });
    
    return highlightedText;
}

// Show transcript modal with highlighted chunks
function showTranscriptModal(transcriptData, company, quarter, relevantChunks = []) {
    
    // Determine which transcript to use
    const transcriptToShow = transcriptData.highlighted_transcript || transcriptData.transcript_text || 'No transcript available';
    
    // Remove any existing transcript modal first
    const existingModal = document.getElementById('transcriptModal');
    if (existingModal) {
        existingModal.remove();
    }
    
    // Create modal container and append directly to document.documentElement (html element)
    // This ensures it's outside any stacking contexts created by body children
    const modalOverlay = document.createElement('div');
    modalOverlay.id = 'transcriptModal';
    
    // Apply styles via setAttribute to ensure they're not overridden
    modalOverlay.setAttribute('style', `
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        z-index: 2147483647 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background: rgba(0, 0, 0, 0.75) !important;
        backdrop-filter: blur(4px) !important;
        padding: 1rem !important;
        margin: 0 !important;
        box-sizing: border-box !important;
    `);
    
    modalOverlay.innerHTML = `
        <div class="card max-w-4xl w-full max-h-[90vh] bg-bg-secondary dark:bg-bg-primary" style="position: relative !important; z-index: 2147483647 !important; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5) !important;">
            <div class="px-6 py-4 border-b border-border-primary">
                <div class="flex items-center justify-between">
                    <h3 class="text-lg font-semibold flex items-center gap-2">
                        <i class="fas fa-file-contract text-accent-primary"></i>
                        ${company} ${quarter} Earnings Transcript
                        ${relevantChunks.length > 0 ? `<span class="text-sm font-normal text-text-secondary">(${relevantChunks.length} relevant sections highlighted)</span>` : ''}
                    </h3>
                    <button id="transcriptModalCloseBtn" class="w-10 h-10 flex items-center justify-center text-text-tertiary hover:text-text-primary rounded-full hover:bg-slate-100 dark:hover:bg-slate-700/50 transition-all">
                        <i class="fas fa-times text-lg"></i>
                    </button>
                </div>
            </div>
            <div class="p-6 overflow-y-auto max-h-[70vh]">
                <div class="mb-4">
                    <div class="flex items-center gap-4 text-sm text-text-secondary">
                        <span><i class="fas fa-building mr-1"></i>${company}</span>
                        <span><i class="fas fa-calendar mr-1"></i>${quarter}</span>
                        ${transcriptData.metadata?.date ? `<span><i class="fas fa-clock mr-1"></i>${transcriptData.metadata.date}</span>` : ''}
                        ${relevantChunks.length > 0 ? `<span><i class="fas fa-highlighter mr-1"></i>${relevantChunks.length} relevant sections</span>` : ''}
                    </div>
                </div>
                <div class="prose prose-sm max-w-none bg-bg-primary p-4 rounded-lg border border-border-primary">
                    <div id="transcriptText" class="text-sm leading-relaxed">
                        ${formatTranscriptWithSpeakers(transcriptToShow, relevantChunks)}
                    </div>
                </div>
            </div>
            <div class="px-6 py-4 border-t border-border-primary bg-bg-tertiary">
                <div class="flex items-center justify-end gap-3">
                    <button class="btn btn-primary" onclick="downloadTranscript('${company}', '${quarter}')">
                        <i class="fas fa-download mr-2"></i>Download
                    </button>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modalOverlay);
    
    // Add event listeners for closing modal
    addTranscriptModalEventListeners();
    
    // Scroll to first highlighted section and add summary
    setTimeout(() => {
        const highlightedChunks = document.querySelectorAll('.highlighted-chunk');
        
        // Also check for any span elements that might contain highlights
        const allSpans = document.querySelectorAll('#transcriptText span');
        
        // Check if any spans have highlighting classes
        const spansWithClasses = Array.from(allSpans).filter(span => span.className.includes('highlighted'));
        
        if (allSpans.length > 0) {
        }
        
        // Add highlighted chunks summary
        if (highlightedChunks.length > 0) {
            addHighlightedChunksSummary(highlightedChunks.length);
        } else {
            const transcriptText = document.getElementById('transcriptText');
            if (transcriptText) {
            }
        }
        
        const firstHighlight = document.querySelector('.highlighted-chunk');
        if (firstHighlight) {
            firstHighlight.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
        
        // Add click handlers to highlighted chunks
        addHighlightClickHandlers();
    }, 100);
}

// Add click handlers to highlighted chunks
function addHighlightClickHandlers() {
    const highlightedChunks = document.querySelectorAll('.highlighted-chunk');
    
    highlightedChunks.forEach((chunk, index) => {
        chunk.addEventListener('click', function() {
            // Add a visual feedback
            this.style.transform = 'scale(1.05)';
            this.style.boxShadow = '0 6px 12px rgba(0, 112, 216, 0.4)';
            this.style.borderColor = 'var(--accent-primary)';
            
            setTimeout(() => {
                this.style.transform = '';
                this.style.boxShadow = '';
                this.style.borderColor = '';
            }, 300);
            
            // Show a tooltip or highlight info
            const chunkId = this.getAttribute('data-chunk-id');
            const chunkText = this.textContent;
            
            // Create a temporary tooltip
            showChunkTooltip(this, {
                chunkIndex: index + 1,
                chunkId: chunkId,
                text: chunkText.substring(0, 100) + (chunkText.length > 100 ? '...' : '')
            });
            
        });
        
        // Add hover effect with enhanced tooltip
        chunk.addEventListener('mouseenter', function() {
            this.title = `Relevant chunk ${index + 1} from search results - Click for details`;
            
            // Add subtle glow effect
            this.style.boxShadow = '0 2px 8px rgba(0, 112, 216, 0.3)';
        });
        
        chunk.addEventListener('mouseleave', function() {
            this.style.boxShadow = '';
        });
    });
}

// Add highlighted chunks summary to transcript modal
function addHighlightedChunksSummary(chunkCount) {
    const transcriptContainer = document.querySelector('.prose.prose-sm');
    if (!transcriptContainer) return;
    
    // Create summary element
    const summary = document.createElement('div');
    summary.className = 'mb-4 p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg';
    summary.innerHTML = `
        <div class="flex items-center gap-2 mb-2">
            <i class="fas fa-highlight text-blue-600 dark:text-blue-400"></i>
            <span class="font-semibold text-blue-800 dark:text-blue-200">${chunkCount} Relevant Chunk${chunkCount !== 1 ? 's' : ''} Found</span>
        </div>
        <p class="text-sm text-blue-700 dark:text-blue-300">Click on any highlighted text below to see details about that relevant section.</p>
    `;
    
    // Insert before transcript text
    transcriptContainer.parentNode.insertBefore(summary, transcriptContainer);
}

// Show tooltip for chunk details
function showChunkTooltip(element, chunkInfo) {
    // Remove any existing tooltip
    const existingTooltip = document.querySelector('.chunk-tooltip');
    if (existingTooltip) {
        existingTooltip.remove();
    }
    
    // Create tooltip element
    const tooltip = document.createElement('div');
    tooltip.className = 'chunk-tooltip';
    tooltip.innerHTML = `
        <div class="chunk-tooltip-content">
            <div class="chunk-tooltip-header">
                <i class="fas fa-highlight text-accent-primary"></i>
                <span>Relevant Chunk ${chunkInfo.chunkIndex}</span>
            </div>
            <div class="chunk-tooltip-body">
                <p><strong>Text:</strong> ${chunkInfo.text}</p>
                ${chunkInfo.chunkId ? `<p><strong>ID:</strong> ${chunkInfo.chunkId}</p>` : ''}
            </div>
        </div>
    `;
    
    // Position tooltip
    const rect = element.getBoundingClientRect();
    tooltip.style.position = 'fixed';
    tooltip.style.top = `${rect.top - 10}px`;
    tooltip.style.left = `${rect.left}px`;
    tooltip.style.zIndex = '10000';
    
    document.body.appendChild(tooltip);
    
    // Auto-remove after 3 seconds
    setTimeout(() => {
        if (tooltip.parentNode) {
            tooltip.remove();
        }
    }, 3000);
}

// Close transcript modal
function closeTranscriptModal() {
    // Prevent multiple simultaneous close calls
    if (window.transcriptClosing) {
        return;
    }
    
    const modal = document.getElementById('transcriptModal');
    if (!modal) {
        return;
    }
    
    window.transcriptClosing = true;
    
    // Clean up escape key listener
    if (modal._escapeHandler) {
        document.removeEventListener('keydown', modal._escapeHandler);
    }
    
    // Remove the modal
    modal.remove();
    
    // Clear the closing flag after a short delay
    setTimeout(() => {
        window.transcriptClosing = false;
    }, 100);
}

// Add click outside to close functionality
function addTranscriptModalEventListeners() {
    const modalOverlay = document.getElementById('transcriptModal');
    if (modalOverlay) {
        // Check if event listeners are already attached
        if (modalOverlay._listenersAttached) {
            return;
        }
        
        
        // Close on overlay click (outside modal)
        modalOverlay.addEventListener('click', function(e) {
            if (e.target === modalOverlay) {
                closeTranscriptModal();
            }
        });
        
        // Close button in header
        const closeBtn = document.getElementById('transcriptModalCloseBtn');
        if (closeBtn) {
            closeBtn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                closeTranscriptModal();
            });
        }
        
        // Close on Escape key
        const escapeHandler = function(e) {
            if (e.key === 'Escape') {
                closeTranscriptModal();
                document.removeEventListener('keydown', escapeHandler);
            }
        };
        document.addEventListener('keydown', escapeHandler);
        
        // Store the escape handler for cleanup and mark as attached
        modalOverlay._escapeHandler = escapeHandler;
        modalOverlay._listenersAttached = true;
    }
}

// Download transcript
function downloadTranscript(company, quarter) {
    // This would implement transcript download functionality
    // Implementation would go here
}


// Escape HTML function
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Expose functions globally for app.js integration
window.viewCompleteTranscript = viewCompleteTranscript;
window.showTranscriptModal = showTranscriptModal;
window.closeTranscriptModal = closeTranscriptModal;

window.testChatButton = function() {
    const sendButton = document.getElementById('sendChatButton');
    
    // Test button state switching
    if (window.chatInterface) {
        window.chatInterface.isLoading = true;
        window.chatInterface.updateButtonStates();
        
        window.chatInterface.isLoading = false;
        window.chatInterface.updateButtonStates();
    }
};

window.testStopButton = function() {
    if (window.chatInterface) {
        window.chatInterface.isLoading = true;
        window.chatInterface.updateButtonStates();
    }
};

// Initialize chat when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Wait a bit for other scripts to load
    setTimeout(() => {
        if (typeof CONFIG !== 'undefined' || typeof window.STRATALENS_CONFIG !== 'undefined') {
            window.chatInterface = new ChatInterface();
        } else {
            // Retry after another second
            setTimeout(() => {
                if (typeof window.STRATALENS_CONFIG !== 'undefined') {
                    window.chatInterface = new ChatInterface();
                } else {
                    // Fallback: create instance anyway for ticker autocomplete
                    window.chatInterface = new ChatInterface();
                }
            }, 1000);
        }
    }, 1000);
});

// Toggle sources visibility
function toggleSources(button) {
    const citationsDiv = button.closest('.citations');
    const citationsContent = citationsDiv.querySelector('.citations-content');
    const chevronIcon = button.querySelector('i');
    
    if (citationsDiv.classList.contains('collapsed')) {
        // Expand
        citationsDiv.classList.remove('collapsed');
        citationsContent.style.display = 'block';
        chevronIcon.className = 'fas fa-chevron-up';
    } else {
        // Collapse
        citationsDiv.classList.add('collapsed');
        citationsContent.style.display = 'none';
        chevronIcon.className = 'fas fa-chevron-down';
    }
}

// Export for global access
window.ChatInterface = ChatInterface;
window.ChatHistoryManager = ChatHistoryManager;
window.insertQuickMessage = insertQuickMessage;
window.toggleCitation = toggleCitation;
window.toggleSources = toggleSources;

// Test function for ticker autocomplete
window.testTickerAutocomplete = function() {
    console.log('Testing ticker autocomplete...');
    const chatInput = document.getElementById('chatInput');
    if (!chatInput) {
        console.log('chatInput not found');
        return;
    }
    
    if (!window.chatInterface) {
        console.log('chatInterface not initialized');
        return;
    }
    
    console.log('chatInterface found, testing...');
    chatInput.value = '$AAP';
    chatInput.dispatchEvent(new Event('input'));
};