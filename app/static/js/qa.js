/* JavaScript for Document Q&A Interface - v1.2 */

document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const documentSelector = document.getElementById('document-selector');
    const loadingDocuments = document.getElementById('loading-documents');
    const questionInput = document.getElementById('question-input');
    const submitButton = document.getElementById('submit-question');
    const answerContainer = document.getElementById('answer-container');
    const loadingSpinner = document.getElementById('loading-spinner');
    const answerContent = document.getElementById('answer-content');
    const citationsContainer = document.getElementById('citations-container');
    const selectAllCheckbox = document.getElementById('select-all-documents');
    const errorContainer = document.getElementById('error-container');
    const processingTime = document.getElementById('processing-time');

    // Fetch available documents
    fetchAvailableDocuments();

    // Event Listeners
    submitButton.addEventListener('click', handleQuestionSubmit);
    questionInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleQuestionSubmit();
        }
    });

    // Select all documents checkbox
    if (selectAllCheckbox) {
        selectAllCheckbox.addEventListener('change', (e) => {
            const isChecked = e.target.checked;
            const documentCheckboxes = document.querySelectorAll('.document-checkbox');
            
            documentCheckboxes.forEach(checkbox => {
                checkbox.checked = isChecked;
                checkbox.disabled = isChecked;
            });
        });
    }

    // Functions
    async function fetchAvailableDocuments() {
        try {
            // Use the same API endpoint as the dashboard instead of a duplicate endpoint
            const response = await apiGet('/api/documents');
            
            if (!response || !response.ok) {
                throw new Error('Failed to fetch documents');
            }

            const data = await response.json();
            // Handle documents from the standard documents API
            const documents = data.documents || data.items || (Array.isArray(data) ? data : []);
            
            // Filter to only show processed documents for Q&A
            const processedDocuments = documents.filter(doc => doc.status === "processed");
            
            renderDocumentSelector(processedDocuments);
        } catch (error) {
            console.error('Error fetching documents:', error);
            documentSelector.innerHTML = `
                <div class="alert alert-danger">
                    Failed to load documents. Please refresh the page.
                </div>
            `;
        } finally {
            if (loadingDocuments) {
                loadingDocuments.style.display = 'none';
            }
        }
    }

    function renderDocumentSelector(documents) {
        if (documents.length === 0) {
            documentSelector.innerHTML = `
                <div class="text-center py-3">
                    <p class="mb-0">No documents available. Upload documents from the dashboard first.</p>
                </div>
            `;
            return;
        }

        const docElements = documents.map((doc, index) => `
            <div class="form-check">
                <input class="form-check-input document-checkbox" type="checkbox" value="${doc.id}" 
                       id="doc-${doc.id}" disabled ${selectAllCheckbox && selectAllCheckbox.checked ? 'checked' : ''}>
                <label class="form-check-label d-block text-truncate" for="doc-${doc.id}" title="${doc.filename}">
                    ${doc.filename}
                </label>
            </div>
        `).join('');

        documentSelector.innerHTML = `
            <div class="document-options">
                ${docElements}
            </div>
        `;
    }

    async function handleQuestionSubmit() {
        const question = questionInput.value.trim();
        
        if (!question) {
            showError('Please enter a question');
            return;
        }

        // Get selected document IDs
        let documentIds = [];
        
        if (selectAllCheckbox && selectAllCheckbox.checked) {
            // All documents selected
            documentIds = [];  // Empty array will be treated as "all documents" on the server
        } else {
            // Get checked document IDs
            const checkboxes = document.querySelectorAll('.document-checkbox:checked');
            documentIds = Array.from(checkboxes).map(cb => cb.value);
            
            if (documentIds.length === 0) {
                showError('Please select at least one document');
                return;
            }
        }

        // Set loading state
        setLoadingState(true);
        hideError();
        
        // Record start time for processing
        const startTime = new Date().getTime();
        
        // Update question count in localStorage if applicable
        const questionCount = parseInt(localStorage.getItem('questionCount') || '0') + 1;
        localStorage.setItem('questionCount', questionCount);
        if (window.statQuestions) {
            window.statQuestions.textContent = questionCount;
        }

        try {
            console.log("Sending question to API:", {
                query: question,
                document_ids: documentIds,
                use_cache: true
            });
            
            // Use the specialized function for asking questions
            const response = await apiAskQuestion(question, documentIds, true);
            
            if (!response || !response.ok) {
                const errorData = await response.json();
                console.error("API Error Response:", errorData);
                
                // Check for Ollama service errors (HTTP 503)
                if (response.status === 503) {
                    showError(`Ollama Service Error: ${errorData.detail || 'The AI service is currently unavailable'}`);
                    
                    // Add helpful troubleshooting steps
                    const helpText = document.createElement('div');
                    helpText.className = 'mt-3 small';
                    helpText.innerHTML = `
                        <p><strong>Troubleshooting steps:</strong></p>
                        <ol>
                            <li>Make sure Ollama is running: <code>ollama serve</code></li>
                            <li>Check if the model is available: <code>ollama list</code></li>
                            <li>If not, pull the model: <code>ollama pull llama3.2</code></li>
                        </ol>
                    `;
                    errorContainer.appendChild(helpText);
                } else {
                    throw new Error(errorData.detail || 'Failed to process question');
                }
                return;
            }

            const result = await response.json();
            
            // Calculate processing time
            const endTime = new Date().getTime();
            const timeTaken = (endTime - startTime) / 1000; // Convert to seconds
            
            if (processingTime) {
                processingTime.textContent = `Processed in ${timeTaken.toFixed(2)}s`;
            }
            
            renderAnswer(result);
        } catch (error) {
            console.error('Error asking question:', error);
            showError(`Error: ${error.message}`);
            answerContainer.classList.add('d-none');
        } finally {
            setLoadingState(false);
        }
    }

    function renderAnswer(result) {
        // Display the answer 
        if (result.answer) {
            // Check if citations are available
            if (result.citations && result.citations.length > 0) {
                // Process answer text to highlight citation numbers
                let processedAnswer = result.answer;
                
                // Replace citation markers like [1], [2], etc. with more visible span elements
                for (let i = 1; i <= result.citations.length; i++) {
                    const regex = new RegExp(`\\[${i}\\]`, 'g');
                    processedAnswer = processedAnswer.replace(
                        regex, 
                        `<span class="citation-ref" title="Citation ${i}">[${i}]</span>`
                    );
                }
                
                answerContent.innerHTML = processedAnswer;
            } else {
                answerContent.innerHTML = result.answer;
            }
        } else {
            answerContent.innerHTML = "No answer was generated.";
        }
        
        // Show the answer container
        answerContainer.classList.remove('d-none');
        
        // Display provider information if available
        const providerBadge = document.getElementById('provider-badge');
        if (providerBadge) {
            if (result.metadata && result.metadata.error) {
                providerBadge.textContent = 'Service Error';
                providerBadge.classList.remove('d-none');
                providerBadge.classList.add('bg-danger');
                providerBadge.classList.remove('bg-info');
            } else if (result.metadata && result.metadata.provider) {
                providerBadge.textContent = `${result.metadata.provider} ${result.metadata.model || ''}`;
                providerBadge.classList.remove('d-none');
            } else {
                providerBadge.textContent = 'AI Provider';
                providerBadge.classList.remove('d-none');
            }
        }
        
        // Handle citations if available
        const citationsContainer = document.getElementById('citations-container');
        const citationsList = document.getElementById('citations-list');
        
        if (result.citations && result.citations.length > 0) {
            console.log("Found citations:", result.citations);
            
            // Clear previous citations
            citationsList.innerHTML = '';
            
            // Add each citation
            result.citations.forEach((citation, index) => {
                const citationElement = document.createElement('div');
                citationElement.className = 'citation-item py-2';
                
                // Extract filename without path
                const filename = citation.filename || 'Unknown Document';
                
                // Format page info if available
                const pageInfo = citation.metadata && citation.metadata.page 
                    ? `(Page ${citation.metadata.page})` 
                    : '';
                
                // Build the citation HTML
                citationElement.innerHTML = `
                    <strong>${index + 1}.</strong> 
                    <span class="citation-text">${filename}</span>
                    <div class="citation-source small text-muted">
                        <i class="bi bi-file-earmark-text me-1"></i>
                        ${pageInfo}
                        ${citation.metadata && citation.metadata.title ? `<div class="mt-1">${citation.metadata.title}</div>` : ''}
                    </div>
                `;
                citationsList.appendChild(citationElement);
            });
            
            // Make sure the citations container is visible
            citationsContainer.classList.remove('d-none');
        } else {
            // No citations available
            console.log("No citations found in response");
            citationsContainer.classList.add('d-none');
        }
    }

    function setLoadingState(isLoading) {
        if (isLoading) {
            submitButton.disabled = true;
            loadingSpinner.classList.add('processing');
            questionInput.disabled = true;
        } else {
            submitButton.disabled = false;
            loadingSpinner.classList.remove('processing');
            questionInput.disabled = false;
        }
    }

    function showError(message) {
        errorContainer.textContent = message;
        errorContainer.classList.remove('d-none');
    }

    function hideError() {
        errorContainer.classList.add('d-none');
    }
}); 