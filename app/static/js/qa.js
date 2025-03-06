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
                
                // Replace citation markers like [1], [2], etc. with more visible interactive elements
                for (let i = 1; i <= result.citations.length; i++) {
                    const regex = new RegExp(`\\[${i}\\]`, 'g');
                    processedAnswer = processedAnswer.replace(
                        regex, 
                        `<a href="#citation-${i}" class="citation-ref" data-citation-id="${i}" 
                           onclick="event.preventDefault(); highlightCitation(${i});" 
                           title="View source ${i}: ${result.citations[i-1].filename || 'Unknown'}">
                           [${i}]
                         </a>`
                    );
                }
                
                answerContent.innerHTML = processedAnswer;
                
                // Add the highlight citation function if it doesn't exist
                if (!window.highlightCitation) {
                    window.highlightCitation = function(id) {
                        // Remove highlight from all citations
                        document.querySelectorAll('.citation-item').forEach(item => {
                            item.classList.remove('citation-highlight');
                        });
                        
                        // Add highlight to the clicked citation
                        const citationItem = document.getElementById(`citation-${id}`);
                        if (citationItem) {
                            citationItem.classList.add('citation-highlight');
                            citationItem.scrollIntoView({ behavior: 'smooth', block: 'center' });
                        }
                        
                        // Also highlight all references to this citation in the text
                        document.querySelectorAll(`.citation-ref[data-citation-id="${id}"]`).forEach(ref => {
                            ref.classList.add('citation-active');
                            // Remove the active class after a delay
                            setTimeout(() => {
                                ref.classList.remove('citation-active');
                            }, 3000);
                        });
                    }
                }
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
                citationElement.id = `citation-${index + 1}`;
                
                // Extract filename without path
                const filename = citation.filename || 'Unknown Document';
                
                // Format page info if available
                const pageInfo = citation.metadata && citation.metadata.page 
                    ? `Page ${citation.metadata.page}` 
                    : '';
                
                // Format title if available
                const titleInfo = citation.metadata && citation.metadata.title 
                    ? citation.metadata.title 
                    : '';
                
                // Get relevance score if available
                const scoreInfo = citation.score !== undefined 
                    ? Math.round(citation.score * 100) 
                    : null;
                
                // Get snippet if available
                const snippet = citation.snippet || '';
                
                // Build the citation HTML with more details, snippet, and a "jump to text" button
                citationElement.innerHTML = `
                    <div class="d-flex justify-content-between align-items-start">
                        <div>
                            <span class="citation-number badge bg-primary me-2">${index + 1}</span>
                            <span class="citation-text fw-bold">${filename}</span>
                            ${scoreInfo ? `<span class="badge bg-secondary ms-2" title="Relevance score">Match: ${scoreInfo}%</span>` : ''}
                        </div>
                        <button class="btn btn-sm btn-outline-primary citation-jump" 
                                onclick="findCitationInText(${index + 1})" 
                                title="Find in text">
                            <i class="bi bi-arrow-up-square"></i>
                        </button>
                    </div>
                    <div class="citation-source small text-muted mt-1">
                        <div class="d-flex gap-2 mb-1">
                            <i class="bi bi-file-earmark-text"></i>
                            ${pageInfo ? `<span>${pageInfo}</span>` : ''}
                            ${titleInfo ? `<span class="text-truncate">${titleInfo}</span>` : ''}
                        </div>
                        ${snippet ? `
                            <div class="citation-snippet mt-2 p-2 bg-white rounded border">
                                <small class="text-muted fw-light">Excerpt:</small>
                                <div class="small fw-light">${snippet}</div>
                            </div>
                        ` : ''}
                    </div>
                `;
                citationsList.appendChild(citationElement);
            });
            
            // Add the find citation in text function if it doesn't exist
            if (!window.findCitationInText) {
                window.findCitationInText = function(id) {
                    const citationRef = document.querySelector(`.citation-ref[data-citation-id="${id}"]`);
                    if (citationRef) {
                        // Highlight all instances of this citation
                        document.querySelectorAll(`.citation-ref[data-citation-id="${id}"]`).forEach(ref => {
                            ref.classList.add('citation-active');
                            // Remove the highlight after a delay
                            setTimeout(() => {
                                ref.classList.remove('citation-active');
                            }, 3000);
                        });
                        
                        // Scroll to the first instance
                        citationRef.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                }
            }
            
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