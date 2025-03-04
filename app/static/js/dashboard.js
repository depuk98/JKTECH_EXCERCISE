/* Dashboard JavaScript */

document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const searchForm = document.getElementById('search-form');
    const searchInput = document.getElementById('search-input');
    const searchButton = document.getElementById('search-button');
    const searchResultsContainer = document.getElementById('search-results-container');
    const searchResults = document.getElementById('search-results');
    const searchLoading = document.getElementById('search-loading');
    const noResults = document.getElementById('no-results');
    
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const selectFileBtn = document.getElementById('select-file-btn');
    const uploadContent = document.getElementById('upload-content');
    const uploadProgress = document.getElementById('upload-progress');
    const progressBar = document.querySelector('.progress-bar');
    const uploadStatus = document.getElementById('upload-status');
    
    const documentList = document.getElementById('document-list');
    const noDocuments = document.getElementById('no-documents');
    const documentError = document.getElementById('document-error');
    const documentSuccess = document.getElementById('document-success');
    
    const usernameDisplay = document.getElementById('username-display');
    const emailDisplay = document.getElementById('email-display');
    const createdAtDisplay = document.getElementById('created-at-display');
    
    const updateForm = document.getElementById('update-form');
    const emailInput = document.getElementById('email');
    const passwordInput = document.getElementById('password');
    const confirmPasswordInput = document.getElementById('confirm-password');
    const errorMessage = document.getElementById('error-message');
    const successMessage = document.getElementById('success-message');
    
    // Stats elements
    const statTotalDocuments = document.getElementById('stat-total-documents');
    const statProcessedDocuments = document.getElementById('stat-processed-documents');
    const statSearches = document.getElementById('stat-searches');
    const statQuestions = document.getElementById('stat-questions');
    
    // Initialize counters from local storage
    let searchCount = parseInt(localStorage.getItem('searchCount') || '0');
    let questionCount = parseInt(localStorage.getItem('questionCount') || '0');
    statSearches.textContent = searchCount;
    statQuestions.textContent = questionCount;
    
    // Document upload handling
    if (dropArea) {
        // Prevent default behavior to open dropped files
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        // Highlight drop area when drag item enters
        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight() {
            dropArea.classList.add('highlight');
        }
        
        function unhighlight() {
            dropArea.classList.remove('highlight');
        }
        
        // Handle drop
        dropArea.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length > 0) {
                handleFiles(files[0]);
            }
        }
        
        // Handle click to select files
        selectFileBtn.addEventListener('click', () => {
            fileInput.click();
        });
        
        dropArea.addEventListener('click', () => {
            if (event.target === dropArea || event.target === uploadContent) {
                fileInput.click();
            }
        });
        
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                handleFiles(fileInput.files[0]);
            }
        });
        
        function handleFiles(file) {
            uploadFile(file);
        }
        
        // Handle file upload with proper auth
        function uploadFile(file) {
            // Create form data
            const formData = new FormData();
            formData.append('file', file);
            
            // Show upload progress
            showUploadProgress();
            
            // Upload using apiFormPost
            apiFormPost('/api/documents/upload', formData, updateProgressBar)
                .then(response => {
                    if (!response || !response.ok) {
                        throw new Error('Upload failed');
                    }
                    return response.json();
                })
                .then(data => {
                    // Update progress to 100%
                    updateProgressBar(100);
                    uploadStatus.textContent = 'Upload complete!';
                    
                    // Show success message
                    documentSuccess.classList.remove('d-none');
                    
                    // Reset the upload area after a delay
                    setTimeout(resetUploadArea, 3000);
                    
                    // Reload document list
                    loadDocuments();
                    
                    // Hide success message after delay
                    setTimeout(() => {
                        documentSuccess.classList.add('d-none');
                    }, 3000);
                })
                .catch(error => {
                    console.error('Upload error:', error);
                    documentError.textContent = `Upload failed: ${error.message}`;
                    documentError.classList.remove('d-none');
                    uploadError('Upload failed');
                    
                    // Reset the upload area after a delay
                    setTimeout(resetUploadArea, 3000);
                });
        }
        
        function showUploadProgress() {
            uploadContent.classList.add('d-none');
            uploadProgress.classList.remove('d-none');
        }
        
        function resetUploadArea() {
            uploadContent.classList.remove('d-none');
            uploadProgress.classList.add('d-none');
            progressBar.style.width = '0%';
            progressBar.textContent = '0%';
            uploadStatus.textContent = 'Uploading document...';
            fileInput.value = '';
        }
        
        function resetUploadStatus() {
            documentError.classList.add('d-none');
            documentSuccess.classList.add('d-none');
        }
        
        function updateProgressBar(percent) {
            progressBar.style.width = percent + '%';
            progressBar.textContent = percent + '%';
        }
        
        function uploadError(message) {
            uploadStatus.textContent = 'Upload failed';
            progressBar.classList.remove('progress-bar-animated');
            progressBar.classList.remove('bg-primary');
            progressBar.classList.add('bg-danger');
            
            // Show error message
            documentError.textContent = message;
            documentError.classList.remove('d-none');
            
            // Reset the upload area after a delay
            setTimeout(resetUploadArea, 3000);
        }
    }
    
    // Load and display documents
    function loadDocuments() {
        console.log('Loading documents...');
        apiGet('/api/documents')
            .then(response => {
                if (!response || !response.ok) {
                    throw new Error('Failed to fetch documents');
                }
                return response.json();
            })
            .then(data => {
                console.log('Document data received:', data);
                // Check for documents property first (API returns {total: X, documents: [...]}), 
                // then items (in case API returns {total: X, items: [...]}),
                // or fall back to using the array directly if neither exists
                const items = data.documents || data.items || (Array.isArray(data) ? data : []);
                console.log('Documents to display:', items);
                displayDocuments(items);
                updateStats(items);
            })
            .catch(error => {
                console.error('Error loading documents:', error);
                documentList.innerHTML = `<tr><td colspan="5" class="text-center text-danger">Failed to load documents: ${error.message}</td></tr>`;
            });
    }
    
    function updateStats(documents) {
        if (!documents) return;
        
        // Update document stats
        statTotalDocuments.textContent = documents.length;
        
        // Count processed documents
        const processedDocs = documents.filter(doc => doc.status === 'processed').length;
        statProcessedDocuments.textContent = processedDocs;
    }
    
    function displayDocuments(documents) {
        if (!documents || documents.length === 0) {
            documentList.innerHTML = '';
            noDocuments.classList.remove('d-none');
            return;
        }
        
        noDocuments.classList.add('d-none');
        
        // Sort documents by creation date (newest first)
        documents.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
        
        let html = '';
        documents.forEach(doc => {
            const createdDate = new Date(doc.created_at).toLocaleDateString();
            
            html += `
                <tr>
                    <td>${doc.filename}</td>
                    <td>${getContentTypeDisplay(doc.content_type)}</td>
                    <td>
                        <span class="status-badge ${doc.status}">
                            ${capitalizeFirst(doc.status)}
                        </span>
                    </td>
                    <td>${createdDate}</td>
                    <td>
                        <div class="btn-group btn-group-sm">
                            <button class="btn btn-outline-primary view-document" data-id="${doc.id}" title="View Document">
                                <i class="bi bi-eye"></i>
                            </button>
                            <button class="btn btn-outline-danger delete-document" data-id="${doc.id}" title="Delete Document">
                                <i class="bi bi-trash"></i>
                            </button>
                        </div>
                    </td>
                </tr>
            `;
        });
        
        documentList.innerHTML = html;
        
        // Add event listeners to buttons
        document.querySelectorAll('.delete-document').forEach(button => {
            button.addEventListener('click', function() {
                const docId = this.getAttribute('data-id');
                confirmDeleteDocument(docId);
            });
        });
        
        document.querySelectorAll('.view-document').forEach(button => {
            button.addEventListener('click', function() {
                const docId = this.getAttribute('data-id');
                viewDocument(docId);
            });
        });
    }
    
    function confirmDeleteDocument(docId) {
        if (confirm('Are you sure you want to delete this document? This action cannot be undone.')) {
            deleteDocument(docId);
        }
    }
    
    function deleteDocument(docId) {
        apiDelete(`/api/documents/${docId}`)
            .then(response => {
                if (!response) {
                    throw new Error('Failed to delete document');
                }
                
                if (!response.ok) {
                    throw new Error(`Failed to delete document: ${response.status} ${response.statusText}`);
                }
                
                // For 204 No Content responses, we don't need to parse the response body
                if (response.status === 204) {
                    return null;
                }
                
                return safeJsonParse(response);
            })
            .then(data => {
                // Refresh the document list
                loadDocuments();
                // Show success message
                documentSuccess.textContent = 'Document deleted successfully!';
                documentSuccess.classList.remove('d-none');
                setTimeout(() => {
                    documentSuccess.classList.add('d-none');
                }, 3000);
            })
            .catch(error => {
                console.error('Error deleting document:', error);
                documentError.textContent = `Failed to delete document: ${error.message}`;
                documentError.classList.remove('d-none');
                setTimeout(() => {
                    documentError.classList.add('d-none');
                }, 3000);
            });
    }
    
    function viewDocument(docId) {
        // Redirect to document view page
        window.location.href = `/documents/${docId}`;
    }
    
    // Helper functions
    function getContentTypeDisplay(contentType) {
        const contentTypeMap = {
            'application/pdf': 'PDF',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'DOCX',
            'text/plain': 'TXT'
        };
        
        return contentTypeMap[contentType] || contentType;
    }
    
    function capitalizeFirst(string) {
        return string.charAt(0).toUpperCase() + string.slice(1);
    }
    
    // Search functionality
    if (searchForm) {
        searchForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const query = searchInput.value.trim();
            
            if (!query) {
                return;
            }
            
            // Update search count
            searchCount++;
            localStorage.setItem('searchCount', searchCount);
            statSearches.textContent = searchCount;
            
            performSearch(query);
        });
    }
    
    function performSearch(query) {
        // Show loading indicator
        searchLoading.classList.remove('d-none');
        searchResults.innerHTML = '';
        noResults.classList.add('d-none');
        searchResultsContainer.classList.remove('d-none');
        
        // Update search count in statistics
        searchCount++;
        localStorage.setItem('searchCount', searchCount);
        statSearches.textContent = searchCount;
        
        console.log('Performing search for:', query);
        apiGet(`/api/documents/search?query=${encodeURIComponent(query)}`)
            .then(response => {
                if (!response || !response.ok) {
                    throw new Error('Search failed');
                }
                return response.json();
            })
            .then(data => {
                console.log('Search results received:', data);
                // Hide loading indicator
                searchLoading.classList.add('d-none');
                
                // Check different data structures
                let results = [];
                if (data.items && Array.isArray(data.items)) {
                    results = data.items;
                } else if (Array.isArray(data)) {
                    results = data;
                } else if (data.results && Array.isArray(data.results)) {
                    results = data.results;
                }
                
                console.log('Parsed search results:', results);
                
                // Display results
                if (results.length > 0) {
                    displaySearchResults(results, query);
                } else {
                    noResults.classList.remove('d-none');
                }
            })
            .catch(error => {
                console.error('Search error:', error);
                searchLoading.classList.add('d-none');
                searchResults.innerHTML = `<div class="alert alert-danger">Search failed: ${error.message}</div>`;
            });
    }
    
    function displaySearchResults(results, query) {
        console.log('Displaying search results:', results);
        searchLoading.classList.add('d-none');
        
        if (!results || results.length === 0) {
            noResults.classList.remove('d-none');
            return;
        }
        
        let html = '';
        results.forEach((result, index) => {
            // Add a delay to the animation for each result
            const animationDelay = index * 0.1;
            
            // Normalize result structure
            const document = result.document || result.doc || {};
            const documentId = result.document_id || document.id || result.id || '';
            const documentName = document.filename || result.document_name || 'Document';
            const score = result.similarity_score || result.score || result.similarity || 0.7;
            const text = result.text || result.content || result.chunk_text || '';
            
            html += `
                <div class="list-group-item search-result p-3" style="animation-delay: ${animationDelay}s">
                    <div class="search-result-header d-flex justify-content-between align-items-center mb-1">
                        <span class="document-name">
                            <i class="bi bi-file-earmark-text me-1"></i> ${documentName}
                        </span>
                        <span class="search-result-score">
                            ${Math.round(score * 100)}% match
                        </span>
                    </div>
                    <p class="search-result-text">${highlightQuery(text, query)}</p>
                    <div class="mt-2">
                        <a href="/documents/${documentId}" class="btn btn-sm btn-outline-primary">
                            <i class="bi bi-eye me-1"></i> View Document
                        </a>
                        <a href="/qa?document=${documentId}" class="btn btn-sm btn-outline-secondary">
                            <i class="bi bi-question-circle me-1"></i> Ask about this
                        </a>
                    </div>
                </div>
            `;
        });
        
        searchResults.innerHTML = html;
        searchResultsContainer.classList.remove('d-none');
    }
    
    function highlightQuery(text, query) {
        // Simple highlighting - split query into words and highlight each occurrence
        const words = query.toLowerCase().split(/\s+/).filter(w => w.length > 2);
        let highlighted = text;
        
        words.forEach(word => {
            // Use regex to replace all occurrences with highlighted version
            const regex = new RegExp(`(${word})`, 'gi');
            highlighted = highlighted.replace(regex, '<span class="highlight">$1</span>');
        });
        
        return highlighted;
    }
    
    // User profile
    function loadUserProfile() {
        apiGet('/api/users/me')
            .then(response => {
                if (!response || !response.ok) {
                    throw new Error('Failed to load user profile');
                }
                return response.json();
            })
            .then(data => {
                displayUserProfile(data);
                
                // Pre-fill the email field with current email
                if (emailInput) {
                    emailInput.value = data.email;
                }
            })
            .catch(error => {
                console.error('Error loading user profile:', error);
                if (usernameDisplay) usernameDisplay.textContent = 'Error loading profile';
                if (emailDisplay) emailDisplay.textContent = 'Error loading profile';
                if (createdAtDisplay) createdAtDisplay.textContent = 'Error loading profile';
            });
    }
    
    function displayUserProfile(user) {
        usernameDisplay.textContent = user.username;
        emailDisplay.textContent = user.email;
        emailInput.value = user.email;
        
        const createdDate = new Date(user.created_at).toLocaleString();
        createdAtDisplay.textContent = createdDate;
    }
    
    // Update profile form
    if (updateForm) {
        updateForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const email = emailInput.value;
            const password = passwordInput.value;
            const confirmPassword = confirmPasswordInput.value;
            
            // Clear previous messages
            errorMessage.classList.add('d-none');
            successMessage.classList.add('d-none');
            
            // Validate passwords match if provided
            if (password && password !== confirmPassword) {
                errorMessage.textContent = 'Passwords do not match';
                errorMessage.classList.remove('d-none');
                return;
            }
            
            // Prepare data object, only including password if provided
            const userData = { email };
            if (password) {
                userData.password = password;
            }
            
            // Update user profile
            apiPut('/api/users/me', userData)
                .then(response => {
                    if (!response || !response.ok) {
                        return response.json().then(data => {
                            throw new Error(data.detail || 'Update failed');
                        });
                    }
                    return response.json();
                })
                .then(data => {
                    // Show success message
                    successMessage.classList.remove('d-none');
                    
                    // Clear password fields
                    passwordInput.value = '';
                    confirmPasswordInput.value = '';
                    
                    // Reload user data
                    loadUserProfile();
                    
                    // Hide success message after delay
                    setTimeout(() => {
                        successMessage.classList.add('d-none');
                    }, 3000);
                })
                .catch(error => {
                    console.error('Update error:', error);
                    errorMessage.textContent = error.message;
                    errorMessage.classList.remove('d-none');
                });
        });
    }
    
    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.classList.remove('d-none');
    }
    
    // Initialize
    loadDocuments();
    loadUserProfile();
}); 