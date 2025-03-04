/**
 * API utilities for authenticated requests
 */

/**
 * Make an authenticated API request
 * @param {string} url - The API endpoint URL
 * @param {Object} options - Fetch API options
 * @returns {Promise} - Fetch promise
 */
async function apiRequest(url, options = {}) {
    const token = localStorage.getItem('token');
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
        }
    };
    
    // Merge options
    const mergedOptions = { 
        ...defaultOptions, 
        ...options,
        headers: {
            ...defaultOptions.headers,
            ...options.headers
        }
    };
    
    // Add authorization header if token exists
    if (token) {
        mergedOptions.headers['Authorization'] = `Bearer ${token}`;
        console.log('Adding token to request:', url, 'Token exists:', !!token);
    } else {
        console.log('No token found for request:', url);
    }
    
    try {
        console.log('Sending API request to:', url, 'with options:', JSON.stringify(mergedOptions));
        const response = await fetch(url, mergedOptions);
        console.log('API response:', url, 'Status:', response.status);
        
        // If unauthorized and not already on login/signup page, redirect to login
        if (response.status === 401 && 
            !window.location.pathname.includes('/login') && 
            !window.location.pathname.includes('/signup') &&
            !window.location.pathname.includes('/landing')) {
            console.log('Unauthorized access, redirecting to login...');
            localStorage.removeItem('token');
            window.location.href = '/login';
            return null;
        }
        
        return response;
    } catch (error) {
        console.error('API request error:', error);
        throw error;
    }
}

/**
 * Simple GET request with authentication
 * @param {string} url - The API endpoint URL
 * @returns {Promise} - Fetch promise
 */
async function apiGet(url) {
    return apiRequest(url, { method: 'GET' });
}

/**
 * POST request with authentication
 * @param {string} url - The API endpoint URL
 * @param {Object} data - The data to send
 * @returns {Promise} - Fetch promise
 */
async function apiPost(url, data) {
    return apiRequest(url, {
        method: 'POST',
        body: JSON.stringify(data)
    });
}

/**
 * PUT request with authentication
 * @param {string} url - The API endpoint URL
 * @param {Object} data - The data to send
 * @returns {Promise} - Fetch promise
 */
async function apiPut(url, data) {
    return apiRequest(url, {
        method: 'PUT',
        body: JSON.stringify(data)
    });
}

/**
 * DELETE request with authentication
 * @param {string} url - The API endpoint URL
 * @returns {Promise} - Fetch promise
 */
async function apiDelete(url) {
    return apiRequest(url, { method: 'DELETE' });
}

/**
 * Form POST request with authentication (for file uploads)
 * @param {string} url - The API endpoint URL
 * @param {FormData} formData - The form data to send
 * @param {Function} progressCallback - Optional callback for upload progress
 * @returns {Promise} - Fetch promise
 */
async function apiFormPost(url, formData, progressCallback = null) {
    const token = localStorage.getItem('token');
    const options = {
        method: 'POST',
        body: formData,
        headers: {}
    };
    
    // Add authorization header if token exists
    if (token) {
        options.headers['Authorization'] = `Bearer ${token}`;
    }
    
    // If XHR with progress is needed
    if (progressCallback) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            xhr.open('POST', url);
            
            if (token) {
                xhr.setRequestHeader('Authorization', `Bearer ${token}`);
            }
            
            xhr.upload.addEventListener('progress', (event) => {
                if (event.lengthComputable) {
                    const percentComplete = Math.round((event.loaded / event.total) * 100);
                    progressCallback(percentComplete);
                }
            });
            
            xhr.onload = function() {
                if (xhr.status >= 200 && xhr.status < 300) {
                    resolve({
                        ok: true,
                        status: xhr.status,
                        json: () => JSON.parse(xhr.responseText)
                    });
                } else {
                    reject({
                        status: xhr.status,
                        statusText: xhr.statusText
                    });
                }
            };
            
            xhr.onerror = function() {
                reject({
                    status: xhr.status,
                    statusText: xhr.statusText
                });
            };
            
            xhr.send(formData);
        });
    }
    
    // Regular fetch without progress
    return apiRequest(url, options);
}

/**
 * Specialized function for asking questions to the Q&A system
 * Ensures consistent parameter naming between frontend and backend
 * 
 * @param {string} questionText - The user's question text
 * @param {Array} documentIds - Array of document IDs to search within  
 * @param {boolean} useCache - Whether to use cached responses
 * @returns {Promise} - Fetch promise
 */
async function apiAskQuestion(questionText, documentIds = [], useCache = true) {
    // Log the request for debugging
    console.log('Asking question with parameters:', {
        query: questionText,
        document_ids: documentIds,
        use_cache: useCache
    });
    
    // Always use 'query' parameter name as expected by backend
    return apiPost('/api/qa/ask', {
        query: questionText,  // This is the key parameter name that must match backend
        document_ids: documentIds,
        use_cache: useCache
    });
}

/**
 * Safely parse a response as JSON
 * @param {Response} response - Fetch Response object
 * @returns {Promise} - Promise resolving to parsed JSON or null for empty responses
 */
async function safeJsonParse(response) {
    // No content or no content-type
    if (response.status === 204 || !response.headers.get('content-type')) {
        return null;
    }
    
    // Content-type is not JSON
    if (!response.headers.get('content-type').includes('application/json')) {
        const text = await response.text();
        console.warn('Response is not JSON:', text);
        return text;
    }
    
    try {
        const text = await response.text();
        // If the text is empty, return null instead of trying to parse
        if (!text || text.trim() === '') {
            return null;
        }
        return JSON.parse(text);
    } catch (error) {
        console.error('Error parsing JSON response:', error);
        throw new Error('Failed to parse response: ' + error.message);
    }
} 