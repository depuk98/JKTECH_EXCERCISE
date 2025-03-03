/**
 * Main JavaScript file for the application.
 */

// Check if user is authenticated
function checkAuth() {
    const token = localStorage.getItem('token');
    if (token) {
        document.getElementById('login-nav').classList.add('d-none');
        document.getElementById('signup-nav').classList.add('d-none');
        document.getElementById('dashboard-nav').classList.remove('d-none');
        document.getElementById('logout-nav').classList.remove('d-none');
    } else {
        document.getElementById('login-nav').classList.remove('d-none');
        document.getElementById('signup-nav').classList.remove('d-none');
        document.getElementById('dashboard-nav').classList.add('d-none');
        document.getElementById('logout-nav').classList.add('d-none');
    }
}

// Logout function
function logout() {
    localStorage.removeItem('token');
    window.location.href = '/landing';
}

// Fetch user data
async function fetchUserData() {
    try {
        const response = await fetch('/api/users/me', {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });
        
        if (response.ok) {
            const userData = await response.json();
            
            // Display user data
            if (document.getElementById('username-display')) {
                document.getElementById('username-display').textContent = userData.username;
            }
            if (document.getElementById('email-display')) {
                document.getElementById('email-display').textContent = userData.email;
            }
            if (document.getElementById('created-at-display')) {
                document.getElementById('created-at-display').textContent = new Date(userData.created_at).toLocaleString();
            }
            
            // Pre-fill update form
            if (document.getElementById('email')) {
                document.getElementById('email').value = userData.email;
            }
        } else {
            // If unauthorized, redirect to login
            if (response.status === 401) {
                localStorage.removeItem('token');
                window.location.href = '/login';
            }
        }
    } catch (error) {
        console.error('Error fetching user data:', error);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Check authentication status
    checkAuth();
    
    // If on dashboard page, fetch user data
    if (window.location.pathname === '/dashboard') {
        const token = localStorage.getItem('token');
        if (!token) {
            window.location.href = '/login';
            return;
        }
        
        fetchUserData();
    }
}); 