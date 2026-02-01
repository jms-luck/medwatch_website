const API_BASE_URL = 'http://localhost:8000';

// Removed Firebase authentication - application now works without login

function enterApplication() {
    showMessage('Welcome to MedWatch AI!');
    setTimeout(() => {
<<<<<<< HEAD
        window.location.href = 'home.html';
=======
        window.location.href = '/home';
>>>>>>> 932bf0f (v3)
    }, 1000);
}

function showMessage(msg) {
    const messageElement = document.getElementById('message');
    if (messageElement) {
        messageElement.textContent = msg;
    }
}

function logout() {
    try {
        localStorage.removeItem('mwUser');
    } catch (err) {
        console.warn('Unable to clear stored user profile', err);
    }
<<<<<<< HEAD
    window.location.href = 'login.html';
=======
    window.location.href = '/';
>>>>>>> 932bf0f (v3)
}

function storeUser(user) {
    try {
        localStorage.setItem('mwUser', JSON.stringify(user));
    } catch (err) {
        console.warn('Unable to store user profile', err);
    }
}

function loadStoredUser() {
    try {
        const raw = localStorage.getItem('mwUser');
        return raw ? JSON.parse(raw) : null;
    } catch (err) {
        console.warn('Unable to read stored user profile', err);
        return null;
    }
}

// Handle registration form submission
document.addEventListener('DOMContentLoaded', () => {
    const registerForm = document.getElementById('registerForm');
    const loginForm = document.getElementById('loginForm');
    populateProfile();
    const logoutBtns = document.querySelectorAll('.logout-btn');
    logoutBtns.forEach(btn => btn.addEventListener('click', logout));

    if (registerForm) {
        registerForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const formData = new FormData(registerForm);
        const payload = Object.fromEntries(formData.entries());

        try {
            const response = await fetch(`${API_BASE_URL}/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.detail || 'Registration failed');
            }

            showMessage('Account created successfully!');
            registerForm.reset();
            setTimeout(() => {
                window.location.href = 'login.html';
            }, 700);
        } catch (error) {
            showMessage(error.message);
        }
    });
    }

    if (loginForm) {
        loginForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            const formData = new FormData(loginForm);
            const payload = Object.fromEntries(formData.entries());

            try {
                const response = await fetch(`${API_BASE_URL}/login`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });

                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.detail || 'Login failed');
                }

                storeUser(data.user);
                showMessage('Login successful');
                // redirect to home page after successful login
                setTimeout(() => {
<<<<<<< HEAD
                    window.location.href = 'home.html';
=======
                    window.location.href = '/home';
>>>>>>> 932bf0f (v3)
                }, 500);
            } catch (error) {
                showMessage(error.message);
            }
        });
    }
});

function populateProfile() {
    const user = loadStoredUser();
    const nameEl = document.getElementById('profile-name');
    const mailEl = document.getElementById('profile-mail');
    const userEl = document.getElementById('profile-username');
    const ageEl = document.getElementById('profile-age');
    const phoneEl = document.getElementById('profile-phone');
    const messageEl = document.getElementById('profile-message');

    if (!nameEl || !user) {
        if (messageEl) {
            messageEl.textContent = 'No profile data found. Please log in first.';
        }
        return;
    }

    nameEl.textContent = user.name || 'Not set';
    mailEl.textContent = user.mail_id || 'Not set';
    userEl.textContent = user.username || 'Not set';
    ageEl.textContent = user.age != null ? user.age : 'Not set';
    phoneEl.textContent = user.phone_number || 'Not set';
    if (messageEl) {
        messageEl.textContent = '';
    }
}

// Notification code
if ('Notification' in window) {
    if (Notification.permission === 'granted') {
        new Notification('Welcome to MEdWATCH AI');
    } else if (Notification.permission !== 'denied') {
        Notification.requestPermission().then(permission => {
            if (permission === 'granted') {
                new Notification('Welcome to MEdWATCH AI');
            }
        });
    }
}