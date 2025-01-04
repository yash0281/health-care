const express = require('express');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = 3000;

// Set up middleware to serve static files (like CSS, JS)
app.use(express.static(path.join(__dirname, 'public')));

// Set up middleware to parse JSON and urlencoded data
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Folder to store user credentials (JSON file)
const usersFile = path.join(__dirname, 'users.json');

// Helper function to read users from the file
function readUsers() {
    if (!fs.existsSync(usersFile)) return [];
    const data = fs.readFileSync(usersFile);
    return JSON.parse(data);
}

// Helper function to save users to the file
function saveUsers(users) {
    fs.writeFileSync(usersFile, JSON.stringify(users, null, 2));
}

// Route to serve login page from templates folder
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'templates', 'login.html'));
});

// Route to serve register page from templates folder
app.get('/register', (req, res) => {
    res.sendFile(path.join(__dirname, 'templates', 'register.html'));
});

// Register endpoint
app.post('/register', (req, res) => {
    const { username, email, password } = req.body;

    if (!username || !email || !password) {
        return res.json({ success: false, message: 'All fields are required.' });
    }

    const users = readUsers();
    const userExists = users.some(user => user.username === username || user.email === email);

    if (userExists) {
        return res.json({ success: false, message: 'Username or Email already taken.' });
    }

    // Add new user
    const newUser = { username, email, password }; // In a real app, you should hash passwords!
    users.push(newUser);
    saveUsers(users);

    res.json({ success: true, message: 'Registration successful.' });
});

// Login endpoint
app.post('/login', (req, res) => {
    const { username, password } = req.body;

    if (!username || !password) {
        return res.json({ success: false, message: 'Username and password are required.' });
    }

    const users = readUsers();
    const user = users.find(user => user.username === username && user.password === password);

    if (user) {
        res.json({ success: true, message: 'Login successful.' });
    } else {
        res.json({ success: false, message: 'Invalid username or password.' });
    }
});

// Start server
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
