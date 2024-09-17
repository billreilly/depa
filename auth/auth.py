from flask import Blueprint, render_template, redirect, url_for, request, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User  # Import `db` and `User` from `models.py`
import requests
import base64

auth_bp = Blueprint('auth', __name__)

# Thai Bulk SMS API credentials
API_KEY = 'HjN1MEJnJ60Yb9LKsBiZDpO8a2AX4U'
API_SECRET = 'Bf6hT92BEsWaLXmVk0IH7dev2VabfS'
SMS_API_URL = 'https://api-v2.thaibulksms.com/sms'

# Utility function to send SMS notification
def send_login_notification(phone_number, username):
    auth_string = f'{API_KEY}:{API_SECRET}'
    auth_bytes = auth_string.encode('utf-8')
    auth_base64 = base64.b64encode(auth_bytes).decode('utf-8')

    message = f'User {username} has just logged in.'

    payload = {
        'msisdn': phone_number,
        'message': message,
        'sender': 'Demo',  # Update with a valid sender name
        'force': 'standard'
    }

    headers = {
        'Authorization': f'Basic {auth_base64}',
        'Content-Type': 'application/json'
    }

    response = requests.post(SMS_API_URL, json=payload, headers=headers)

    if response.status_code == 200:
        return True
    else:
        return False

# Utility function to send SMS
def send_fall_notification(phone_number):
    auth_string = f'{API_KEY}:{API_SECRET}'
    auth_bytes = auth_string.encode('utf-8')
    auth_base64 = base64.b64encode(auth_bytes).decode('utf-8')

    message = 'Fall detected for 5 seconds. Please check!'

    payload = {
        'msisdn': phone_number,
        'message': message,
        'sender': 'Demo',  # Update with a valid sender name
        'force': 'standard'
    }

    headers = {
        'Authorization': f'Basic {auth_base64}',
        'Content-Type': 'application/json'
    }

    response = requests.post(SMS_API_URL, json=payload, headers=headers)

    if response.status_code == 200:
        print("SMS sent successfully")
    else:
        print("Failed to send SMS", response.text)


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            session['username'] = username  # Store the username in session

            # Send SMS notification upon successful login
            phone_number = '095296942'  # Hardcoded phone number for demo, replace with user's actual phone number
            if send_login_notification(phone_number, username):
                flash('Login successful. A notification has been sent.')
            else:
                flash('Login successful, but failed to send login notification.')

            return redirect(url_for('index'))  # Redirect to the index page after login
        else:
            flash('Invalid credentials, please try again.')
    return render_template('login.html')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('Username already exists!')
        else:
            hashed_password = generate_password_hash(password, method='sha256')
            new_user = User(username=username, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            session['username'] = username
            return redirect(url_for('index'))
    return render_template('register.html')

@auth_bp.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('auth.login'))
