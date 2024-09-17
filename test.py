from flask import Flask, request, jsonify
import json

app = Flask(__name__)

# Route to handle push subscription
@app.route('/subscribe', methods=['POST'])
def subscribe():
    # Get subscription info from request
    subscription_info = request.get_json()

    # Log the subscription info to verify structure
    print(subscription_info)

    # Here you can save the subscription_info to a file or database
    # For now, we will just send a response back
    return jsonify({"message": "Subscription received"}), 201

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
