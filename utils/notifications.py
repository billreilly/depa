# utils/notifications.py
import json
from pywebpush import webpush, WebPushException

VAPID_PUBLIC_KEY = "BG6cA-uSuEEKynpTzohODifm0-J5S6yNoaXgDNk9SZFmN4s9PUcsmuxcA2Q735fDDHNo9Se7PfbVcqRmPXeVG8U"
VAPID_PRIVATE_KEY = "JbRY-2mSKUo-QB62zd5ZWAs_199-JFEgvluX3DiqFSc"
VAPID_CLAIMS = {"sub": "mailto:your_email@example.com"}

def save_subscription(subscription_info):
    # Save subscription to the database or file for later use
    # This is where you would integrate with your database
    pass


def send_notification(subscription_info, message):
    vapid_claims = {
        "sub": "mailto:your_email@example.com",
        "aud": subscription_info['endpoint'].split("/")[2]  # Extract audience from endpoint
    }

    try:
        webpush(
            subscription_info=subscription_info,
            data=json.dumps(message),
            vapid_private_key=VAPID_PRIVATE_KEY,
            vapid_claims=vapid_claims
        )
    except WebPushException as ex:
        print(f"Error sending notification: {ex}")


def subscribe_user(subscription_info):
    save_subscription(subscription_info)
