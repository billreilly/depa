from gpiozero import AngularServo
from time import sleep

# Adjust these pulse width values to experiment with different speeds
servo = AngularServo(18, min_pulse_width=0.0008, max_pulse_width=0.0025)

while True:
    servo.angle = 90
    sleep(2)
    servo.angle = 0
    sleep(2)
    servo.angle = -90
    sleep(2)
