import lgpio
from time import sleep

# Setup GPIO for servo control
try:
    gpio = lgpio.gpiochip_open(0)  # Open GPIO chip 0
    lgpio.gpio_claim_output(gpio, 18)  # Claim GPIO pin 18 for output
except Exception as e:
    print(f"Failed to initialize GPIO: {e}")
    gpio = None  # Prevent further errors if GPIO setup fails

# Function to set PWM duty cycle for a specified time duration
def set_pwm_duty_cycle(duty, duration):
    if gpio is not None:
        print(f"Setting PWM duty cycle: {duty}% for {duration} seconds")
        lgpio.tx_pwm(gpio, 18, 50, duty)  # 50Hz frequency, with specified duty cycle
        sleep(duration)
    else:
        print("GPIO not initialized, unable to set PWM duty cycle.")

# Function to move the servo clockwise once with a pause
def move_servo_clockwise():
    print("Moving servo clockwise")
    set_pwm_duty_cycle(7.3, 0.3)  # Adjust the duty cycle and duration as needed
    sleep(1)  # Pause for 1 second after movement to reduce jitter
    set_pwm_duty_cycle(7.5, 0.5)  # Return to neutral
    sleep(1)  # Pause for 1 second to stabilize

# Function to move the servo counterclockwise once with a pause
def move_servo_counterclockwise():
    print("Moving servo counterclockwise")
    set_pwm_duty_cycle(7.7, 0.3)  # Adjust the duty cycle and duration as needed
    sleep(1)  # Pause for 1 second after movement to reduce jitter
    set_pwm_duty_cycle(7.5, 0.5)  # Return to neutral
    sleep(1)  # Pause for 1 second to stabilize

# Clean up GPIO resources
def cleanup_servo():
    if gpio is not None:
        lgpio.gpiochip_close(gpio)
        print("GPIO resources cleaned up.")
    else:
        print("GPIO was not initialized, no cleanup needed.")

def stop_servo():
    print("Stopping the servo")
    lgpio.tx_pwm(gpio, 18, 50, 0)  # Set the duty cycle to 0 or stop PWM


# Function to move the servo clockwise for a custom duration
def custom_move_servo_clockwise():
    print("Custom clockwise movement")
    set_pwm_duty_cycle(7.3, 0.3)  # Adjust for clockwise movement
    sleep(2)  # Pause after movement to prevent jitter
    stop_servo()

# Function to move the servo counterclockwise for a custom duration
def custom_move_servo_counterclockwise():
    print("Custom counterclockwise movement")
    set_pwm_duty_cycle(7.7, 0.3)  # Adjust for counterclockwise movement
    sleep(2)  # Pause after movement to prevent jitter
    stop_servo()

