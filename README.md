<h2>Output: </h2>
![image](https://github.com/user-attachments/assets/05eff209-328a-49f7-91e2-2d07943461e1)
![image](https://github.com/user-attachments/assets/cf619ec4-943d-4686-b82d-eba05f78fe17)
![image](https://github.com/user-attachments/assets/2148356b-4759-472d-b89a-d6a8a52dd48f)

<h2>WORKING OF PROJECT:</h2>
1)	EEG Data Capture: EEG data is fetched for different lobes and formatted into compatible inputs.
2)	Data Processing:
  •	Software Implementation:
        Machine learning algorithms process data for classification of the state of eye as open or closed alongwith the status of intoxication. This involves scaling the input data, training a Random Forest Classifier, and making predictions based for the new inputs.
  •	Hardware Implementation:
        The Raspberry Pi Pico W computes the results and enables the GPIO pins to control LEDs.
3)	Prediction and Visualization:
   a.	The eye state prediction triggers the corresponding LED that are Closed Eye LED and Open Eye LED.

<h2>Steps</h2>
1) Firstly run the ML code on VS Code Editor.
2) Then connect you Raspberry Pi Pico W with your laptop either by wifi or by B-Type Lan cable.
3) Then in Thonny code editor write the code provided below.
   import ujson as json
    import math
    import time
    import machine  
    
    led_closed_eye = machine.Pin(15, machine.Pin.OUT)  # LED for Closed Eye (GPIO 15)
    led_open_eye = machine.Pin(16, machine.Pin.OUT)    # LED for Open Eye (GPIO 16)
    
    with open("averages.json", "r") as f:
        averages = json.load(f)
    
    class_0_avg = averages["class_0_avg"]
    class_1_avg = averages["class_1_avg"]
    
    # Euclidean distance calculation
    def euclidean_distance(a, b):
        return math.sqrt(sum((a[i] - b[i])**2 for i in range(len(a))))
    
    # Predict mental state based on input EEG data
    def predict_mental_state(eeg_data):
        dist_to_class_0 = euclidean_distance(eeg_data, class_0_avg)
        dist_to_class_1 = euclidean_distance(eeg_data, class_1_avg)
        return "Closed Eye" if dist_to_class_0 < dist_to_class_1 else "Open Eye"
    
    def blink_led(state):
        if state == "Closed Eye":
            led_closed_eye.on()  # Turn on LED for Closed Eye
            led_open_eye.off()   # Turn off LED for Open Eye
            print("Closed Eye - LED1 ON")
        else:
            led_open_eye.on()    # Turn on LED for Open Eye
            led_closed_eye.off() # Turn off LED for Closed Eye
            print("Open Eye - LED2 ON")
        
        time.sleep(1)
        led_closed_eye.off()
        led_open_eye.off()
        time.sleep(1)
    
    def main():
        print("Pico W EEG Predictor is running...")
        # data for testing
        eeg_data = [4294.36,4008.21,4264.1,4105.64,4337.44,4623.59,4062.56,4615.9,4197.95,4241.03,4164.62,4269.23,4603.08,4357.95]  # Replace with actual values
    
        state = predict_mental_state(eeg_data)
        print(f"Predicted Mental State: {state}")
    
        blink_led(state)
    
    if __name__ == "__main__":
        main()
4) Save the code and run on Thonny and you'll get to see the result.
