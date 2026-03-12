import cv2
import cv2.aruco as aruco
import numpy as np
import time
import random

ARUCO_IDS = [0, 2]
PROXIMITY_THRESHOLD = 300 
SPEED_THRESHOLD = 15       
STATUS_UPDATE_INTERVAL = 0.1
SCALING_FACTOR = 10.3

# Speed / servo limits
MIN_SPEED  = 0.45
MAX_SPEED  = 0.60
MAX_SERVO  = 0.50

tracker = {
    0: {'position': None, 'last_time': None, 'status': None, 'speed': 0.0, 'last_sent': {}},
    2: {'position': None, 'last_time': None, 'status': None,  'speed': 40.0,'last_sent': {}}
}

detected = {}


def estimate_speed(curr_pos, prev_pos, dt):
    if prev_pos is None or dt == 0:
        return 0.0
    dist = np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))
    return float(dist / (dt * SCALING_FACTOR))

def run(frame: np.ndarray, car_id: int):
    """
    Process a pre-captured frame and return (motor_speed, servo_angle, modified_frame).

    Parameters
    ----------
    frame   : BGR image already read by the camera producer thread
    car_id  : integer marker ID for this car
    """
    # Validate incoming frame
    # if frame is None or frame.size == 0:
    #     print(f"[Car {car_id}] Empty frame received.")
    #     return 0.0, 0.0, np.zeros((720, 1280, 3), dtype=np.uint8)

    # Work on a copy so the producer's shared frame is never mutated
    car_frame = frame.copy()

    current_time = time.time()
    gray    = cv2.cvtColor(car_frame, cv2.COLOR_BGR2GRAY)
    adict   = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    params  = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(adict, params)
    corners, ids, _ = detector.detectMarkers(gray)

    detected.clear()

    servo, motor = 0.0, 0.0
    if ids is not None:
        ids = ids.flatten()
        for i, marker_id in enumerate(ids):
            marker_id = int(marker_id)
            if marker_id not in ARUCO_IDS:
                continue

            corner = corners[i][0]
            center = tuple(np.mean(corner, axis=0).astype(int))
            detected[marker_id] = center


            # Draw marker and ID
            cv2.circle(car_frame, center, 6, (255, 0, 0), -1)
            cv2.putText(car_frame, f"ID {marker_id}", (center[0]+10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Tracking and status update
            last_pos = tracker[marker_id]['position']
            last_time = tracker[marker_id]['last_time']
            status = tracker[marker_id]['status']


            if last_time is None or (current_time - last_time >= STATUS_UPDATE_INTERVAL):
                dt = current_time - last_time if last_time else 0
                speed = estimate_speed(center, last_pos, dt)
                new_status = "moving" if speed > SPEED_THRESHOLD else "stopped"
                
                tracker[marker_id]['speed'] = speed

                if new_status != status:
                    print(f" Car {marker_id}: {new_status.upper()} (Speed: {speed:.2f})")
                    tracker[marker_id]['status'] = new_status

                tracker[marker_id]['position'] = center
                tracker[marker_id]['last_time'] = current_time

            # Show status on car_frame
            display_status = tracker[marker_id]['status'] or "unknown"
            cv2.putText(car_frame, f"{display_status}", (center[0], center[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Proximity check
    if all(i in detected for i in ARUCO_IDS):
        dist = np.linalg.norm(np.array(detected[0]) - np.array(detected[2]))
        cv2.line(car_frame, detected[0], detected[2], (255, 0, 255), 2)
        cv2.putText(car_frame, f"Distance: {int(dist)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        if dist < PROXIMITY_THRESHOLD:
            for marker_id in ARUCO_IDS:
                last_sent = tracker[marker_id]['last_sent']
                if not last_sent.get("proximity_alert"):
                    # send_command(marker_id, "proximity_alert")
                    print(f"Car {car_id}: Proximity alert received!")
                    last_sent["proximity_alert"] = True


            # faster car change track
            speed_0 = tracker[0].get('speed')
            speed_1 = tracker[2].get('speed')

            if speed_0 > speed_1 and tracker[0]['status'] == "moving":
                if not tracker[0]['last_sent'].get("0.25"):
                    # send_command(1, "50")
                    if car_id == 0:
                        servo, motor = random.choice([MAX_SERVO / 2, -MAX_SERVO / 2]), MIN_SPEED
                        tracker[0]['last_sent']["0.25"] = True
                        print(f"0 change track with turning angle {servo} and motor speed {motor}")
            elif speed_1 > speed_0 and tracker[2]['status'] == "moving":
                if not tracker[2]['last_sent'].get("0.25"):
                    # send_command(3, "50")
                    if car_id == 2:
                        servo, motor = random.choice([MAX_SERVO / 2, -MAX_SERVO / 2]), MIN_SPEED
                        tracker[2]['last_sent']["0.25"] = True
                        print(f"2 change track with turning angle {servo} and motor speed {motor}")
        else:
            for marker_id in ARUCO_IDS:
                last_sent = tracker[marker_id]['last_sent']
                if last_sent.get("proximity_alert") or last_sent.get("0.25"):
                    print(f"Reset Proximity cleared for Car {marker_id}")
                    last_sent.pop("proximity_alert", None)
                    last_sent.pop("0.25", None)
                
                if not last_sent.get("0.0"):
                    # send_command(marker_id, "90")
                    if car_id == 2:
                        servo, motor = 0.0, MAX_SPEED
                        last_sent["0.0"] = True
                        print(f"2 continue with turning angle {servo} and motor speed {motor}")
                else:
                    last_sent.pop("0.0", None)


    # # Display - Just for testing purposes
    # cv2.imshow("Tracking", car_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     sys.exit("\nQuitting the car_frame!")
    return round(servo, 2), round(motor, 2), car_frame


# # Just for testing purposes
# def init_camera(cam_index=0):
#     if os.name == "nt":
#         cap = cv2.VideoCapture(cam_index)
#     elif os.name == "posix":
#         cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)

#     if not cap.isOpened():
#         sys.exit("Camera not found!\n")

#     cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

#     print("Camera opened successfully!\n")
#     return cap

# # Just for testing purposes
# if __name__ == '__main__':
#     capture = init_camera(1)
#     car_idx = 1

#     while True:
#         ret, frame = capture.read()
#         if ret and frame is not None:
#             motor, servo, vis = run(frame, car_idx)
#             cv2.imshow("Tracking", vis)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     capture.release()
#     cv2.destroyAllWindows()