import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
import random
import time

class MirrorCloneFX:
    def __init__(self):
        # Initialize MediaPipe
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarkerOptions = vision.HandLandmarkerOptions
        HandLandmarker = vision.HandLandmarker
        
        # Create hand landmarker
        base_options = BaseOptions(model_asset_path='hand_landmarker.task')
        options = HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1
        )
        self.hands = HandLandmarker.create_from_options(options)
        
        # Visual modes
        self.modes = {
            0: "Dots",
            1: "Lines", 
            2: "ASCII",
            3: "Particles"
        }
        self.current_mode = 0
        
        # Particles system
        self.particles = []
        self.max_particles = 200
        
        # ASCII characters for ASCII mode (from dense to sparse)
        self.ascii_chars = "█▉▊▋▌▍▎▏ "
        
        # Window dimensions
        self.window_width = 1280
        self.window_height = 720
        self.half_width = self.window_width // 2
        
    def detect_hand_gesture(self, landmarks):
        """Detect hand gestures from landmarks"""
        if not landmarks:
            return None
            
        # Get landmark positions
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]
        
        # Check if fingers are extended
        fingers_up = []
        
        # Thumb (different logic - compare x coordinates)
        if thumb_tip.x > thumb_ip.x:
            fingers_up.append(1)
        else:
            fingers_up.append(0)
            
        # Other fingers (compare y coordinates)
        finger_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
        finger_pips = [index_pip, middle_pip, ring_pip, pinky_pip]
        
        for tip, pip in zip(finger_tips, finger_pips):
            if tip.y < pip.y:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        
        # Gesture recognition
        # One finger (index) - Lines mode
        if fingers_up == [0, 1, 0, 0, 0]:
            return 1
        
        # Two fingers (index + middle) - Dots mode  
        elif fingers_up == [0, 1, 1, 0, 0]:
            return 0
            
        # Thumb + pinky - ASCII mode
        elif fingers_up == [1, 0, 0, 0, 1]:
            return 2
            
        # Open palm (all fingers) - Particles mode
        elif sum(fingers_up) >= 4:
            return 3
            
        return None
    
    def draw_hand_landmarks(self, frame, landmarks):
        """Draw hand landmarks manually"""
        if not landmarks:
            return
            
        # Hand connections (same as MediaPipe's HAND_CONNECTIONS)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
            (5, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
            (9, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
            (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (0, 17)  # Palm
        ]
        
        h, w = frame.shape[:2]
        
        # Draw connections
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        # Draw landmarks
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
    
    def create_dots_effect(self, frame):
        """Create stippled dot rendering"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = np.zeros_like(frame)
        
        height, width = gray.shape
        dot_spacing = 12
        
        for y in range(0, height, dot_spacing):
            for x in range(0, width, dot_spacing):
                if y < height and x < width:
                    intensity = gray[y, x]
                    if intensity > 60:
                        radius = int((intensity / 255) * 6) + 1
                        color = frame[y, x].astype(int)
                        color = np.clip(color * 1.2, 0, 255).astype(int)
                        cv2.circle(result, (x, y), radius, color.tolist(), -1)
        
        return result
    
    def create_lines_effect(self, frame):
        """Create edge outline rendering"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 80)
        result = np.zeros_like(frame)
        
        edge_points = np.where(edges > 0)
        for y, x in zip(edge_points[0], edge_points[1]):
            original_color = frame[y, x].astype(int)
            enhanced_color = np.clip(original_color * 1.5, 0, 255).astype(int)
            result[y, x] = enhanced_color
        
        kernel = np.ones((3,3), np.uint8)
        result = cv2.dilate(result, kernel, iterations=1)
        
        return result
    
    def create_ascii_effect(self, frame):
        """Create ASCII art rendering"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = np.zeros_like(frame)
        
        height, width = gray.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        char_width = 16
        char_height = 20
        
        for y in range(0, height, char_height):
            for x in range(0, width, char_width):
                if y + char_height < height and x + char_width < width:
                    region = gray[y:y+char_height, x:x+char_width]
                    avg_intensity = np.mean(region)
                    
                    if avg_intensity > 30:
                        char_index = int(((255 - avg_intensity) / 255) * (len(self.ascii_chars) - 1))
                        char = self.ascii_chars[char_index]
                        
                        if avg_intensity > 150:
                            color = [255, 255, 255]
                        elif avg_intensity > 100:
                            color = [0, 255, 0]
                        else:
                            color = [0, 255, 255]
                        
                        cv2.putText(result, char, (x + 2, y + char_height - 4), 
                                   font, font_scale, color, thickness)
        
        return result
    
    def update_particles(self, frame, landmarks):
        """Update particle system"""
        if landmarks:
            for landmark in landmarks[::2]:
                if len(self.particles) < self.max_particles:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    
                    particle = {
                        'x': x + random.randint(-20, 20),
                        'y': y + random.randint(-20, 20),
                        'vx': random.uniform(-2, 2),
                        'vy': random.uniform(-2, 2),
                        'life': 60,
                        'color': [random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)]
                    }
                    self.particles.append(particle)
        
        self.particles = [p for p in self.particles if p['life'] > 0]
        
        for particle in self.particles:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['life'] -= 1
            particle['vy'] += 0.1
    
    def create_particles_effect(self, frame, landmarks):
        """Create particle effect"""
        result = np.zeros_like(frame)
        self.update_particles(frame, landmarks)
        
        for particle in self.particles:
            if 0 <= particle['x'] < frame.shape[1] and 0 <= particle['y'] < frame.shape[0]:
                alpha = particle['life'] / 60.0
                radius = max(1, int(alpha * 4))
                color = [int(c * alpha) for c in particle['color']]
                cv2.circle(result, (int(particle['x']), int(particle['y'])), 
                          radius, color, -1)
        
        return result
    
    def process_frame(self, frame, landmarks):
        """Process frame based on current mode"""
        if self.current_mode == 0:
            return self.create_dots_effect(frame)
        elif self.current_mode == 1:
            return self.create_lines_effect(frame)
        elif self.current_mode == 2:
            return self.create_ascii_effect(frame)
        elif self.current_mode == 3:
            return self.create_particles_effect(frame, landmarks)
        else:
            return frame
    
    def run(self):
        """Main application loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("MirrorCloneFX started!")
        print("Hand gestures:")
        print("✌️  Two fingers → Dots mode")
        print("☝️  One finger → Lines mode") 
        print("🤙 Thumb + pinky → ASCII mode")
        print("✋ Open palm → Particles mode")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = self.hands.detect(mp_image)
            landmarks = None
            
            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    landmarks = hand_landmarks
                    
                    gesture = self.detect_hand_gesture(landmarks)
                    if gesture is not None:
                        self.current_mode = gesture
                    
                    self.draw_hand_landmarks(frame, landmarks)
            
            frame_resized = cv2.resize(frame, (self.half_width, self.window_height))
            stylized = self.process_frame(frame, landmarks)
            stylized_resized = cv2.resize(stylized, (self.half_width, self.window_height))
            split_screen = np.hstack([frame_resized, stylized_resized])
            
            mode_text = f"Mode: {self.modes[self.current_mode]}"
            cv2.putText(split_screen, mode_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.line(split_screen, (self.half_width, 0), 
                    (self.half_width, self.window_height), (255, 255, 255), 2)
            
            cv2.putText(split_screen, "Original", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(split_screen, "Clone", (self.half_width + 10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('MirrorCloneFX', split_screen)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    app = MirrorCloneFX()
    app.run()

if __name__ == "__main__":
    main()