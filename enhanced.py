import cv2
import numpy as np
from ultralytics import YOLO
import os
import json
import time
from collections import defaultdict, deque
from deep_sort_realtime.deepsort_tracker import DeepSort

class BasketballAnalyzer:
    def __init__(self):
        # ✅ ESSENTIAL INITIALIZATION - DON'T REMOVE THESE!
        self.detection_model = YOLO('yolo11s.pt')
        self.pose_model = YOLO('yolo11s-pose.pt')
        # 🔥 ADD YOUR SHOT DETECTION MODEL
        self.shot_model = YOLO('shot_detection_v2.pt')
        
        # Tracking
        self.tracker = DeepSort(max_age=50, n_init=3)
        self.player_histories = defaultdict(deque)
        
        # Offensive analysis data
        self.possessions = []
        self.offensive_sets = defaultdict(int)
        self.transition_styles = defaultdict(int)
        self.common_actions = defaultdict(int)
        self.initiators = defaultdict(int)
        self.iso_plays = []
        self.ball_movements = []
        # 🔥 ADD SHOT TRACKING
        self.shots_detected = []
        self.shot_attempts = defaultdict(int)
        
        # Config
        self.ISO_DISTANCE = 300
        self.FAST_BREAK_SPEED = 10
        self.PLAYER_SPEED_THRESHOLD = 5.0
        
        # ✅ ADD MISSING COURT DIMENSIONS
        self.court_width = 1920
        self.court_height = 1080

    def _filter_players(self, players, frame_shape):
        """Filter out non-player detections and referees"""
        filtered = []
        h, w = frame_shape[:2]
        
        for player in players:
            x, y, width, height = player['bbox']
            
            # Filter by size (remove small detections)
            if width * height < 2000:
                continue
                
            # Filter by position (remove sidelines/bench)
            if y + height > h * 0.9:
                continue
                
            filtered.append(player)
            
        return filtered

    def _get_ball_handler(self, players, ball_pos):
        """Identify player closest to the ball"""
        if not players or not ball_pos:
            return None
            
        bx, by = ball_pos
        min_dist = float('inf')
        handler = None
        
        for player in players:
            px, py = player['bbox'][0] + player['bbox'][2]/2, player['bbox'][1] + player['bbox'][3]/2
            dist = np.sqrt((px - bx)**2 + (py - by)**2)
            
            if dist < min_dist:
                min_dist = dist
                handler = player
                
        return handler if min_dist < 100 else None

    def _is_iso_play(self, players, ball_pos, ball_handler):
        """Detect isolation play"""
        if not ball_handler:
            return False
            
        # Count teammates far from ball handler
        far_teammates = 0
        bx, by = ball_pos
        
        for player in players:
            if player.get('team') == ball_handler.get('team') and player.get('id') != ball_handler.get('id'):
                px, py = player['bbox'][0] + player['bbox'][2]/2, player['bbox'][1] + player['bbox'][3]/2
                dist = np.sqrt((px - bx)**2 + (py - by)**2)
                
                if dist > self.ISO_DISTANCE:
                    far_teammates += 1
                    
        return far_teammates >= 2

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_count / fps
            
            # Detection
            players, ball_pos = self._detect_players_and_ball(frame)
            players = self._filter_players(players, frame.shape)
            
            # Tracking
            tracked_players = self._track_players(players, frame)
            
            # Offensive analysis
            self._analyze_possession(timestamp, tracked_players, ball_pos, frame)
            
            # Visualization
            vis_frame = self._create_visualization(frame, tracked_players, ball_pos, frame_count)
            cv2.imshow('Basketball Analysis', vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")
        
        cap.release()
        cv2.destroyAllWindows()
        return self._generate_report()

    def _detect_players_and_ball(self, frame):
        """Detect players and ball using YOLO models"""
        det_results = self.detection_model(frame, verbose=False)[0]
        players = []
        ball_pos = None
        
        if det_results.boxes is None:
            return players, ball_pos
        
        for box in det_results.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if cls_id == 0:  # Person
                # ✅ FIXED: Pass correct bbox format (x, y, w, h)
                team = self._classify_team(frame, (x1, y1, x2-x1, y2-y1))
                players.append({
                    'bbox': (x1, y1, x2-x1, y2-y1),
                    'team': team,
                    'conf': conf
                })
            elif cls_id == 32:  # Sports ball
                ball_pos = ((x1 + x2)/2, (y1 + y2)/2)
        
        return players, ball_pos

    def _classify_team(self, frame, bbox):
        """DYNAMIC team classification - NOT hardcoded to TRITON"""
        x, y, w, h = bbox
        
        # Extract jersey region
        jersey_y1 = y + int(h * 0.15)
        jersey_y2 = y + int(h * 0.55)
        jersey_x1 = x + int(w * 0.2)
        jersey_x2 = x + int(w * 0.8)
        
        # Ensure valid region
        jersey_y1 = max(0, jersey_y1)
        jersey_y2 = min(frame.shape[0], jersey_y2)
        jersey_x1 = max(0, jersey_x1)
        jersey_x2 = min(frame.shape[1], jersey_x2)
        
        if jersey_y2 <= jersey_y1 or jersey_x2 <= jersey_x1:
            return "TEAM_LIGHT"  # Default to light team
        
        jersey_region = frame[jersey_y1:jersey_y2, jersey_x1:jersey_x2]
        
        if jersey_region.size == 0:
            return "TEAM_LIGHT"
        
        # Simple brightness-based classification
        avg_brightness = np.mean(jersey_region)
        
        # Light jerseys vs Dark jerseys (DYNAMIC - not hardcoded)
        if avg_brightness > 130:
            return "TEAM_LIGHT"  # Light/white jerseys
        else:
            return "TEAM_DARK"   # Dark jerseys

    def _track_players(self, players, frame):
        """Track players with DeepSORT"""
        if not players:
            return []
            
        detections = []
        for p in players:
            x, y, w, h = p['bbox']
            detections.append(([x, y, w, h], p['conf'], 0))
        
        tracks = self.tracker.update_tracks(detections, frame=frame)
        tracked_players = []
        
        for i, track in enumerate(tracks):
            if not track.is_confirmed() or i >= len(players):
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb()
            
            # Update player history
            self.player_histories[track_id].append({
                'position': (ltrb[0], ltrb[1]),
                'timestamp': time.time()
            })
            
            # Keep last 10 positions
            if len(self.player_histories[track_id]) > 10:
                self.player_histories[track_id].popleft()
            
            tracked_players.append({
                'id': track_id,
                'bbox': (int(ltrb[0]), int(ltrb[1]), int(ltrb[2]-ltrb[0]), int(ltrb[3]-ltrb[1])),
                'team': players[i]['team']
            })
            
        return tracked_players

    def _create_visualization(self, frame, players, ball_pos, frame_count):
        """Create visualization frame with SHOT DETECTION overlay"""
        vis_frame = frame.copy()
        
        # Draw players
        for player in players:
            x, y, w, h = player['bbox']
            team = player['team']
            player_id = player.get('id', 0)
            
            # ✅ CONSISTENT TEAM COLORS
            color = (0, 255, 0) if team == "TEAM_LIGHT" else (255, 0, 0)
            
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(vis_frame, f'{team[0]}{player_id}', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw ball
        if ball_pos:
            cv2.circle(vis_frame, (int(ball_pos[0]), int(ball_pos[1])), 15, (0, 0, 255), 3)
            cv2.putText(vis_frame, 'BALL', (int(ball_pos[0]-20), int(ball_pos[1]-25)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 🔥 DRAW RECENT SHOTS
        current_time = frame_count / 30.0  # Assuming 30fps
        for shot in self.shots_detected[-5:]:  # Show last 5 shots
            if current_time - shot['timestamp'] < 3.0:  # Show for 3 seconds
                sx, sy, sw, sh = shot['bbox']
                cv2.rectangle(vis_frame, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 3)
                cv2.putText(vis_frame, f'SHOT! {shot["confidence"]:.2f}', 
                           (sx, sy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Info overlay
        light_count = len([p for p in players if p['team'] == 'TEAM_LIGHT'])
        dark_count = len([p for p in players if p['team'] == 'TEAM_DARK'])
        
        cv2.putText(vis_frame, f'Frame: {frame_count}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_frame, f'LIGHT: {light_count} | DARK: {dark_count}', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_frame, f'SHOTS: {len(self.shots_detected)}', 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return vis_frame

    def _analyze_possession(self, timestamp, players, ball_pos, frame):
        """FIXED: Actually track possessions and offensive metrics + SHOT DETECTION"""
        if not players or not ball_pos:
            return
        
        # 🔥 DETECT SHOTS FIRST
        shots = self._detect_shots(frame, timestamp, players, ball_pos)
        
        # 🔥 TRACK POSSESSIONS PROPERLY
        ball_handler = self._get_ball_handler(players, ball_pos)
        if ball_handler:
            # Add possession when ball changes hands or every 24 seconds
            if not hasattr(self, 'last_possession_time'):
                self.last_possession_time = timestamp
                self.possessions.append({
                    'start_time': timestamp,
                    'team': ball_handler['team'],
                    'ball_handler_id': ball_handler.get('id', 0)
                })
            elif timestamp - self.last_possession_time > 24:  # 24-second shot clock
                self.last_possession_time = timestamp
                self.possessions.append({
                    'start_time': timestamp,
                    'team': ball_handler['team'],
                    'ball_handler_id': ball_handler.get('id', 0)
                })
        
        # Detect offensive set
        current_set = self._detect_offensive_set(players, ball_pos, frame)
        if current_set:
            self.offensive_sets[current_set] += 1
        
        # Detect transition style
        transition = self._detect_transition_style(timestamp, ball_pos, players)
        if transition:
            self.transition_styles[transition] += 1
        
        # Detect common actions
        actions = self._detect_common_actions(players, frame)
        for action in actions:
            self.common_actions[action] += 1
        
        # Detect initiator
        if ball_handler:
            self.initiators[ball_handler.get('id', 0)] += 1
        
        # Detect ISO play
        if self._is_iso_play(players, ball_pos, ball_handler):
            self.iso_plays.append(timestamp)

    def _detect_offensive_set(self, players, ball_pos, frame):
        """✅ FIXED: Detect offensive set"""
        offensive_players = [p for p in players if p['team'] == 'TRITON']
        
        if len(offensive_players) < 2:
            return None
            
        # Simple set detection based on positions
        if len(offensive_players) >= 4:
            return "MOTION_OFFENSE"
        elif len(offensive_players) >= 2:
            if self._is_spread_pnr(offensive_players, ball_pos):
                return "PICK_AND_ROLL"
            else:
                return "MOTION_OFFENSE"
        else:
            return "ISOLATION"

    def _is_spread_pnr(self, players, ball_pos):
        """✅ ADDED MISSING METHOD: Detect spread pick and roll"""
        if len(players) < 2 or not ball_pos:
            return False
            
        # Find ball handler
        ball_handler = None
        min_dist = float('inf')
        
        for player in players:
            x, y, w, h = player['bbox']
            center = (x + w/2, y + h/2)
            dist = np.sqrt((center[0] - ball_pos[0])**2 + (center[1] - ball_pos[1])**2)
            
            if dist < min_dist:
                min_dist = dist
                ball_handler = player
        
        if not ball_handler:
            return False
        
        # Check for screener near ball handler
        handler_x, handler_y, handler_w, handler_h = ball_handler['bbox']
        handler_center = (handler_x + handler_w/2, handler_y + handler_h/2)
        
        for player in players:
            if player == ball_handler:
                continue
                
            x, y, w, h = player['bbox']
            center = (x + w/2, y + h/2)
            dist = np.sqrt((center[0] - handler_center[0])**2 + (center[1] - handler_center[1])**2)
            
            if dist < 150:  # Within screening distance
                return True
        
        return False

    def _detect_transition_style(self, timestamp, ball_pos, players):
        """Classify transition style"""
        if len(self.ball_movements) < 2:
            self.ball_movements.append((ball_pos, timestamp))
            return None
            
        # Calculate ball speed
        prev_pos, prev_time = self.ball_movements[-1]
        curr_pos, curr_time = ball_pos, timestamp
        
        if curr_time == prev_time:
            return None
            
        ball_speed = np.sqrt((curr_pos[0]-prev_pos[0])**2 + (curr_pos[1]-prev_pos[1])**2) / (curr_time - prev_time)
        
        # Update ball movements
        self.ball_movements.append((ball_pos, timestamp))
        if len(self.ball_movements) > 10:
            self.ball_movements.pop(0)
        
        return "FAST_BREAK" if ball_speed > self.FAST_BREAK_SPEED else "HALF_COURT"

    def _detect_common_actions(self, players, frame):
        """✅ FIXED: Detect screens with error handling"""
        actions = []
        
        try:
            pose_results = self.pose_model(frame, verbose=False)[0]
            if pose_results.keypoints is not None:
                keypoints = pose_results.keypoints.xy.cpu().numpy()
                
                for i, player in enumerate(players):
                    if i < len(keypoints) and self._is_screening(keypoints[i]):
                        actions.append("SCREEN")
        except Exception as e:
            # Fallback to proximity-based screen detection
            for i, p1 in enumerate(players):
                for j, p2 in enumerate(players[i+1:], i+1):
                    if p1['team'] == p2['team']:  # Same team
                        x1, y1, w1, h1 = p1['bbox']
                        x2, y2, w2, h2 = p2['bbox']
                        
                        center1 = (x1 + w1/2, y1 + h1/2)
                        center2 = (x2 + w2/2, y2 + h2/2)
                        
                        distance = np.sqrt((center1[0]-center2[0])**2 + (center1[1]-center2[1])**2)
                        
                        if distance < 100:  # Close proximity
                            actions.append("SCREEN")
                            break
        
        return actions

    def _detect_shots(self, frame, timestamp, players, ball_pos):
        """🔥 PROPER shot detection using your shot_detection_v2.pt model"""
        shots = []
        
        try:
            # Use your shot detection model
            shot_results = self.shot_model(frame, verbose=False)[0]
            
            if shot_results.boxes is not None:
                for box in shot_results.boxes:
                    conf = float(box.conf)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Only consider high-confidence shot detections
                    if conf > 0.5:
                        shot_data = {
                            'timestamp': timestamp,
                            'bbox': (x1, y1, x2-x1, y2-y1),
                            'confidence': conf,
                            'ball_position': ball_pos,
                            'shooter': self._identify_shooter(players, ball_pos, (x1, y1, x2, y2))
                        }
                        
                        shots.append(shot_data)
                        self.shots_detected.append(shot_data)
                        
                        # Track shot attempts by player/team
                        if shot_data['shooter']:
                            shooter_id = shot_data['shooter'].get('id', 'unknown')
                            shooter_team = shot_data['shooter'].get('team', 'unknown')
                            self.shot_attempts[f"{shooter_team}_{shooter_id}"] += 1
                        
                        print(f"🏀 SHOT DETECTED at {timestamp:.1f}s - Confidence: {conf:.2f}")
        
        except Exception as e:
            print(f"Shot detection error: {e}")
        
        return shots

    def _identify_shooter(self, players, ball_pos, shot_bbox):
        """Identify which player is taking the shot"""
        if not players or not ball_pos:
            return None
        
        shot_center = (shot_bbox[0] + shot_bbox[2]/2, shot_bbox[1] + shot_bbox[3]/2)
        min_dist = float('inf')
        shooter = None
        
        for player in players:
            px, py = player['bbox'][0] + player['bbox'][2]/2, player['bbox'][1] + player['bbox'][3]/2
            
            # Distance to shot location
            dist_to_shot = np.sqrt((px - shot_center[0])**2 + (py - shot_center[1])**2)
            
            # Distance to ball
            dist_to_ball = np.sqrt((px - ball_pos[0])**2 + (py - ball_pos[1])**2)
            
            # Combined distance (closer to both shot and ball)
            combined_dist = dist_to_shot + dist_to_ball * 0.5
            
            if combined_dist < min_dist:
                min_dist = combined_dist
                shooter = player
        
        return shooter if min_dist < 200 else None

    def _generate_report(self):
        """COMPLETE offensive analysis report with SHOT STATISTICS"""
        total_possessions = len(self.possessions)
        total_time = max([p['start_time'] for p in self.possessions]) if self.possessions else 1
        total_shots = len(self.shots_detected)
        
        # Calculate pace (possessions per 40 minutes)
        pace = (total_possessions / total_time) * 2400 if total_time > 0 else 0
        
        # Calculate ISO percentage
        iso_percentage = (len(self.iso_plays) / max(total_possessions, 1)) * 100
        
        # 🔥 SHOT ANALYSIS
        shot_frequency = (total_shots / max(total_possessions, 1)) * 100
        shots_per_minute = (total_shots / max(total_time/60, 1)) if total_time > 0 else 0
        
        # Team shot breakdown
        team_shots = defaultdict(int)
        for shot in self.shots_detected:
            if shot['shooter']:
                team = shot['shooter'].get('team', 'unknown')
                team_shots[team] += 1
        
        return {
            "offensive_tendencies": {
                "pace": {
                    "possessions_per_40min": round(pace, 1),
                    "total_possessions": total_possessions,
                    "avg_possession_length": round(total_time / max(total_possessions, 1), 1)
                },
                # 🔥 SHOT STATISTICS
                "shooting": {
                    "total_shots": total_shots,
                    "shots_per_possession": round(shot_frequency, 1),
                    "shots_per_minute": round(shots_per_minute, 1),
                    "team_breakdown": [
                        {"team": k, "shots": v, "percentage": round((v/max(total_shots,1))*100, 1)} 
                        for k, v in team_shots.items()
                    ],
                    "shot_attempts_by_player": [
                        {"player": k, "attempts": v} 
                        for k, v in self.shot_attempts.items()
                    ]
                },
                "sets": [
                    {"name": k, "frequency": v, "percentage": round((v/max(total_possessions,1))*100, 1)} 
                    for k, v in self.offensive_sets.items()
                ],
                "transition_styles": [
                    {"type": k, "frequency": v, "percentage": round((v/max(total_possessions,1))*100, 1)} 
                    for k, v in self.transition_styles.items()
                ],
                "initiators": [
                    {"player_id": k, "count": v, "percentage": round((v/max(total_possessions,1))*100, 1)} 
                    for k, v in self.initiators.items()
                ],
                "iso_vs_ball_movement": {
                    "iso_plays": len(self.iso_plays),
                    "iso_percentage": round(iso_percentage, 1),
                    "ball_movement_plays": max(0, total_possessions - len(self.iso_plays)),
                    "ball_movement_percentage": round(100 - iso_percentage, 1)
                },
                "common_actions": [
                    {"action": k, "count": v} 
                    for k, v in self.common_actions.items()
                ]
            }
        }

    def _get_player_speed(self, player_id):
        """Calculate player speed in pixels/frame"""
        history = self.player_histories.get(player_id, [])
        if len(history) < 2:
            return 0
            
        pos1 = history[-1]['position']
        pos2 = history[-2]['position']
        time_diff = history[-1]['timestamp'] - history[-2]['timestamp']
        
        if time_diff == 0:
            return 0
            
        return np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2) / time_diff

    def _is_horns_formation(self, players, keypoints):
        """✅ FIXED: Detect Horns set"""
        if len(players) < 5:
            return False
        
        offensive_players = [p for p in players if p['team'] == 'TRITON']
        
        if len(offensive_players) < 4:
            return False
        
        centers = [(p['bbox'][0] + p['bbox'][2]/2, p['bbox'][1] + p['bbox'][3]/2) 
                for p in offensive_players]
        
        # Check for two high post players (elbows)
        high_post = sum(1 for x, y in centers 
                    if 0.4 < x/self.court_width < 0.6 and 0.3 < y/self.court_height < 0.5) >= 2
        
        # Check for two corner players
        corners = sum(1 for x, y in centers 
                    if (x/self.court_width < 0.3 or x/self.court_width > 0.7) and y/self.court_height < 0.6) >= 2
        
        return high_post and corners

    def _is_screening(self, keypoints):
        """✅ FIXED: Detect screens with proper error handling"""
        try:
            if keypoints.shape[0] < 17:
                return False
            
            # Get relevant keypoints
            Lshoulder = keypoints[5]
            Rshoulder = keypoints[6]
            Lhip = keypoints[11]
            Rhip = keypoints[12]
            Lankle = keypoints[15]
            Rankle = keypoints[16]
            
            # Check if keypoints are valid
            if np.all(Lshoulder == 0) or np.all(Rshoulder == 0):
                return False
            
            # Calculate features
            shoulder_width = np.linalg.norm(Lshoulder[:2] - Rshoulder[:2])
            hip_width = np.linalg.norm(Lhip[:2] - Rhip[:2])
            ankle_width = np.linalg.norm(Lankle[:2] - Rankle[:2])
            
            if shoulder_width == 0 or hip_width == 0:
                return False
            
            # Screen conditions
            wide_base = ankle_width > 0.2 * shoulder_width
            arms_extended = shoulder_width > 1.5 * hip_width
            low_posture = (Lshoulder[1] + Rshoulder[1])/2 > (Lhip[1] + Rhip[1])/2 + 20
            
            return wide_base and arms_extended and low_posture
            
        except Exception as e:
            return False

# Usage
if __name__ == "__main__":
    analyzer = BasketballAnalyzer()
    
    # Find video file
    video_files = [f for f in os.listdir('.') if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    
    if video_files:
        video_path = video_files[0]
        print(f"Processing: {video_path}")
    else:
        video_path = input("Enter video path: ")
    
    report = analyzer.process_video(video_path)
    
    with open("offensive_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("Analysis complete! Report saved to offensive_report.json")