import cv2
import numpy as np
from ultralytics import YOLO
import os
import json
import time
from collections import defaultdict, deque
from deep_sort_realtime.deepsort_tracker import DeepSort

class OptimizedBasketballAnalyzer:
    def __init__(self):
        # ESSENTIAL INITIALIZATION - SIMPLIFIED FOR SPEED
        self.detection_model = YOLO('yolo11s.pt')
        self.shot_model = YOLO('shot_detection_v2.pt')
        self.ball_detector = YOLO('ball_detector_model.pt')
        
        # FRAME SKIP for speed
        self.frame_skip_counter = 0
        
        # Tracking
        self.tracker = DeepSort(
            max_age=30,           
            n_init=2,             
            max_iou_distance=0.7,
            max_cosine_distance=0.2
        )
        self.player_histories = defaultdict(deque)
        
        # TEAM CLASSIFICATION IMPROVEMENTS
        self.team_assignments = {}  # Persistent team assignments
        self.team_confidence = defaultdict(list)  # Track confidence over time
        

        
        # Analysis data
        self.possessions = []
        self.current_possession = None
        self.possession_start_time = None
        self.offensive_sets = defaultdict(int)
        self.transition_styles = defaultdict(int)
        self.common_actions = defaultdict(int)
        self.initiators = defaultdict(int)
        self.iso_plays = []
        self.ball_movements = []
        self.shots_detected = []
        self.shot_attempts = defaultdict(int)
        self.passes = []
        self.interceptions = []
        
        # TEAM PLAY ANALYSIS
        self.team_play_sequences = []
        self.iso_sequences = []
        
        # Ball possession
        self.possession_threshold = 50
        self.min_frames = 11
        self.containment_threshold = 0.8
        
        # Config
        self.ISO_DISTANCE = 300
        self.FAST_BREAK_SPEED = 10
        self.PLAYER_SPEED_THRESHOLD = 5.0
        self.court_width = 1920
        self.court_height = 1080

    def process_video(self, video_path):
        """Process video with improved tracking and team classification"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        
        print("Starting analysis")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_count / fps
            
            # Detection and tracking
            players, ball_pos = self._detect_players_and_ball(frame)
            players = self._filter_players(players, frame.shape)
            
            # Tracking
            tracked_players = self._track_players_improved(players, frame)
            
            # Team classification
            tracked_players = self._assign_teams_improved(tracked_players, frame, timestamp)
            
            # Analysis (skip some frames for speed)
            self.frame_skip_counter += 1
            
            # Shot detection every 10 frames only
            if self.frame_skip_counter % 10 == 0:
                self._detect_shots(frame, timestamp, tracked_players)
            
            # Core analysis every frame
            self._analyze_possession_complete(timestamp, tracked_players, ball_pos, frame)
            
            # Other analysis every 3 frames
            if self.frame_skip_counter % 3 == 0:
                self._analyze_offensive_sets(timestamp, tracked_players, ball_pos)
                self._analyze_transition_style(timestamp, tracked_players, ball_pos)
                self._analyze_common_actions(timestamp, tracked_players)
                self._detect_passes_and_interceptions(timestamp, tracked_players)
            
            # Visualization
            vis_frame = self._create_enhanced_visualization(frame, tracked_players, ball_pos, frame_count, timestamp)
            cv2.imshow('Basketball Analysis', vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")
        
        cap.release()
        cv2.destroyAllWindows()
        return self._generate_complete_report()

    def _detect_players_and_ball(self, frame):
        """Detect players and ball using separate models"""
        # Player detection
        det_results = self.detection_model(frame, verbose=False)[0]
        players = []
        
        if det_results.boxes is not None:
            for box in det_results.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                if cls_id == 0 and conf > 0.5:  # Person
                    players.append({
                        'bbox': (x1, y1, x2-x1, y2-y1),
                        'conf': conf
                    })
        
        # Ball detection using dedicated model
        ball_results = self.ball_detector(frame, verbose=False)[0]
        ball_pos = None
        
        if ball_results.boxes is not None:
            for box in ball_results.boxes:
                conf = float(box.conf)
                if conf > 0.4:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    ball_pos = ((x1 + x2)/2, (y1 + y2)/2)
                    break
        
        return players, ball_pos

    def _track_players_improved(self, players, frame):
        """Better player tracking"""
        if not players:
            return []
            
        detections = []
        for p in players:
            x, y, w, h = p['bbox']
            detections.append(([x, y, w, h], p['conf'], 0))
        
        tracks = self.tracker.update_tracks(detections, frame=frame)
        tracked_players = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb()
            
            # Update player history
            current_pos = (ltrb[0], ltrb[1])
            self.player_histories[track_id].append({
                'position': current_pos,
                'timestamp': time.time(),
                'bbox': (int(ltrb[0]), int(ltrb[1]), int(ltrb[2]-ltrb[0]), int(ltrb[3]-ltrb[1]))
            })
            
            # Keep last 5 positions for smoothing
            if len(self.player_histories[track_id]) > 5:
                self.player_histories[track_id].popleft()
            
            tracked_players.append({
                'id': track_id,
                'bbox': (int(ltrb[0]), int(ltrb[1]), int(ltrb[2]-ltrb[0]), int(ltrb[3]-ltrb[1])),
                'conf': 0.8  # Default confidence
            })
            
        return tracked_players

    def _assign_teams_improved(self, players, frame, timestamp):
        """Proper team classification with persistence"""
        for player in players:
            player_id = player['id']
            bbox = player['bbox']
            
            # Check if we already know this player's team
            if player_id in self.team_assignments:
                # Use existing assignment but verify occasionally
                if len(self.team_confidence[player_id]) < 10:
                    team = self._classify_team_robust(frame, bbox)
                    self.team_confidence[player_id].append(team)
                    
                    # Update assignment based on majority vote
                    if len(self.team_confidence[player_id]) >= 5:
                        team_votes = self.team_confidence[player_id]
                        light_votes = team_votes.count("TEAM_LIGHT")
                        dark_votes = team_votes.count("TEAM_DARK")
                        self.team_assignments[player_id] = "TEAM_LIGHT" if light_votes > dark_votes else "TEAM_DARK"
                
                player['team'] = self.team_assignments[player_id]
            else:
                # New player - classify and store
                team = self._classify_team_robust(frame, bbox)
                self.team_assignments[player_id] = team
                self.team_confidence[player_id] = [team]
                player['team'] = team
        
        return players

    def _classify_team_robust(self, frame, bbox):
        """Color pattern detection for white+blue vs full blue"""
        x, y, w, h = bbox
        
        # Extract jersey region (chest area)
        jersey_y1 = y + int(h * 0.15)  # Start from chest
        jersey_y2 = y + int(h * 0.55)  # End at waist
        jersey_x1 = x + int(w * 0.2)   # Center area
        jersey_x2 = x + int(w * 0.8)
        
        # Ensure valid region
        jersey_y1 = max(0, jersey_y1)
        jersey_y2 = min(frame.shape[0], jersey_y2)
        jersey_x1 = max(0, jersey_x1)
        jersey_x2 = min(frame.shape[1], jersey_x2)
        
        if jersey_y2 <= jersey_y1 or jersey_x2 <= jersey_x1:
            return "TEAM_DARK"
        
        jersey_region = frame[jersey_y1:jersey_y2, jersey_x1:jersey_x2]
        
        if jersey_region.size == 0:
            return "TEAM_DARK"
        
        try:
            # Color pattern analysis
            hsv_region = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
            
            # Analyze color distribution
            h_channel = hsv_region[:, :, 0]  # Hue
            s_channel = hsv_region[:, :, 1]  # Saturation  
            v_channel = hsv_region[:, :, 2]  # Value/Brightness
            
            # RGB channels
            b_channel = jersey_region[:, :, 0]
            g_channel = jersey_region[:, :, 1]
            r_channel = jersey_region[:, :, 2]
            
            # Simple brightness-based classification
            avg_brightness = np.mean(v_channel)
            
            # White jersey detection - much simpler approach
            if avg_brightness > 100:  # Bright = white jersey
                return "TEAM_LIGHT"
            else:  # Dark = dark jersey
                return "TEAM_DARK"
            
        except Exception as e:
            # Fallback: Simple brightness
            avg_brightness = np.mean(jersey_region)
            return "TEAM_LIGHT" if avg_brightness > 120 else "TEAM_DARK"

    def _analyze_possession_complete(self, timestamp, players, ball_pos, frame):
        """Track possessions, team play, and ISO"""
        if not players or not ball_pos:
            return
        
        # Get ball handler
        ball_handler = self._get_ball_handler(players, ball_pos)
        
        if ball_handler:
            current_team = ball_handler['team']
            player_id = ball_handler.get('id', 0)
            
            # Track initiators
            self.initiators[f"{current_team}_{player_id}"] += 1
            
            # Possession tracking
            if self.current_possession != current_team:
                # Possession changed
                if self.current_possession is not None and self.possession_start_time is not None:
                    # End previous possession
                    possession_duration = timestamp - self.possession_start_time
                    self.possessions.append({
                        'team': self.current_possession,
                        'duration': possession_duration,
                        'start_time': self.possession_start_time,
                        'end_time': timestamp
                    })
                
                # Start new possession
                self.current_possession = current_team
                self.possession_start_time = timestamp
            
            # Team play vs ISO analysis
            self._analyze_play_style(timestamp, players, ball_pos, ball_handler)

    def _analyze_play_style(self, timestamp, players, ball_pos, ball_handler):
        """Analyze team play vs ISO play"""
        if not ball_handler:
            return
        
        # Count teammates near ball handler
        bx, by = ball_pos
        teammates_close = 0
        teammates_total = 0
        
        for player in players:
            if player.get('team') == ball_handler.get('team') and player.get('id') != ball_handler.get('id'):
                teammates_total += 1
                px, py = player['bbox'][0] + player['bbox'][2]/2, player['bbox'][1] + player['bbox'][3]/2
                dist = np.sqrt((px - bx)**2 + (py - by)**2)
                
                if dist < self.ISO_DISTANCE:
                    teammates_close += 1
        
        # Determine play style
        if teammates_total > 0:
            if teammates_close <= 1:
                # ISO PLAY
                self.iso_sequences.append({
                    'timestamp': timestamp,
                    'team': ball_handler['team'],
                    'player_id': ball_handler.get('id', 0)
                })
            else:
                # TEAM PLAY
                self.team_play_sequences.append({
                    'timestamp': timestamp,
                    'team': ball_handler['team'],
                    'teammates_involved': teammates_close
                })

    def _create_enhanced_visualization(self, frame, players, ball_pos, frame_count, timestamp):
        """Enhanced visualization with team info"""
        vis_frame = frame.copy()
        
        # Draw players with DIFFERENT COLORS
        for player in players:
            x, y, w, h = player['bbox']
            team = player.get('team', 'UNKNOWN')
            player_id = player.get('id', 0)
            
            # Different colors for teams
            if team == "TEAM_LIGHT":
                color = (0, 255, 0)      # GREEN for light team
                team_label = "LIGHT"
            else:
                color = (0, 0, 255)      # RED for dark team  
                team_label = "DARK"
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 3)
            
            # Draw player label
            cv2.putText(vis_frame, f'{team_label[0]}{player_id}', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw ball
        if ball_pos:
            cv2.circle(vis_frame, (int(ball_pos[0]), int(ball_pos[1])), 15, (255, 255, 0), 3)
            cv2.putText(vis_frame, 'BALL', (int(ball_pos[0]-20), int(ball_pos[1]-25)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Info overlay
        light_count = len([p for p in players if p.get('team') == 'TEAM_LIGHT'])
        dark_count = len([p for p in players if p.get('team') == 'TEAM_DARK'])
        
        # Basic info
        cv2.putText(vis_frame, f'Frame: {frame_count}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_frame, f'LIGHT: {light_count} | DARK: {dark_count}', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Possession info
        if self.current_possession and self.possession_start_time:
            possession_time = timestamp - self.possession_start_time
            cv2.putText(vis_frame, f'POSSESSION: {self.current_possession} ({possession_time:.1f}s)', 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Play style info
        recent_iso = len([p for p in self.iso_sequences if timestamp - p['timestamp'] < 5])
        recent_team_play = len([p for p in self.team_play_sequences if timestamp - p['timestamp'] < 5])
        
        if recent_iso > recent_team_play:
            play_style = "ISO PLAY"
            style_color = (0, 165, 255)  # Orange
        elif recent_team_play > 0:
            play_style = "TEAM PLAY"
            style_color = (0, 255, 255)  # Yellow
        else:
            play_style = "TRANSITION"
            style_color = (255, 255, 255)  # White
        
        cv2.putText(vis_frame, f'STYLE: {play_style}', (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, style_color, 2)
        
        # Shots and interceptions info
        recent_shots = len([s for s in self.shots_detected if timestamp - s['timestamp'] < 3])
        recent_interceptions = len([i for i in self.interceptions if timestamp - i['timestamp'] < 5])
        
        if recent_shots > 0:
            cv2.putText(vis_frame, f'SHOT DETECTED!', (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if recent_interceptions > 0:
            cv2.putText(vis_frame, f'INTERCEPTION!', (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return vis_frame

    def _get_ball_handler(self, players, ball_pos):
        """ADVANCED BALL POSSESSION (15-keypoint + containment from GitHub)"""
        if not players or not ball_pos:
            return None
        
        best_player = None
        best_score = float('inf')
        
        # Create ball bbox (estimate)
        ball_size = 20
        ball_bbox = (ball_pos[0] - ball_size//2, ball_pos[1] - ball_size//2, 
                    ball_pos[0] + ball_size//2, ball_pos[1] + ball_size//2)
        
        for player in players:
            player_bbox = player['bbox']
            x, y, w, h = player_bbox
            player_bbox_xyxy = (x, y, x+w, y+h)
            
            # METHOD 1: Containment ratio
            containment = self._calculate_ball_containment_ratio(player_bbox_xyxy, ball_bbox)
            
            if containment > self.containment_threshold:
                return player
            
            # METHOD 2: 15-keypoint distance analysis
            min_distance = self._find_minimum_distance_to_ball(ball_pos, player_bbox_xyxy)
            
            if min_distance < best_score:
                best_score = min_distance
                best_player = player
        
        return best_player if best_score < self.possession_threshold else None

    def _get_key_basketball_player_assignment_points(self, player_bbox, ball_center):
        """15-KEYPOINT ANALYSIS (from GitHub)"""
        ball_center_x, ball_center_y = ball_center
        x1, y1, x2, y2 = player_bbox
        width = x2 - x1
        height = y2 - y1
        
        output_points = []
        
        # Dynamic points based on ball position
        if y1 < ball_center_y < y2:
            output_points.append((x1, ball_center_y))
            output_points.append((x2, ball_center_y))
        
        if x1 < ball_center_x < x2:
            output_points.append((ball_center_x, y1))
            output_points.append((ball_center_x, y2))
        
        # 15 strategic points
        output_points += [
            (x1 + width//2, y1),          # top center
            (x2, y1),                      # top right
            (x1, y1),                      # top left
            (x2, y1 + height//2),          # center right
            (x1, y1 + height//2),          # center left
            (x1 + width//2, y1 + height//2), # center point
            (x2, y2),                      # bottom right
            (x1, y2),                      # bottom left
            (x1 + width//2, y2),          # bottom center
            (x1 + width//2, y1 + height//3), # mid-top center
            (x1 + width//4, y1 + height//4),  # quarter points
        ]
        return output_points

    def _calculate_ball_containment_ratio(self, player_bbox, ball_bbox):
        """CONTAINMENT RATIO CALCULATION (from GitHub)"""
        px1, py1, px2, py2 = player_bbox
        bx1, by1, bx2, by2 = ball_bbox
        
        intersection_x1 = max(px1, bx1)
        intersection_y1 = max(py1, by1)
        intersection_x2 = min(px2, bx2)
        intersection_y2 = min(py2, by2)
        
        if intersection_x2 < intersection_x1 or intersection_y2 < intersection_y1:
            return 0.0
        
        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
        ball_area = (bx2 - bx1) * (by2 - by1)
        
        return intersection_area / ball_area if ball_area > 0 else 0.0

    def _find_minimum_distance_to_ball(self, ball_center, player_bbox):
        """MINIMUM DISTANCE CALCULATION (from GitHub)"""
        key_points = self._get_key_basketball_player_assignment_points(player_bbox, ball_center)
        distances = [np.sqrt((ball_center[0] - point[0])**2 + (ball_center[1] - point[1])**2) 
                    for point in key_points]
        return min(distances) if distances else float('inf')

    def _detect_shots(self, frame, timestamp, players):
        """IMPROVED SHOT DETECTION - Filter out dribbling"""
        try:
            shot_results = self.shot_model(frame, verbose=False)[0]
            
            if shot_results.boxes is not None:
                for box in shot_results.boxes:
                    conf = float(box.conf)
                    if conf > 0.7:  # HIGHER THRESHOLD to reduce false positives
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        shot_center = ((x1 + x2)/2, (y1 + y2)/2)
                        
                        # VALIDATE SHOT: Check if it's actually a shot, not dribbling
                        if self._validate_shot_detection(shot_center, players, frame):
                            closest_player = self._get_closest_player(players, shot_center)
                            
                            # PREVENT DUPLICATE SHOTS (same player within 2 seconds)
                            if self._is_duplicate_shot(timestamp, closest_player):
                                continue
                            
                            shot_data = {
                                'timestamp': timestamp,
                                'confidence': conf,
                                'bbox': (x1, y1, x2-x1, y2-y1),
                                'shooter': closest_player.get('id', 0) if closest_player else 0,
                                'team': closest_player.get('team', 'UNKNOWN') if closest_player else 'UNKNOWN'
                            }
                            
                            self.shots_detected.append(shot_data)
                            
                            if closest_player:
                                team = closest_player.get('team', 'UNKNOWN')
                                self.shot_attempts[team] += 1
                                
        except Exception as e:
            print(f"Shot detection error: {e}")

    def _validate_shot_detection(self, shot_center, players, frame):
        """VALIDATE: Is this actually a shot or just dribbling?"""
        shot_x, shot_y = shot_center
        
        # RULE 1: Shot should be in upper part of frame (not ground level)
        frame_height = frame.shape[0]
        if shot_y > frame_height * 0.7:  # Too low = likely dribbling
            return False
        
        # RULE 2: Check player arm position (shot = arms up)
        closest_player = self._get_closest_player(players, shot_center)
        if closest_player:
            player_bbox = closest_player['bbox']
            player_center_y = player_bbox[1] + player_bbox[3]/2
            
            # Shot should be at or above player's center (not below = dribbling)
            if shot_y > player_center_y + 50:
                return False
        
        # RULE 3: Ball should be moving upward (not bouncing down)
        # This is a simplified check - in real implementation you'd track ball trajectory
        
        return True

    def _is_duplicate_shot(self, timestamp, player):
        """PREVENT DUPLICATE SHOTS from same player"""
        if not player:
            return False
            
        player_id = player.get('id', 0)
        
        # Check if same player shot within last 2 seconds
        for shot in self.shots_detected[-5:]:  # Check last 5 shots
            if (shot['shooter'] == player_id and 
                timestamp - shot['timestamp'] < 2.0):
                return True
                
        return False

    def _analyze_offensive_sets(self, timestamp, players, ball_pos):
        """ANALYZE offensive sets"""
        if not players or not ball_pos:
            return
        
        # Count players in different court areas
        players_in_paint = 0
        players_on_perimeter = 0
        
        for player in players:
            x, y = player['bbox'][0] + player['bbox'][2]/2, player['bbox'][1] + player['bbox'][3]/2
            
            # Simple court area detection
            if x > self.court_width * 0.3 and x < self.court_width * 0.7:
                players_in_paint += 1
            else:
                players_on_perimeter += 1
        
        # Classify offensive set
        if players_in_paint >= 3:
            self.offensive_sets['POST_UP'] += 1
        elif players_on_perimeter >= 4:
            self.offensive_sets['MOTION_OFFENSE'] += 1
        else:
            self.offensive_sets['PICK_AND_ROLL'] += 1

    def _analyze_transition_style(self, timestamp, players, ball_pos):
        """ANALYZE transition style"""
        if not players or not ball_pos:
            return
        
        # Calculate average player speed
        total_speed = 0
        speed_count = 0
        
        for player in players:
            player_id = player.get('id', 0)
            if len(self.player_histories[player_id]) >= 2:
                recent_positions = list(self.player_histories[player_id])[-2:]
                if len(recent_positions) == 2:
                    pos1 = recent_positions[0]['position']
                    pos2 = recent_positions[1]['position']
                    time_diff = recent_positions[1]['timestamp'] - recent_positions[0]['timestamp']
                    
                    if time_diff > 0:
                        distance = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
                        speed = distance / time_diff
                        total_speed += speed
                        speed_count += 1
        
        if speed_count > 0:
            avg_speed = total_speed / speed_count
            if avg_speed > self.FAST_BREAK_SPEED:
                self.transition_styles['FAST_BREAK'] += 1
            else:
                self.transition_styles['HALF_COURT'] += 1

    def _analyze_common_actions(self, timestamp, players):
        """ANALYZE common actions"""
        if len(players) < 2:
            return
        
        # Screen detection (players close together)
        for i, player1 in enumerate(players):
            for j, player2 in enumerate(players[i+1:], i+1):
                if player1.get('team') == player2.get('team'):
                    p1_center = (player1['bbox'][0] + player1['bbox'][2]/2, 
                               player1['bbox'][1] + player1['bbox'][3]/2)
                    p2_center = (player2['bbox'][0] + player2['bbox'][2]/2, 
                               player2['bbox'][1] + player2['bbox'][3]/2)
                    
                    distance = np.sqrt((p1_center[0] - p2_center[0])**2 + 
                                     (p1_center[1] - p2_center[1])**2)
                    
                    if distance < 80:  # Close proximity
                        self.common_actions['SCREEN'] += 1
                        break

    def _detect_passes_and_interceptions(self, timestamp, players):
        """PASS/INTERCEPTION DETECTION"""
        if not hasattr(self, 'prev_ball_handler'):
            self.prev_ball_handler = None
            return
        
        current_ball_handler = self._get_ball_handler(players, None)
        
        if current_ball_handler and self.prev_ball_handler:
            if current_ball_handler['id'] != self.prev_ball_handler['id']:
                # Ball changed hands
                prev_team = self.prev_ball_handler.get('team', 'UNKNOWN')
                curr_team = current_ball_handler.get('team', 'UNKNOWN')
                
                if prev_team == curr_team:
                    # PASS (same team)
                    self.passes.append({
                        'timestamp': timestamp,
                        'from_player': self.prev_ball_handler['id'],
                        'to_player': current_ball_handler['id'],
                        'team': curr_team
                    })
                elif prev_team != curr_team and prev_team != 'UNKNOWN' and curr_team != 'UNKNOWN':
                    # INTERCEPTION (different teams)
                    self.interceptions.append({
                        'timestamp': timestamp,
                        'intercepting_player': current_ball_handler['id'],
                        'intercepting_team': curr_team,
                        'losing_team': prev_team
                    })
        
        self.prev_ball_handler = current_ball_handler

    def _get_closest_player(self, players, position):
        """Get closest player to a position"""
        if not players:
            return None
            
        px, py = position
        min_dist = float('inf')
        closest = None
        
        for player in players:
            player_x = player['bbox'][0] + player['bbox'][2]/2
            player_y = player['bbox'][1] + player['bbox'][3]/2
            dist = np.sqrt((player_x - px)**2 + (player_y - py)**2)
            
            if dist < min_dist:
                min_dist = dist
                closest = player
                
        return closest

    def _filter_players(self, players, frame_shape):
        """Filter out non-player detections and referees"""
        filtered = []
        h, w = frame_shape[:2]
        
        for player in players:
            x, y, width, height = player['bbox']
            
            # Filter by size
            if width * height < 2000:
                continue
                
            # Filter by position
            if y + height > h * 0.9:
                continue
                
            filtered.append(player)
            
        return filtered

    def _generate_complete_report(self):
        """Generate complete offensive analysis report"""
        total_possessions = len(self.possessions)
        total_iso = len(self.iso_sequences)
        total_team_play = len(self.team_play_sequences)
        total_shots = len(self.shots_detected)
        total_passes = len(self.passes)
        total_interceptions = len(self.interceptions)
        
        # Team possession breakdown
        team_possessions = defaultdict(float)
        for poss in self.possessions:
            team_possessions[poss['team']] += poss['duration']
        
        # Calculate pace (possessions per 40 minutes)
        total_time = sum(team_possessions.values()) if team_possessions else 1
        pace = (total_possessions / total_time) * 2400 if total_time > 0 else 0
        
        # Team shot breakdown
        team_shots = defaultdict(int)
        for shot in self.shots_detected:
            team_shots[shot['team']] += 1
        
        # Team interception breakdown
        team_interceptions = defaultdict(int)
        for interception in self.interceptions:
            team_interceptions[interception['intercepting_team']] += 1
        
        # Generate JSON report
        return {
            "possession_analysis": {
                "total_possessions": total_possessions,
                "pace_per_40min": round(pace, 1),
                "team_possessions": dict(team_possessions),
                "possession_breakdown": [
                    {
                        "team": team,
                        "total_time": round(time_val, 1),
                        "percentage": round((time_val/sum(team_possessions.values()))*100, 1) if team_possessions else 0
                    }
                    for team, time_val in team_possessions.items()
                ]
            },
            "offensive_sets": dict(self.offensive_sets),
            "transition_style": dict(self.transition_styles),
            "play_style_analysis": {
                "iso_plays": total_iso,
                "team_plays": total_team_play,
                "iso_percentage": round((total_iso/(total_iso + total_team_play))*100, 1) if (total_iso + total_team_play) > 0 else 0,
                "ball_movement_percentage": round((total_team_play/(total_iso + total_team_play))*100, 1) if (total_iso + total_team_play) > 0 else 0
            },
            "common_actions": dict(self.common_actions),
            "initiators": {
                "top_ball_handlers": [
                    {"player": initiator, "possessions": count}
                    for initiator, count in sorted(self.initiators.items(), key=lambda x: x[1], reverse=True)[:5]
                ]
            },
            "shot_analysis": {
                "total_shots": total_shots,
                "team_breakdown": dict(team_shots)
            },
            "pass_interception_analysis": {
                "total_passes": total_passes,
                "total_interceptions": total_interceptions,
                "team_interceptions": dict(team_interceptions)
            },
            "summary_metrics": {
                "pace": round(pace, 1),
                "ball_movement_vs_iso": {
                    "ball_movement": round((total_team_play/(total_iso + total_team_play))*100, 1) if (total_iso + total_team_play) > 0 else 0,
                    "iso": round((total_iso/(total_iso + total_team_play))*100, 1) if (total_iso + total_team_play) > 0 else 0
                },
                "turnover_rate": round((total_interceptions/total_possessions)*100, 1) if total_possessions > 0 else 0,
                "shot_frequency": round((total_shots/total_possessions), 2) if total_possessions > 0 else 0
            }
        }

# Usage
if __name__ == "__main__":
    analyzer = OptimizedBasketballAnalyzer()
    
    video_files = [f for f in os.listdir('.') if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    
    if video_files:
        video_path = video_files[0]
        print(f"Processing: {video_path}")
    else:
        video_path = input("Enter video path: ")
    
    start_time = time.time()
    report = analyzer.process_video(video_path)
    end_time = time.time()
    
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    
    with open("basketball_analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("Analysis complete! Report saved to basketball_analysis_report.json")