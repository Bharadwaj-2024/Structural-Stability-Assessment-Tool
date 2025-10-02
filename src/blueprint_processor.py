import os
import json
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
import math
import random
from scipy.spatial.distance import euclidean
from collections import deque

class BlueprintProcessor:
    """
    A class to process blueprint PDFs and perform analysis with rescue-focused features
    """
    
    def __init__(self, blueprint_dir):
        """
        Initialize the BlueprintProcessor
        
        Args:
            blueprint_dir (str): Directory containing blueprint PDFs
        """
        self.blueprint_dir = Path(blueprint_dir)
        self.output_dir = Path('output')
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.converted_dir = self.output_dir / 'converted'
        self.annotations_dir = self.output_dir / 'annotations'
        self.damage_dir = self.output_dir / 'damage_assessment'
        self.risk_zones_dir = self.output_dir / 'risk_zones'
        
        for dir_path in [self.converted_dir, self.annotations_dir, self.damage_dir, self.risk_zones_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Damage types and their risk levels
        self.damage_types = {
            'collapsed_wall': {'risk': 'high', 'color': (255, 0, 0), 'radius': 50},
            'crack': {'risk': 'moderate', 'color': (255, 165, 0), 'radius': 30},
            'leaning_beam': {'risk': 'high', 'color': (255, 0, 0), 'radius': 40},
            'blocked_passage': {'risk': 'moderate', 'color': (255, 165, 0), 'radius': 35}
        }
        
    def convert_pdfs(self):
        """
        Convert PDFs to images for annotation
        
        Returns:
            list: List of converted image files
        """
        converted_files = []
        
        if not self.blueprint_dir.exists():
            print(f"Blueprint directory {self.blueprint_dir} does not exist")
            return converted_files
            
        # Find all PDF files
        pdf_files = list(self.blueprint_dir.glob('*.pdf'))
        
        for pdf_file in pdf_files:
            try:
                # Convert PDF to images
                images = convert_from_path(pdf_file)
                
                for i, image in enumerate(images):
                    # Save each page as a separate image
                    output_filename = f"{pdf_file.stem}_page_{i+1}.png"
                    output_path = self.converted_dir / output_filename
                    image.save(output_path, 'PNG')
                    converted_files.append(output_filename)
                    
            except Exception as e:
                print(f"Error converting {pdf_file}: {str(e)}")
                
        return converted_files
    
    def save_annotations(self, filename, annotations):
        """
        Save annotations for a given file
        
        Args:
            filename (str): Name of the annotated file
            annotations (dict): Annotation data
            
        Returns:
            Path: Path to the saved annotation file
        """
        # Create annotation filename
        base_name = Path(filename).stem
        annotation_file = self.annotations_dir / f"{base_name}_annotations.json"
        
        # Save annotations as JSON
        with open(annotation_file, 'w') as f:
            json.dump(annotations, f, indent=2)
            
        return annotation_file
    
    def analyze_blueprint(self, file):
        """
        Analyze a blueprint image using computer vision with structural detection
        
        Args:
            file: Uploaded file object
            
        Returns:
            dict: Analysis results including structural elements and auto-detected damage
        """
        try:
            # Save uploaded file temporarily
            temp_path = self.output_dir / f"temp_{file.filename}"
            file.save(temp_path)
            
            # Load and process image
            image = cv2.imread(str(temp_path))
            
            if image is None:
                return {'error': 'Could not load image'}
                
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Basic image analysis
            height, width = gray.shape
            
            # Enhanced structural analysis
            structural_analysis = self._analyze_blueprint_structure(gray)
            
            # Auto-detect potential damage
            auto_damage = self._detect_structural_damage(gray)
            
            # Basic text detection using OCR
            try:
                text = pytesseract.image_to_string(gray)
                detected_text = [line.strip() for line in text.split('\n') if line.strip()]
            except:
                detected_text = []
            
            # Calculate basic metrics
            results = {
                'filename': file.filename,
                'dimensions': {'width': width, 'height': height},
                'contours_found': structural_analysis['total_contours'],
                'detected_text_lines': len(detected_text),
                'detected_text': detected_text[:10],  # First 10 lines only
                'structural_elements': structural_analysis,
                'auto_detected_damage': auto_damage,
                'status': 'success'
            }
            
            # Save structural data for pathfinding
            struct_file = self.output_dir / f"{file.filename}_structure.json"
            with open(struct_file, 'w') as f:
                json.dump({
                    'walls': structural_analysis['walls'],
                    'rooms': structural_analysis['rooms'],
                    'corridors': structural_analysis['corridors'],
                    'building_outline': structural_analysis['building_outline'],
                    'walkable_areas': structural_analysis['walkable_areas'],
                    'indoor_mask': structural_analysis['indoor_mask'],
                    'obstacles': structural_analysis['obstacles'],
                    'dimensions': {'width': width, 'height': height}
                }, f)
            
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
                
            return results
            
        except Exception as e:
            return {'error': f'Analysis failed: {str(e)}', 'status': 'error'}
    
    def _analyze_blueprint_structure(self, gray_image):
        """
        Analyze blueprint structure to identify walls, rooms, and walkable areas with enhanced building detection
        
        Args:
            gray_image: Grayscale blueprint image
            
        Returns:
            dict: Structural analysis results (JSON serializable)
        """
        height, width = gray_image.shape
        
        # Multi-level edge detection for better structure recognition
        edges_canny = cv2.Canny(gray_image, 30, 100)
        edges_strong = cv2.Canny(gray_image, 100, 200)
        
        # Detect thick lines (major walls)
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thick_walls = cv2.morphologyEx(edges_strong, cv2.MORPH_CLOSE, kernel_line)
        
        # Find building outline (largest enclosed area)
        building_outline = self._find_building_outline(gray_image)
        
        # Detect internal walls and structures
        internal_walls = self._detect_internal_walls(gray_image, building_outline)
        
        # Find rooms and corridors
        rooms, corridors = self._detect_rooms_and_corridors(gray_image, building_outline, internal_walls)
        
        # Create enhanced walkable map
        walkable_map, indoor_mask = self._create_enhanced_walkable_map(
            gray_image, building_outline, internal_walls, rooms
        )
        
        # Detect automatic obstacles (furniture, fixtures, etc.)
        obstacles = self._detect_automatic_obstacles(gray_image, indoor_mask)
        
        # Convert everything to JSON-serializable format
        return {
            'total_contours': int(len(internal_walls) + len(rooms)),
            'walls': self._convert_to_serializable(internal_walls),
            'rooms': self._convert_to_serializable(rooms),
            'corridors': self._convert_to_serializable(corridors),
            'building_outline': self._convert_to_serializable(building_outline),
            'walkable_areas': walkable_map,
            'indoor_mask': indoor_mask.tolist() if hasattr(indoor_mask, 'tolist') else indoor_mask,
            'obstacles': self._convert_to_serializable(obstacles),
            'structure_detected': len(internal_walls) > 0 or len(rooms) > 0
        }
    
    def _convert_to_serializable(self, data):
        """
        Convert numpy types and complex objects to JSON-serializable format
        """
        if isinstance(data, (list, tuple)):
            return [self._convert_to_serializable(item) for item in data]
        elif isinstance(data, dict):
            return {key: self._convert_to_serializable(value) for key, value in data.items()}
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif hasattr(data, 'tolist'):  # numpy arrays
            return data.tolist()
        else:
            return data
    
    def _find_building_outline(self, gray_image):
        """
        Find the main building outline/perimeter
        """
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour (building outline)
        if contours:
            building_contour = max(contours, key=cv2.contourArea)
            
            # Simplify the contour
            epsilon = 0.02 * cv2.arcLength(building_contour, True)
            building_outline = cv2.approxPolyDP(building_contour, epsilon, True)
            
            return {
                'contour': building_outline.tolist(),
                'area': float(cv2.contourArea(building_contour)),
                'bbox': [int(x) for x in cv2.boundingRect(building_contour)]
            }
        
        return None
    
    def _detect_internal_walls(self, gray_image, building_outline):
        """
        Detect internal walls within the building
        """
        walls = []
        
        # Create mask for building interior
        if building_outline:
            mask = np.zeros(gray_image.shape, dtype=np.uint8)
            contour = np.array(building_outline['contour'], dtype=np.int32)
            cv2.fillPoly(mask, [contour], 255)
        else:
            mask = np.ones(gray_image.shape, dtype=np.uint8) * 255
        
        # Detect lines using HoughLinesP
        edges = cv2.Canny(gray_image, 50, 150)
        edges_masked = cv2.bitwise_and(edges, mask)
        
        lines = cv2.HoughLinesP(edges_masked, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if length > 20:  # Minimum wall length
                    walls.append({
                        'start': [x1, y1],
                        'end': [x2, y2],
                        'length': length,
                        'type': 'internal_wall'
                    })
        
        return walls
    
    def _detect_rooms_and_corridors(self, gray_image, building_outline, walls):
        """
        Detect rooms and corridors using flood fill and area analysis
        """
        rooms = []
        corridors = []
        
        # Create a mask for analysis
        height, width = gray_image.shape
        analysis_mask = np.ones((height, width), dtype=np.uint8) * 255
        
        # Block out walls
        for wall in walls:
            start = tuple(wall['start'])
            end = tuple(wall['end'])
            cv2.line(analysis_mask, start, end, 0, 3)
        
        # Find connected components (rooms)
        num_labels, labels = cv2.connectedComponents(analysis_mask)
        
        for label in range(1, num_labels):
            # Get region for this label
            region_mask = (labels == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(contour)
                
                if area > 1000:  # Minimum room size
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
                    
                    room_data = {
                        'contour': contour.tolist(),
                        'bbox': [x, y, w, h],
                        'area': area,
                        'center': [x + w//2, y + h//2],
                        'aspect_ratio': aspect_ratio
                    }
                    
                    # Classify as room or corridor based on aspect ratio
                    if aspect_ratio > 3:  # Long and narrow = corridor
                        corridors.append(room_data)
                    else:
                        rooms.append(room_data)
        
        return rooms, corridors
    
    def _create_enhanced_walkable_map(self, gray_image, building_outline, walls, rooms):
        """
        Create enhanced walkable map that forces indoor navigation
        """
        height, width = gray_image.shape
        
        # Start with everything as non-walkable
        walkable_map = np.zeros((height, width), dtype=np.uint8)
        indoor_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Mark building interior as potentially walkable
        if building_outline:
            contour = np.array(building_outline['contour'], dtype=np.int32)
            cv2.fillPoly(indoor_mask, [contour], 255)
            cv2.fillPoly(walkable_map, [contour], 255)
        else:
            # If no building outline detected, use center area
            center_margin = min(width, height) // 6
            cv2.rectangle(indoor_mask, (center_margin, center_margin), 
                         (width-center_margin, height-center_margin), 255, -1)
            cv2.rectangle(walkable_map, (center_margin, center_margin), 
                         (width-center_margin, height-center_margin), 255, -1)
        
        # Block out walls from walkable areas
        for wall in walls:
            start = tuple(wall['start'])
            end = tuple(wall['end'])
            cv2.line(walkable_map, start, end, 0, 8)  # Thick wall blocking
        
        # Add room interiors as definitely walkable
        for room in rooms:
            contour = np.array(room['contour'], dtype=np.int32)
            # Shrink room slightly to avoid wall edges
            contour_center = np.mean(contour, axis=0).astype(int)
            shrunk_contour = contour_center + 0.8 * (contour - contour_center)
            cv2.fillPoly(walkable_map, [shrunk_contour.astype(np.int32)], 255)
        
        # Apply morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        walkable_map = cv2.morphologyEx(walkable_map, cv2.MORPH_OPEN, kernel)
        walkable_map = cv2.morphologyEx(walkable_map, cv2.MORPH_CLOSE, kernel)
        
        return walkable_map.tolist(), indoor_mask
    
    def _detect_automatic_obstacles(self, gray_image, indoor_mask):
        """
        Automatically detect obstacles like furniture, fixtures, etc.
        """
        obstacles = []
        
        # Apply mask to focus on indoor areas only
        masked_image = cv2.bitwise_and(gray_image, indoor_mask)
        
        # Detect dark rectangular areas (furniture)
        _, binary = cv2.threshold(masked_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find small to medium contours (furniture-sized)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 200 < area < 5000:  # Furniture-like size
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
                
                # Filter out wall-like objects
                if aspect_ratio < 4:  # Not too elongated
                    obstacles.append({
                        'bbox': [x, y, w, h],
                        'center': [x + w//2, y + h//2],
                        'area': area,
                        'type': 'furniture'
                    })
        
        return obstacles[:10]  # Limit to most significant obstacles
    
    def _detect_structural_damage(self, gray_image):
        """
        Auto-detect potential structural damage in blueprint
        
        Args:
            gray_image: Grayscale blueprint image
            
        Returns:
            list: Auto-detected damage points
        """
        auto_damage = []
        
        # Detect irregular patterns that might indicate damage
        
        # 1. Detect potential cracks (thin irregular lines)
        kernel_crack = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
        cracks = cv2.filter2D(gray_image, -1, kernel_crack)
        crack_thresh = cv2.threshold(cracks, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        crack_contours, _ = cv2.findContours(crack_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in crack_contours:
            area = cv2.contourArea(contour)
            if 100 < area < 1000:  # Crack-like size
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    auto_damage.append({
                        'x': cx, 'y': cy,
                        'type': 'crack',
                        'confidence': 0.7,
                        'auto_detected': True
                    })
        
        # 2. Detect potential debris/blocked areas (irregular dark spots)
        blurred = cv2.GaussianBlur(gray_image, (9, 9), 2)
        dark_areas = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        debris_contours, _ = cv2.findContours(dark_areas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in debris_contours:
            area = cv2.contourArea(contour)
            if 500 < area < 3000:  # Debris-like size
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    auto_damage.append({
                        'x': cx, 'y': cy,
                        'type': 'blocked_passage',
                        'confidence': 0.6,
                        'auto_detected': True
                    })
        
        # Limit to most significant detections
        return auto_damage[:5]

    def save_damage_assessment(self, filename, damage_marks):
        """
        Save damage assessment data for a blueprint
        
        Args:
            filename (str): Blueprint filename
            damage_marks (list): List of damage marks with coordinates and types
            
        Returns:
            dict: Assessment results with risk zones
        """
        try:
            base_name = Path(filename).stem
            assessment_file = self.damage_dir / f"{base_name}_damage_assessment.json"
            
            # Generate risk zones based on damage marks
            risk_zones = self._generate_risk_zones(damage_marks)
            
            assessment_data = {
                'filename': filename,
                'damage_marks': damage_marks,
                'risk_zones': risk_zones,
                'timestamp': json.dumps({'timestamp': 'now'}, default=str),
                'total_damage_points': len(damage_marks),
                'high_risk_areas': len([z for z in risk_zones if z['risk_level'] == 'high']),
                'moderate_risk_areas': len([z for z in risk_zones if z['risk_level'] == 'moderate'])
            }
            
            # Save assessment data
            with open(assessment_file, 'w') as f:
                json.dump(assessment_data, f, indent=2)
            
            return {
                'status': 'success',
                'assessment_file': str(assessment_file),
                'risk_zones': risk_zones,
                'damage_summary': {
                    'total_damage': len(damage_marks),
                    'high_risk_zones': assessment_data['high_risk_areas'],
                    'moderate_risk_zones': assessment_data['moderate_risk_areas']
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _generate_risk_zones(self, damage_marks):
        """
        Generate risk zones based on damage marks
        
        Args:
            damage_marks (list): List of damage marks
            
        Returns:
            list: Risk zones with coordinates and risk levels
        """
        risk_zones = []
        
        for damage in damage_marks:
            damage_type = damage.get('type', 'crack')
            x, y = damage.get('x', 0), damage.get('y', 0)
            
            if damage_type in self.damage_types:
                damage_info = self.damage_types[damage_type]
                risk_zone = {
                    'x': x,
                    'y': y,
                    'radius': damage_info['radius'],
                    'risk_level': damage_info['risk'],
                    'damage_type': damage_type,
                    'color': damage_info['color']
                }
                risk_zones.append(risk_zone)
        
        return risk_zones
    
    def find_safe_path(self, filename, start_point, end_point, damage_marks=None):
        """
        Find a safe path through the blueprint using intelligent indoor navigation (simplified and robust)
        
        Args:
            filename (str): Blueprint filename
            start_point (tuple): Starting coordinates (x, y)
            end_point (tuple): Ending coordinates (x, y)
            damage_marks (list): List of damage marks to avoid
            
        Returns:
            dict: Safe path data with coordinates and safety analysis
        """
        try:
            # Load structural data if available
            struct_file = self.output_dir / f"{filename}_structure.json"
            walkable_map = None
            indoor_mask = None
            building_data = None
            
            if struct_file.exists():
                try:
                    with open(struct_file, 'r') as f:
                        struct_data = json.load(f)
                        walkable_map = struct_data.get('walkable_areas', [])
                        indoor_mask = struct_data.get('indoor_mask', [])
                        building_data = struct_data
                except:
                    print("Error loading structural data, using defaults")
            
            # Create simple grid for pathfinding
            grid_width, grid_height = 80, 60  # Simplified grid
            
            # Convert coordinates to grid scale
            start_grid = [
                max(0, min(grid_width-1, int(start_point[0] * grid_width / 800))),
                max(0, min(grid_height-1, int(start_point[1] * grid_height / 600)))
            ]
            end_grid = [
                max(0, min(grid_width-1, int(end_point[0] * grid_width / 800))),
                max(0, min(grid_height-1, int(end_point[1] * grid_height / 600)))
            ]
            
            # Create obstacle map
            obstacle_map = self._create_simple_obstacle_map(
                grid_width, grid_height, damage_marks, walkable_map
            )
            
            # Try pathfinding
            path_grid = self._enhanced_astar(start_grid, end_grid, obstacle_map)
            
            if path_grid and len(path_grid) > 1:
                # Convert back to canvas coordinates
                path_canvas = []
                for gx, gy in path_grid:
                    canvas_x = int(gx * 800 / grid_width)
                    canvas_y = int(gy * 600 / grid_height)
                    path_canvas.append([canvas_x, canvas_y])
                
                # Smooth the path
                path_smoothed = self._smooth_path(path_canvas)
                
                # Analyze path safety
                safety_analysis = self._analyze_path_safety(path_smoothed, damage_marks)
                
                return {
                    'status': 'success',
                    'path': path_smoothed,
                    'safety_analysis': safety_analysis,
                    'path_length': len(path_smoothed),
                    'estimated_time': len(path_smoothed) * 0.15,
                    'follows_structure': True,
                    'indoor_route': True
                }
            else:
                # Create fallback route
                fallback_path = self._create_smart_fallback(start_point, end_point)
                safety_analysis = self._analyze_path_safety(fallback_path, damage_marks)
                
                return {
                    'status': 'warning',
                    'message': 'Using fallback route - please verify safety manually',
                    'path': fallback_path,
                    'safety_analysis': safety_analysis,
                    'path_length': len(fallback_path),
                    'estimated_time': len(fallback_path) * 0.25,
                    'follows_structure': False,
                    'indoor_route': True
                }
                
        except Exception as e:
            print(f"Pathfinding error: {str(e)}")
            # Emergency fallback
            emergency_path = self._create_direct_path(start_point, end_point)
            safety_analysis = self._analyze_path_safety(emergency_path, damage_marks)
            
            return {
                'status': 'warning',
                'message': f'Using emergency direct route due to error: {str(e)}',
                'path': emergency_path,
                'safety_analysis': safety_analysis,
                'path_length': len(emergency_path),
                'estimated_time': len(emergency_path) * 0.3,
                'follows_structure': False,
                'indoor_route': False
            }
    
    def _create_simple_obstacle_map(self, width, height, damage_marks, walkable_map):
        """
        Create a simple obstacle map for pathfinding
        """
        # Start with all areas walkable
        obstacle_map = [[0 for _ in range(width)] for _ in range(height)]
        
        # Add border walls for containment
        for i in range(width):
            obstacle_map[0][i] = 255  # Top border
            obstacle_map[height-1][i] = 255  # Bottom border
        for i in range(height):
            obstacle_map[i][0] = 255  # Left border
            obstacle_map[i][width-1] = 255  # Right border
        
        # Add some basic internal structure if no walkable map
        if not walkable_map or len(walkable_map) == 0:
            # Create basic building structure
            # Central corridor
            corridor_y = height // 2
            for x in range(width // 4, 3 * width // 4):
                for y in range(corridor_y - 2, corridor_y + 3):
                    if 0 <= y < height:
                        obstacle_map[y][x] = 0  # Ensure corridor is walkable
        
        # Add damage zones as obstacles
        if damage_marks:
            for damage in damage_marks:
                # Convert damage coordinates to grid
                grid_x = int(damage.get('x', 0) * width / 800)
                grid_y = int(damage.get('y', 0) * height / 600)
                
                damage_type = damage.get('type', 'crack')
                if damage_type in self.damage_types:
                    radius = max(1, int(self.damage_types[damage_type]['radius'] * width / 800 / 20))
                    
                    # Mark damage area as obstacle
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            if dx*dx + dy*dy <= radius*radius:
                                nx, ny = grid_x + dx, grid_y + dy
                                if 0 <= nx < width and 0 <= ny < height:
                                    obstacle_map[ny][nx] = 255
        
        return obstacle_map
    
    def _create_smart_fallback(self, start_point, end_point):
        """
        Create a smart fallback route with waypoints
        """
        path = [start_point]
        
        start_x, start_y = start_point
        end_x, end_y = end_point
        
        # Create L-shaped path with intermediate points
        # Move horizontally 70% of the way
        mid_x = start_x + (end_x - start_x) * 0.7
        if abs(mid_x - start_x) > 30:
            path.append([int(mid_x), start_y])
        
        # Move vertically most of the way
        if abs(end_y - start_y) > 30:
            path.append([int(mid_x), int(start_y + (end_y - start_y) * 0.8)])
        
        # Move to final position
        path.append(end_point)
        
        return path
    
    def _force_point_indoor(self, point, indoor_mask, walkable_map):
        """
        Force a point to be inside the building if it's outside
        """
        if indoor_mask is None or len(indoor_mask) == 0:
            return point
        
        x, y = int(point[0]), int(point[1])
        height, width = len(indoor_mask), len(indoor_mask[0]) if indoor_mask else (600, 800)
        
        # Clamp to bounds
        x = max(0, min(width-1, x))
        y = max(0, min(height-1, y))
        
        # Check if point is already inside
        if y < len(indoor_mask) and x < len(indoor_mask[0]) and indoor_mask[y][x] > 0:
            return [x, y]
        
        # Find nearest indoor point using distance transform
        indoor_array = np.array(indoor_mask, dtype=np.uint8)
        distance_transform = cv2.distanceTransform(indoor_array, cv2.DIST_L2, 5)
        
        # Find the closest indoor point
        max_distance = 100  # Search radius
        best_point = [x, y]
        min_distance = float('inf')
        
        for dy in range(-max_distance, max_distance + 1, 5):
            for dx in range(-max_distance, max_distance + 1, 5):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if indoor_array[ny, nx] > 0:  # Indoor point
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < min_distance:
                            min_distance = dist
                            best_point = [nx, ny]
        
        return best_point
    
    def _create_default_indoor_map(self, width, height):
        """
        Create a default indoor map when no structural data is available
        """
        walkable_map = np.zeros((height, width), dtype=np.uint8)
        indoor_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Create a building outline with margins
        margin = min(width, height) // 8
        building_rect = (margin, margin, width - 2*margin, height - 2*margin)
        
        # Mark indoor area
        cv2.rectangle(indoor_mask, (building_rect[0], building_rect[1]), 
                     (building_rect[0] + building_rect[2], building_rect[1] + building_rect[3]), 255, -1)
        
        # Create walkable corridors and rooms
        cv2.rectangle(walkable_map, (building_rect[0] + 20, building_rect[1] + 20), 
                     (building_rect[0] + building_rect[2] - 20, building_rect[1] + building_rect[3] - 20), 255, -1)
        
        # Add some internal structure (simulated walls)
        # Horizontal corridor
        corridor_y = height // 2
        cv2.rectangle(walkable_map, (margin + 30, corridor_y - 15), 
                     (width - margin - 30, corridor_y + 15), 255, -1)
        
        # Vertical connections
        cv2.rectangle(walkable_map, (width//3 - 10, margin + 30), 
                     (width//3 + 10, height - margin - 30), 255, -1)
        cv2.rectangle(walkable_map, (2*width//3 - 10, margin + 30), 
                     (2*width//3 + 10, height - margin - 30), 255, -1)
        
        return walkable_map.tolist(), indoor_mask
    
    def _create_smart_obstacle_map(self, walkable_map, indoor_mask, damage_marks, building_data):
        """
        Create an intelligent obstacle map
        """
        walkable_array = np.array(walkable_map, dtype=np.uint8)
        obstacle_map = (walkable_array == 0).astype(np.uint8) * 255
        
        # Add damage zones as obstacles
        if damage_marks:
            height, width = obstacle_map.shape
            for damage in damage_marks:
                x = max(0, min(width-1, int(damage.get('x', 0))))
                y = max(0, min(height-1, int(damage.get('y', 0))))
                damage_type = damage.get('type', 'crack')
                
                if damage_type in self.damage_types:
                    radius = self.damage_types[damage_type]['radius']
                    # Scale radius appropriately
                    scaled_radius = max(5, int(radius * width / 800))
                    cv2.circle(obstacle_map, (x, y), scaled_radius, 255, -1)
        
        # Add automatic obstacles from building data
        if building_data and 'obstacles' in building_data:
            for obstacle in building_data['obstacles']:
                bbox = obstacle['bbox']
                cv2.rectangle(obstacle_map, (bbox[0], bbox[1]), 
                            (bbox[0] + bbox[2], bbox[1] + bbox[3]), 255, -1)
        
        return obstacle_map
    
    def _intelligent_pathfinding(self, start, end, obstacle_map, walkable_map, building_data):
        """
        Intelligent pathfinding that creates realistic indoor routes
        """
        # Try direct path first
        direct_path = self._enhanced_astar(start, end, obstacle_map)
        
        if direct_path and len(direct_path) > 2:
            return direct_path
        
        # If direct path fails, use waypoint-based routing
        waypoints = self._find_corridor_waypoints(start, end, walkable_map, building_data)
        
        full_path = [start]
        current_pos = start
        
        for waypoint in waypoints:
            segment_path = self._enhanced_astar(current_pos, waypoint, obstacle_map)
            if segment_path and len(segment_path) > 1:
                full_path.extend(segment_path[1:])  # Skip first point to avoid duplication
                current_pos = waypoint
            else:
                # If segment fails, try direct connection
                full_path.append(waypoint)
                current_pos = waypoint
        
        # Final segment to destination
        final_segment = self._enhanced_astar(current_pos, end, obstacle_map)
        if final_segment and len(final_segment) > 1:
            full_path.extend(final_segment[1:])
        else:
            full_path.append(end)
        
        return full_path
    
    def _find_corridor_waypoints(self, start, end, walkable_map, building_data):
        """
        Find intermediate waypoints that follow building corridors
        """
        waypoints = []
        
        # Get building dimensions
        if building_data and 'dimensions' in building_data:
            width = building_data['dimensions']['width']
            height = building_data['dimensions']['height']
        else:
            height, width = len(walkable_map), len(walkable_map[0]) if walkable_map else (600, 800)
        
        start_x, start_y = start
        end_x, end_y = end
        
        # Create waypoints based on a realistic building navigation pattern
        # Move to central corridor first, then to destination
        
        central_corridor_y = height // 2
        
        # Waypoint 1: Move to horizontal corridor
        if abs(start_y - central_corridor_y) > 30:
            waypoint1_x = start_x + (end_x - start_x) * 0.3
            waypoints.append([int(waypoint1_x), central_corridor_y])
        
        # Waypoint 2: Move along corridor toward destination
        if abs(end_x - start_x) > 100:
            waypoint2_x = start_x + (end_x - start_x) * 0.7
            waypoints.append([int(waypoint2_x), central_corridor_y])
        
        # Waypoint 3: Move toward final destination
        if abs(end_y - central_corridor_y) > 30:
            waypoint3_x = end_x
            waypoint3_y = central_corridor_y + (end_y - central_corridor_y) * 0.5
            waypoints.append([int(waypoint3_x), int(waypoint3_y)])
        
        return waypoints
    
    def _make_path_realistic(self, path, building_data):
        """
        Make the path more realistic by adding natural curves and following corridors
        """
        if len(path) < 3:
            return path
        
        realistic_path = [path[0]]
        
        # Add intermediate points with slight curves
        for i in range(1, len(path) - 1):
            current = path[i]
            
            # Add slight randomness to make path less robotic
            offset_x = random.randint(-3, 3) if len(path) > 5 else 0
            offset_y = random.randint(-3, 3) if len(path) > 5 else 0
            
            adjusted_point = [
                max(0, current[0] + offset_x),
                max(0, current[1] + offset_y)
            ]
            
            realistic_path.append(adjusted_point)
        
        realistic_path.append(path[-1])
        
        # Smooth the path
        return self._smooth_path(realistic_path)
    
    def _create_emergency_fallback(self, start, end, building_data):
        """
        Create an emergency fallback route when pathfinding fails
        """
        # Create a simple L-shaped path that avoids going outside
        path = [start]
        
        start_x, start_y = start
        end_x, end_y = end
        
        # Move horizontally first, then vertically
        mid_x = start_x + (end_x - start_x) * 0.7
        mid_point = [int(mid_x), start_y]
        
        # Add waypoint only if it's significantly different
        if abs(mid_x - start_x) > 20:
            path.append(mid_point)
        
        # Add another waypoint for vertical movement
        if abs(end_y - start_y) > 20:
            path.append([int(mid_x), end_y])
        
        path.append(end)
        
        return path
    
    def _enhanced_astar(self, start, goal, obstacle_map):
        """
        Enhanced A* pathfinding that considers building structure (simplified and robust)
        """
        if not obstacle_map or len(obstacle_map) == 0:
            return self._create_direct_path(start, goal)
        
        height, width = len(obstacle_map), len(obstacle_map[0])
        
        # Ensure start and goal are within bounds and walkable
        start = (max(0, min(width-1, int(start[0]))), max(0, min(height-1, int(start[1]))))
        goal = (max(0, min(width-1, int(goal[0]))), max(0, min(height-1, int(goal[1]))))
        
        # If start or goal are in obstacles, try to find nearby free spots
        if obstacle_map[start[1]][start[0]] > 0:
            start = self._find_nearest_free_spot(start, obstacle_map)
        if obstacle_map[goal[1]][goal[0]] > 0:
            goal = self._find_nearest_free_spot(goal, obstacle_map)
        
        if not start or not goal:
            return self._create_direct_path([start[0] if start else 0, start[1] if start else 0], 
                                          [goal[0] if goal else width-1, goal[1] if goal else height-1])
        
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        def get_neighbors(pos):
            x, y = pos
            neighbors = []
            
            # 8-directional movement
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if obstacle_map[ny][nx] == 0:  # Walkable
                        cost = 1.4 if dx != 0 and dy != 0 else 1.0  # Diagonal movement costs more
                        neighbors.append(((nx, ny), cost))
            
            return neighbors
        
        # A* algorithm
        open_set = [start]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        closed_set = set()
        
        iterations = 0
        max_iterations = min(width * height, 10000)  # Prevent infinite loops
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            # Get node with lowest f_score
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(list(current))
                    current = came_from[current]
                path.append(list(start))
                return path[::-1]
            
            open_set.remove(current)
            closed_set.add(current)
            
            for neighbor, move_cost in get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in open_set:
                    open_set.append(neighbor)
                elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue
                
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
        
        # If A* fails, return direct path
        return self._create_direct_path(list(start), list(goal))
    
    def _find_nearest_free_spot(self, point, obstacle_map):
        """
        Find the nearest free spot if the given point is in an obstacle
        """
        height, width = len(obstacle_map), len(obstacle_map[0])
        x, y = point
        
        # Search in expanding circles
        for radius in range(1, min(50, min(width, height) // 4)):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx*dx + dy*dy <= radius*radius:  # Circle search
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            if obstacle_map[ny][nx] == 0:  # Free spot
                                return (nx, ny)
        
        return None
    
    def _smooth_path(self, path):
        """
        Smooth the path to make it more natural and reduce sharp turns
        """
        if len(path) < 3:
            return path
        
        smoothed = [path[0]]  # Keep start point
        
        # Use moving average for intermediate points
        window_size = min(5, len(path) // 3)
        
        for i in range(1, len(path) - 1):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(path), i + window_size // 2 + 1)
            
            # Calculate average position
            avg_x = sum(p[0] for p in path[start_idx:end_idx]) / (end_idx - start_idx)
            avg_y = sum(p[1] for p in path[start_idx:end_idx]) / (end_idx - start_idx)
            
            smoothed.append([int(avg_x), int(avg_y)])
        
        smoothed.append(path[-1])  # Keep end point
        
        # Remove points that are too close together
        final_path = [smoothed[0]]
        min_distance = 10  # Minimum distance between points
        
        for point in smoothed[1:]:
            last_point = final_path[-1]
            distance = math.sqrt((point[0] - last_point[0])**2 + (point[1] - last_point[1])**2)
            
            if distance >= min_distance:
                final_path.append(point)
        
        return final_path
    
    def _simple_astar(self, start, goal, obstacle_map, dimensions):
        """
        Simplified A* pathfinding algorithm
        """
        width, height = dimensions
        
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        def get_neighbors(pos):
            x, y = pos
            neighbors = []
            # 8-directional movement
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if obstacle_map[ny, nx] == 0:  # Not an obstacle
                        neighbors.append((nx, ny))
            return neighbors
        
        open_set = [start]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        closed_set = set()
        
        while open_set:
            # Find node with lowest f_score
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            open_set.remove(current)
            closed_set.add(current)
            
            for neighbor in get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in open_set:
                    open_set.append(neighbor)
                elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue
                
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
        
        return None  # No path found
    
    def _create_direct_path(self, start_point, end_point):
        """
        Create a direct line path between two points
        """
        path = []
        x1, y1 = start_point
        x2, y2 = end_point
        
        # Bresenham's line algorithm simplified
        steps = max(abs(x2 - x1), abs(y2 - y1))
        if steps == 0:
            return [[x1, y1]]
        
        step_x = (x2 - x1) / steps
        step_y = (y2 - y1) / steps
        
        for i in range(steps + 1):
            x = int(x1 + i * step_x)
            y = int(y1 + i * step_y)
            path.append([x, y])
        
        return path
    
    def _analyze_path_safety(self, path, damage_marks):
        """
        Analyze the safety of a given path
        
        Args:
            path (list): List of path coordinates
            damage_marks (list): List of damage marks
            
        Returns:
            dict: Safety analysis results
        """
        if not damage_marks or not path:
            return {
                'overall_safety': 'safe',
                'risk_points': 0,
                'warnings': [],
                'safe_segments': len(path)
            }
        
        risk_points = 0
        warnings = []
        dangerous_segments = 0
        
        for point in path:
            x, y = point[0], point[1]
            point_risk = 0
            
            for damage in damage_marks:
                dx, dy = damage.get('x', 0), damage.get('y', 0)
                distance = math.sqrt((x - dx)**2 + (y - dy)**2)
                
                damage_type = damage.get('type', 'crack')
                if damage_type in self.damage_types:
                    danger_radius = self.damage_types[damage_type]['radius']
                    
                    if distance < danger_radius:
                        risk_points += 1
                        point_risk += 1
                        if len(warnings) < 3:  # Limit warnings
                            warnings.append(f"Path passes near {damage_type.replace('_', ' ')} at ({int(dx)}, {int(dy)})")
            
            if point_risk > 0:
                dangerous_segments += 1
        
        # Determine overall safety
        danger_ratio = dangerous_segments / len(path) if path else 0
        
        if danger_ratio > 0.3:  # More than 30% of path is dangerous
            overall_safety = 'dangerous'
        elif danger_ratio > 0.1:  # More than 10% of path has risk
            overall_safety = 'moderate_risk'
        else:
            overall_safety = 'safe'
        
        return {
            'overall_safety': overall_safety,
            'risk_points': risk_points,
            'warnings': warnings,
            'safe_segments': len(path) - dangerous_segments,
            'danger_ratio': round(danger_ratio * 100, 1)
        }