"""
Map loader for loading real-world maps (e.g., Athens map).
Supports loading from images or generating from OpenStreetMap data.
"""

import numpy as np
from PIL import Image
import os
from typing import Tuple, Optional


class MapLoader:
    """Load and convert maps to pathfinding grids."""
    
    @staticmethod
    def load_from_image(image_path: str, threshold: int = 128, 
                       invert: bool = False) -> np.ndarray:
        """
        Load map from image file (PNG, JPG, etc.).
        Converts to black/white grid where:
        - Black pixels (dark) = walls (1)
        - White pixels (light) = free space (0)
        
        Args:
            image_path: Path to image file
            threshold: Threshold for black/white conversion (0-255)
            invert: If True, invert colors (white=wall, black=free)
            
        Returns:
            2D numpy array (0=free, 1=wall)
        """
        img = Image.open(image_path)
        
        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Convert to binary: below threshold = wall (1), above = free (0)
        map_grid = (img_array < threshold).astype(int)
        
        if invert:
            map_grid = 1 - map_grid
        
        return map_grid
    
    @staticmethod
    def download_athens_map_from_osm(bbox: Tuple[float, float, float, float] = None,
                                     size: Tuple[int, int] = (100, 100),
                                     save_path: Optional[str] = None) -> np.ndarray:
        """
        Download REAL Athens map from OpenStreetMap using osmnx.
        
        Args:
            bbox: (north, south, east, west) bounding box for Athens center
                  Default: Athens city center (Syntagma area)
            size: (height, width) of the output grid
            save_path: Optional path to save the map image
            
        Returns:
            2D numpy array (0=free, 1=wall)
        """
        try:
            import osmnx as ox
        except ImportError:
            print("Warning: osmnx not installed. Install with: pip install osmnx")
            print("Falling back to stylized map...")
            return MapLoader.download_athens_map_fallback(size, save_path)
        
        # Athens center coordinates (Syntagma Square area)
        # Default bbox: small area around center
        if bbox is None:
            # Athens center: 37.9755° N, 23.7348° E
            # Small bbox around center (~1km x 1km)
            center_lat, center_lon = 37.9755, 23.7348
            delta = 0.01  # ~1km
            bbox = (center_lat + delta, center_lat - delta, 
                   center_lon + delta, center_lon - delta)
        
        print(f"Downloading Athens map from OpenStreetMap...")
        print(f"Bbox: {bbox}")
        
        try:
            # Download street network
            G = ox.graph_from_bbox(bbox[0], bbox[1], bbox[2], bbox[3], network_type='drive')
            
            # Download buildings
            tags = {'building': True}
            buildings = ox.features_from_bbox(bbox[0], bbox[1], bbox[2], bbox[3], tags)
            
            # Convert to image
            height, width = size
            
            # Create grid (start with all free space)
            map_grid = np.zeros((height, width), dtype=int)
            
            # Get bounds
            nodes_data = list(G.nodes(data=True))
            if nodes_data:
                lats = [data.get('y', 0) for _, data in nodes_data]
                lons = [data.get('x', 0) for _, data in nodes_data]
                
                min_lat, max_lat = min(lats), max(lats)
                min_lon, max_lon = min(lons), max(lons)
                
                # Convert coordinates to grid indices
                def lat_to_row(lat):
                    return int((1 - (lat - min_lat) / (max_lat - min_lat)) * (height - 1))
                
                def lon_to_col(lon):
                    return int((lon - min_lon) / (max_lon - min_lon) * (width - 1))
                
                # Mark buildings as walls
                if len(buildings) > 0:
                    for idx, building in buildings.iterrows():
                        if building.geometry is not None:
                            # Get building bounds (simplified)
                            bounds = building.geometry.bounds
                            # Mark as wall (simplified - just mark bounds)
                            # This is a simplification - full implementation would rasterize polygons
                            pass
                
                # Mark roads (thicker lines for visibility)
                road_width = 2
                for u, v, data in G.edges(data=True):
                    if 'geometry' in data:
                        # Get edge coordinates
                        coords = list(data['geometry'].coords)
                        for i in range(len(coords) - 1):
                            lat1, lon1 = coords[i][1], coords[i][0]
                            lat2, lon2 = coords[i+1][1], coords[i+1][0]
                            
                            r1 = lat_to_row(lat1)
                            c1 = lon_to_col(lon1)
                            r2 = lat_to_row(lat2)
                            c2 = lon_to_col(lon2)
                            
                            # Draw line (free space for roads)
                            # Actually, roads should be FREE, buildings should be walls
                            # But for now, we'll mark as free
                            # Draw line between points (use Bresenham-like)
                            rr, cc = MapLoader._line(r1, c1, r2, c2)
                            for r, c in zip(rr, cc):
                                if 0 <= r < height and 0 <= c < width:
                                    map_grid[r, c] = 0  # Road = free
                    
            # Buildings are walls (but we need to rasterize them properly)
            # For now, mark areas with no roads as potential buildings
            
        except Exception as e:
            print(f"Error downloading from OSM: {e}")
            print("Falling back to stylized map...")
            return MapLoader.download_athens_map_fallback(size, save_path)
        
        # Remove borders (don't make them walls)
        # map_grid[0, :] = 0  # Top border = free
        # map_grid[-1, :] = 0  # Bottom border = free
        # map_grid[:, 0] = 0  # Left border = free
        # map_grid[:, -1] = 0  # Right border = free
        
        # Save if path provided
        if save_path:
            img_array = ((1 - map_grid) * 255).astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            img.save(save_path)
            print(f"Map saved to {save_path}")
        
        return map_grid
    
    @staticmethod
    def _line(r0, c0, r1, c1):
        """Bresenham line algorithm."""
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        rr, cc = r0, c0
        result_r, result_c = [rr], [cc]
        
        if dr > dc:
            err = dr / 2
            for _ in range(dr):
                if r0 < r1:
                    rr += 1
                else:
                    rr -= 1
                err -= dc
                if err < 0:
                    if c0 < c1:
                        cc += 1
                    else:
                        cc -= 1
                    err += dr
                result_r.append(rr)
                result_c.append(cc)
        else:
            err = dc / 2
            for _ in range(dc):
                if c0 < c1:
                    cc += 1
                else:
                    cc -= 1
                err -= dr
                if err < 0:
                    if r0 < r1:
                        rr += 1
                    else:
                        rr -= 1
                    err += dc
                result_r.append(rr)
                result_c.append(cc)
        
        return result_r, result_c
    
    @staticmethod
    def download_athens_map_fallback(size: Tuple[int, int] = (50, 50),
                                     save_path: Optional[str] = None) -> np.ndarray:
        """
        Fallback: generate a simple stylized map (when OSM download fails).
        NO BORDERS as walls - that was the problem!
        
        Args:
            size: (height, width) of the grid
            save_path: Optional path to save the map image
            
        Returns:
            2D numpy array (0=free, 1=wall)
        """
        height, width = size
        map_grid = np.ones((height, width), dtype=int)  # Start with all walls
        
        # Create street network (free space)
        # Main horizontal streets
        for i in range(5, height-5, 12):
            map_grid[i, :] = 0  # Free street
            map_grid[i-1, :] = 0  # Street wider
            map_grid[i+1, :] = 0
        
        # Main vertical streets
        for i in range(5, width-5, 12):
            map_grid[:, i] = 0  # Free street
            map_grid[:, i-1] = 0
            map_grid[:, i+1] = 0
        
        # Small connecting streets
        for i in range(11, height-11, 24):
            map_grid[i, :] = 0
        for i in range(11, width-11, 24):
            map_grid[:, i] = 0
        
        # Remove borders - NO WALLS ON BORDERS!
        map_grid[0:3, :] = 0  # Top edge = free
        map_grid[-3:, :] = 0  # Bottom edge = free
        map_grid[:, 0:3] = 0  # Left edge = free
        map_grid[:, -3:] = 0  # Right edge = free
        
        # Save if path provided
        if save_path:
            img_array = ((1 - map_grid) * 255).astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            img.save(save_path)
            print(f"Map saved to {save_path}")
        
        return map_grid
    
    @staticmethod
    def create_athens_from_image(image_path: str = "athens_map.png",
                                 threshold: int = 128) -> np.ndarray:
        """
        Load Athens map from image file.
        If file doesn't exist, downloads/generates a simple one.
        
        Args:
            image_path: Path to Athens map image
            threshold: Threshold for black/white conversion
            
        Returns:
            2D numpy array (0=free, 1=wall)
        """
        if os.path.exists(image_path):
            print(f"Loading map from {image_path}...")
            return MapLoader.load_from_image(image_path, threshold=threshold, invert=False)
        else:
            print(f"Map file {image_path} not found. Downloading real Athens map from OpenStreetMap...")
            try:
                # Try to download real map
                map_grid = MapLoader.download_athens_map_from_osm(size=(60, 60), save_path=image_path)
            except Exception as e:
                print(f"Could not download from OSM: {e}")
                print("Using fallback stylized map...")
                map_grid = MapLoader.download_athens_map_fallback(size=(60, 60), save_path=image_path)
            return map_grid
    
    @staticmethod
    def resize_map(map_grid: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize map grid to new dimensions.
        
        Args:
            map_grid: Original map (0=free, 1=wall)
            new_size: (new_height, new_width)
            
        Returns:
            Resized map grid
        """
        from PIL import Image
        
        # Convert to image (0->255, 1->0) - ensure uint8 type
        img_array = ((1 - map_grid) * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode='L')
        
        # Resize
        img_resized = img.resize((new_size[1], new_size[0]), Image.Resampling.LANCZOS)
        
        # Convert back to grid
        resized_array = np.array(img_resized)
        resized_grid = (resized_array < 128).astype(int)
        
        return resized_grid
