"""
Correlation-Based Scan Matching (Week 6)

This module aligns consecutive laser scans by searching over candidate
relative poses and scoring each one using grid correlation. It returns
the best-scoring pose along with a covariance estimated from the score
surface's Hessian.

The approach is from Olson (2009) "Real-Time Correlative Scan Matching",
simplified for teaching. It's brute-force (O(n^3) over the search space)
— not elegant, but simple, reliable, and guaranteed to find the global
optimum within the search window.

Student tasks:
  - Implement _build_local_grid()               — rasterise scan into grid
  - Implement _score_alignment()                 — correlation scoring
  - Implement match()                            — brute-force grid search
  - Implement _estimate_covariance_from_hessian() — uncertainty from curvature

References:
  - Olson, "Real-Time Correlative Scan Matching" (2009)
  - Lecture 06: Scan Matching
"""

import numpy as np
from typing import Tuple
from scipy.ndimage import maximum_filter


class ScanMatcher:
    """
    Correlation-based scan matcher.

    Aligns two laser scans by searching over candidate relative poses
    and scoring each one using grid correlation.
    """

    def __init__(self,
                 search_x: float,
                 search_y: float,
                 search_theta: float,
                 resolution_x: float,
                 resolution_y: float,
                 resolution_theta: float,
                 coarse_step_multiplier: float,
                 fine_window_multiplier: float,
                 local_grid_size: int,
                 local_grid_resolution: float,
                 min_score: float):
        self.search_x = search_x
        self.search_y = search_y
        self.search_theta = search_theta
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.resolution_theta = resolution_theta
        self.coarse_step_multiplier = coarse_step_multiplier
        self.fine_window_multiplier = fine_window_multiplier
        self.local_grid_size = local_grid_size
        self.local_grid_resolution = local_grid_resolution
        self.min_score = min_score

    # ========================================================================
    # STUDENT TODO #1: Build Local Occupancy Grid
    # ========================================================================

    def _build_local_grid(self, scan_points: np.ndarray) -> np.ndarray:
        """
        Rasterise scan points into a local occupancy grid for fast correlation.

        The grid is centred at the origin. Each scan point is placed into its
        corresponding cell. After rasterisation, dilate the grid by 1 cell
        (3x3 maximum filter) to tolerate small alignment errors.

        Args:
            scan_points: Nx2 array of (x, y) points in local frame

        Returns:
            2D float32 grid (1.0 where scan points land, 0.0 elsewhere)

        Algorithm:
            1. grid = zeros(local_grid_size x local_grid_size)
            2. offset = local_grid_size // 2  (centre of grid)
            3. For each point (x, y):
               col = int(x / local_grid_resolution) + offset
               row = int(y / local_grid_resolution) + offset
               if in bounds: grid[row, col] = 1.0
            4. grid = maximum_filter(grid, size=3)  (dilate for tolerance)
            5. return grid

        Hints:
            - You can vectorise steps 3-4 with NumPy for speed, or use a loop
            - from scipy.ndimage import maximum_filter is already imported
            - ~10-15 lines of code
        """
        # TODO: YOUR CODE HERE

        # Initialize the grid
        grid = np.zeros((self.local_grid_size, self.local_grid_size))

        # Compute the center offset
        offset = self.local_grid_size // 2

        # Converting scan points into grid points
        cols = (scan_points[:, 0] / self.local_grid_resolution).astype(int) + offset
        rows = (scan_points[:, 1] / self.local_grid_resolution).astype(int) + offset

        # Determining if cells are valid or not
        valid = (
        (rows >= 0) & (rows < self.local_grid_size) &
        (cols >= 0) & (cols < self.local_grid_size)
        )

        grid[rows[valid], cols[valid]] = 1.0 # Set valid grid cells to 1
        grid = maximum_filter(grid, size=3) # Dilate by 3

        return grid
        

    # ========================================================================
    # STUDENT TODO #2: Score Alignment
    # ========================================================================

    def _score_alignment(self, grid: np.ndarray, scan_points: np.ndarray,
                         pose: np.ndarray) -> float:
        """
        Score how well scan_points align with the reference grid
        when transformed by the candidate pose.

        Args:
            grid:        Reference scan's local occupancy grid
            scan_points: Nx2 array of new scan points (local frame)
            pose:        Candidate relative pose [dx, dy, dtheta]

        Returns:
            Correlation score (count of overlapping points)

        Algorithm:
            1. dx, dy, dtheta = pose
            2. c, s = cos(dtheta), sin(dtheta)
            3. For each point (px, py):
               px' = c*px - s*py + dx     (rotate then translate)
               py' = s*px + c*py + dy
               col = int(px' / resolution) + offset
               row = int(py' / resolution) + offset
               if in bounds and grid[row, col] > 0: score += 1
            4. return score

        Hints:
            - Precompute cos/sin once, not per point
            - You can vectorise with NumPy or use a loop
            - ~10-15 lines of code
        """
        # TODO: YOUR CODE HERE
        # Need to extract the actual candidate pose
        dx, dy, dtheta = pose
        # Those points' cos and sin roations are calculated
        c, s = np.cos(dtheta), np.sin(dtheta)

        offset = self.local_grid_size // 2

        # Take our scan points
        px = scan_points[:, 0]
        py = scan_points[:, 1]

        # Apply transformation
        tx = c * px - s * py + dx
        ty = s * px + c * py + dy

        cols = (tx / self.local_grid_resolution).astype(int) + offset
        rows = (ty / self.local_grid_resolution).astype(int) + offset

        # Keep only points that fall inside the grid bounds.
        valid = (
            (rows >= 0) & (rows < self.local_grid_size) &
            (cols >= 0) & (cols < self.local_grid_size)
        )
        
        score = np.sum(grid[rows[valid], cols[valid]] > 0) # Is there an overlap?

        # Return as float to match the function signature.
        return float(score)

    def _score_alignment_map(self, occupancy_grid, scan_points: np.ndarray,
                             pose: np.ndarray) -> float:
        """
        Score how well scan_points align with an occupancy grid map.

        Args:
            occupancy_grid: OccupancyGrid instance (world frame)
            scan_points:    Nx2 array of scan points in local frame
            pose:           Candidate pose [x, y, theta] in map frame

        Returns:
            Correlation score (count of scan points landing on occupied cells)
        """
        x, y, theta = pose
        c, s = np.cos(theta), np.sin(theta)

        px = scan_points[:, 0]
        py = scan_points[:, 1]

        world_x = c * px - s * py + x
        world_y = s * px + c * py + y

        cols = ((world_x - occupancy_grid.origin_x)
                / occupancy_grid.resolution).astype(int)
        rows = ((world_y - occupancy_grid.origin_y)
                / occupancy_grid.resolution).astype(int)

        valid = (
            (rows >= 0) & (rows < occupancy_grid.height) &
            (cols >= 0) & (cols < occupancy_grid.width)
        )

        if not np.any(valid):
            return 0.0

        occupied = occupancy_grid.grid[rows[valid], cols[valid]] > 0.0
        return float(np.sum(occupied))

    def _search_scores(self,
                       score_fn,
                       x_values: np.ndarray,
                       y_values: np.ndarray,
                       theta_values: np.ndarray) -> Tuple[dict, float, np.ndarray, Tuple[int, int, int]]:
        scores = {}
        best_score = -1.0
        best_pose = np.array([x_values[0], y_values[0], theta_values[0]])
        best_idx = (0, 0, 0)

        for ix, x in enumerate(x_values):
            for iy, y in enumerate(y_values):
                for it, theta in enumerate(theta_values):
                    candidate = np.array([x, y, theta])
                    score = score_fn(candidate)
                    scores[(ix, iy, it)] = score

                    if score > best_score:
                        best_score = score
                        best_pose = candidate
                        best_idx = (ix, iy, it)

        return scores, best_score, best_pose, best_idx
        

    # ========================================================================
    # STUDENT TODO #3: Main Scan Matching Function
    # ========================================================================

    def match(self, scan_ref: np.ndarray, scan_new: np.ndarray,
              initial_guess: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Match two scans using correlation-based grid search.

        Finds the relative pose that best aligns scan_new to scan_ref by
        exhaustively searching candidate poses around initial_guess.

        Args:
            scan_ref:      Nx2 reference scan points in local frame
            scan_new:      Mx2 new scan points in local frame
            initial_guess: Initial relative pose estimate [dx, dy, dtheta]

        Returns:
            best_pose:  Best relative pose [dx, dy, dtheta]
            covariance: 3x3 covariance matrix of the match
            score:      Normalised score in [0, 1] (0 if match rejected)

        Algorithm:
            1. ref_grid = _build_local_grid(scan_ref)
            2. Generate search grid with np.arange():
               x_values = [guess_x - search_x ... guess_x + search_x]
               y_values = [guess_y - search_y ... guess_y + search_y]
               theta_values = [guess_t - search_t ... guess_t + search_t]
            3. For each (ix, x), (iy, y), (it, theta):
               candidate = [x, y, theta]
               score = _score_alignment(ref_grid, scan_new, candidate)
               scores[(ix, iy, it)] = score
               Track best score + pose + indices
            4. normalised_score = best_score / len(scan_new)
            5. If normalised_score < min_score: return (initial_guess, default_cov, 0.0)
            6. covariance = _estimate_covariance_from_hessian(scores, best_idx, ...)
            7. Return (best_pose, covariance, normalised_score)

        Hints:
            - Store scores in a dict keyed by (ix, iy, it) integer indices
            - Handle empty scans (return early with score 0.0)
            - default_cov = np.diag([0.1, 0.1, 0.05])
            - The TODO section is ~12-15 lines of code
        """
        default_cov = np.diag([0.1, 0.1, 0.05])

        if len(scan_ref) == 0 or len(scan_new) == 0:
            return initial_guess.copy(), default_cov, 0.0

        # Step 1: Build local occupancy grid from the reference scan
        ref_grid = self._build_local_grid(scan_ref)

        # Step 2: Generate coarse search grid around initial guess
        dx_guess, dy_guess, dtheta_guess = initial_guess

        coarse_step_x = self.resolution_x * self.coarse_step_multiplier
        coarse_step_y = self.resolution_y * self.coarse_step_multiplier
        coarse_step_theta = self.resolution_theta * self.coarse_step_multiplier

        x_values = np.arange(
            dx_guess - self.search_x,
            dx_guess + self.search_x + coarse_step_x * 0.5,
            coarse_step_x
        )
        y_values = np.arange(
            dy_guess - self.search_y,
            dy_guess + self.search_y + coarse_step_y * 0.5,
            coarse_step_y
        )
        theta_values = np.arange(
            dtheta_guess - self.search_theta,
            dtheta_guess + self.search_theta + coarse_step_theta * 0.5,
            coarse_step_theta
        )

        # Step 3: Coarse search
        _, coarse_score, coarse_pose, _ = self._search_scores(
            lambda candidate: self._score_alignment(ref_grid, scan_new, candidate),
            x_values, y_values, theta_values
        )

        # Step 4: Fine search around coarse peak
        fine_window_x = self.resolution_x * self.fine_window_multiplier
        fine_window_y = self.resolution_y * self.fine_window_multiplier
        fine_window_theta = self.resolution_theta * self.fine_window_multiplier

        fine_x_values = np.arange(
            coarse_pose[0] - fine_window_x,
            coarse_pose[0] + fine_window_x + self.resolution_x * 0.5,
            self.resolution_x
        )
        fine_y_values = np.arange(
            coarse_pose[1] - fine_window_y,
            coarse_pose[1] + fine_window_y + self.resolution_y * 0.5,
            self.resolution_y
        )
        fine_theta_values = np.arange(
            coarse_pose[2] - fine_window_theta,
            coarse_pose[2] + fine_window_theta + self.resolution_theta * 0.5,
            self.resolution_theta
        )

        scores, best_score, best_pose, best_idx = self._search_scores(
            lambda candidate: self._score_alignment(ref_grid, scan_new, candidate),
            fine_x_values, fine_y_values, fine_theta_values
        )
        
        # Convert the raw overlap count into a fraction of scan points matched
        normalized_score = best_score / len(scan_new)

        # Reject weak matches
        if normalized_score < self.min_score:
            return initial_guess.copy(), default_cov, 0.0

        # Step 4: Estimate covariance from the Hessian of the score surface
        covariance = self._estimate_covariance_from_hessian(
            scores, best_idx,
            self.resolution_x, self.resolution_y, self.resolution_theta
        )

        return best_pose, covariance, normalized_score

    def match_to_map(self, occupancy_grid, scan_points: np.ndarray,
                     initial_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Align a scan to the global occupancy grid map.

        Args:
            occupancy_grid: OccupancyGrid instance (world frame)
            scan_points:    Nx2 array of scan points in local frame
            initial_pose:   Initial pose estimate [x, y, theta] in map frame

        Returns:
            best_pose:  Best pose [x, y, theta] in map frame
            covariance: 3x3 covariance matrix of the match
            score:      Normalised score in [0, 1] (0 if match rejected)
        """
        default_cov = np.diag([0.1, 0.1, 0.05])

        if len(scan_points) == 0:
            return initial_pose.copy(), default_cov, 0.0

        x_guess, y_guess, theta_guess = initial_pose

        coarse_step_x = self.resolution_x * self.coarse_step_multiplier
        coarse_step_y = self.resolution_y * self.coarse_step_multiplier
        coarse_step_theta = self.resolution_theta * self.coarse_step_multiplier

        x_values = np.arange(
            x_guess - self.search_x,
            x_guess + self.search_x + coarse_step_x * 0.5,
            coarse_step_x
        )
        y_values = np.arange(
            y_guess - self.search_y,
            y_guess + self.search_y + coarse_step_y * 0.5,
            coarse_step_y
        )
        theta_values = np.arange(
            theta_guess - self.search_theta,
            theta_guess + self.search_theta + coarse_step_theta * 0.5,
            coarse_step_theta
        )

        _, coarse_score, coarse_pose, _ = self._search_scores(
            lambda candidate: self._score_alignment_map(
                occupancy_grid, scan_points, candidate),
            x_values, y_values, theta_values
        )

        fine_window_x = self.resolution_x * self.fine_window_multiplier
        fine_window_y = self.resolution_y * self.fine_window_multiplier
        fine_window_theta = self.resolution_theta * self.fine_window_multiplier

        fine_x_values = np.arange(
            coarse_pose[0] - fine_window_x,
            coarse_pose[0] + fine_window_x + self.resolution_x * 0.5,
            self.resolution_x
        )
        fine_y_values = np.arange(
            coarse_pose[1] - fine_window_y,
            coarse_pose[1] + fine_window_y + self.resolution_y * 0.5,
            self.resolution_y
        )
        fine_theta_values = np.arange(
            coarse_pose[2] - fine_window_theta,
            coarse_pose[2] + fine_window_theta + self.resolution_theta * 0.5,
            self.resolution_theta
        )

        scores, best_score, best_pose, best_idx = self._search_scores(
            lambda candidate: self._score_alignment_map(
                occupancy_grid, scan_points, candidate),
            fine_x_values, fine_y_values, fine_theta_values
        )

        normalized_score = best_score / len(scan_points)

        if normalized_score < self.min_score:
            return initial_pose.copy(), default_cov, 0.0

        covariance = self._estimate_covariance_from_hessian(
            scores, best_idx,
            self.resolution_x, self.resolution_y, self.resolution_theta
        )

        return best_pose, covariance, normalized_score

    # ========================================================================
    # PROVIDED: Estimate Covariance from Hessian (do not modify)
    # ========================================================================
    #
    # This function estimates how confident the scan match is by computing
    # the Hessian (matrix of second derivatives) of the score surface at
    # the peak. Sharp peak → small covariance (confident). Flat peak →
    # large covariance (uncertain). See Lecture 08, Section 8.4.
    #
    # The covariance is: Sigma = (-H)^{-1}
    # If -H is not positive definite, the match geometry is degenerate
    # (e.g., a corridor) and we fall back to a conservative default.

    def _estimate_covariance_from_hessian(self,
                                          scores: dict,
                                          best_idx: Tuple[int, int, int],
                                          step_x: float,
                                          step_y: float,
                                          step_theta: float) -> np.ndarray:
        """
        Estimate match covariance from the Hessian of the score surface.

        Args:
            scores:     Dict mapping (ix, iy, it) tuples to score values
            best_idx:   (ix, iy, it) of the peak score
            step_x:     Search step in x (metres)
            step_y:     Search step in y (metres)
            step_theta: Search step in theta (radians)

        Returns:
            3x3 covariance matrix
        """
        default_cov = np.diag([0.1, 0.1, 0.05])

        f0 = scores.get(best_idx, 0.0)
        if f0 == 0.0:
            return default_cov

        steps = [step_x, step_y, step_theta]
        H = np.zeros((3, 3))

        # Diagonal elements: H[i,i] = (f(+step) - 2*f0 + f(-step)) / step_i^2
        for i in range(3):
            idx_plus = list(best_idx)
            idx_minus = list(best_idx)
            idx_plus[i] += 1
            idx_minus[i] -= 1

            f_plus = scores.get(tuple(idx_plus), 0.0)
            f_minus = scores.get(tuple(idx_minus), 0.0)

            H[i, i] = (f_plus - 2.0 * f0 + f_minus) / (steps[i] ** 2)

        # Off-diagonal elements: H[i,j] = (f++ - f+- - f-+ + f--) / (4*si*sj)
        for i in range(3):
            for j in range(i + 1, 3):
                idx_pp = list(best_idx)
                idx_pm = list(best_idx)
                idx_mp = list(best_idx)
                idx_mm = list(best_idx)

                idx_pp[i] += 1; idx_pp[j] += 1
                idx_pm[i] += 1; idx_pm[j] -= 1
                idx_mp[i] -= 1; idx_mp[j] += 1
                idx_mm[i] -= 1; idx_mm[j] -= 1

                f_pp = scores.get(tuple(idx_pp), 0.0)
                f_pm = scores.get(tuple(idx_pm), 0.0)
                f_mp = scores.get(tuple(idx_mp), 0.0)
                f_mm = scores.get(tuple(idx_mm), 0.0)

                H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * steps[i] * steps[j])
                H[j, i] = H[i, j]

        # Covariance = (-H)^{-1}  (negate because H is concave at a maximum)
        neg_H = -H

        # Check positive definiteness
        eigenvalues = np.linalg.eigvalsh(neg_H)
        if np.any(eigenvalues <= 1e-6):
            return default_cov

        try:
            covariance = np.linalg.inv(neg_H)
        except np.linalg.LinAlgError:
            return default_cov

        # Sanity check
        cov_eigenvalues = np.linalg.eigvalsh(covariance)
        if np.any(cov_eigenvalues <= 0):
            return default_cov

        return covariance


# ============================================================================
# Helper function (provided — do not modify)
# ============================================================================

def scans_from_ranges(ranges: np.ndarray, angle_min: float,
                      angle_increment: float, min_range: float = 0.1,
                      max_range: float = 12.0,
                      lidar_yaw_offset: float = 0.0) -> np.ndarray:
    """
    Convert laser scan ranges to (x, y) points in the robot's local frame.

    Converts raw LaserScan ranges into the Nx2 point arrays that the ScanMatcher expects.

    Args:
        ranges:           Array of laser range measurements (metres)
        angle_min:        Start angle of scan (radians)
        angle_increment:  Angular step between beams (radians)
        min_range:        Minimum valid range (metres)
        max_range:        Maximum valid range (metres)
        lidar_yaw_offset: Lidar yaw rotation relative to base_link (radians)

    Returns:
        Nx2 array of (x, y) points in robot's local frame
    """
    points = []
    for i, r in enumerate(ranges):
        if np.isnan(r) or r < min_range or r > max_range:
            continue
        beam_angle = lidar_yaw_offset + angle_min + i * angle_increment
        x = r * np.cos(beam_angle)
        y = r * np.sin(beam_angle)
        points.append([x, y])

    if len(points) == 0:
        return np.empty((0, 2))
    return np.array(points)