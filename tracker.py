import numpy as np
import cv2
from collections import deque

class LaneTracker(object):
    """
    Tracks the lane in a series of consecutive frames.
    """

    def __init__(self, first_frame, n_windows=9):
        """
        Initialises a tracker object.
        Parameters
        ----------
        first_frame     : First frame of the frame series. We use it to get dimensions and initialise values.
        n_windows       : Number of windows we use to track each lane edge.
        """
        (self.h, self.w, _) = first_frame.shape
        self.win_n = n_windows
        self.left = None
        self.right = None
        self.l_windows = []
        self.r_windows = []
        self.initialize_lines(first_frame)

    def initialize_lines(self, frame):
        """
        Finds starting points for left and right lines (e.g. lane edges) and initialises Window and Line objects.
        Parameters
        ----------
        frame   : Frame to scan for lane edges.
        """
        # Take a histogram of the bottom half of the image
        edges = get_edges(frame)
        (flat_edges, _) = flatten_perspective(edges)
        histogram = np.sum(flat_edges[int(self.h / 2):, :], axis=0)

        nonzero = flat_edges.nonzero()
        # Create empty lists to receive left and right lane pixel indices
        l_indices = np.empty([0], dtype=np.int)
        r_indices = np.empty([0], dtype=np.int)
        window_height = int(self.h / self.win_n)

        for i in range(self.win_n):
            l_window = Window(
                y1=self.h - (i + 1) * window_height,
                y2=self.h - i * window_height,
                x=self.l_windows[-1].x if len(self.l_windows) > 0 else np.argmax(histogram[:self.w // 2])
            )
            r_window = Window(
                y1=self.h - (i + 1) * window_height,
                y2=self.h - i * window_height,
                x=self.r_windows[-1].x if len(self.r_windows) > 0 else np.argmax(histogram[self.w // 2:]) + self.w // 2
            )
            # Append nonzero indices in the window boundary to the lists
            l_indices = np.append(l_indices, l_window.pixels_in(nonzero), axis=0)
            r_indices = np.append(r_indices, r_window.pixels_in(nonzero), axis=0)
            self.l_windows.append(l_window)
            self.r_windows.append(r_window)
        self.left = Line(x=nonzero[1][l_indices], y=nonzero[0][l_indices], h=self.h, w=self.w)
        self.right = Line(x=nonzero[1][r_indices], y=nonzero[0][r_indices], h=self.h, w=self.w)

    def scan_frame_with_windows(self, frame, windows):
        """
        Scans a frame using initialised windows in an attempt to track the lane edges.
        Parameters
        ----------
        frame   : New frame
        windows : Array of windows to use for scanning the frame.
        Returns
        -------
        A tuple of arrays containing coordinates of points found in the specified windows.
        """
        indices = np.empty([0], dtype=np.int)
        nonzero = frame.nonzero()
        window_x = None
        for window in windows:
            indices = np.append(indices, window.pixels_in(nonzero, window_x), axis=0)
            window_x = window.mean_x
        return (nonzero[1][indices], nonzero[0][indices])

    def process(self, frame, draw_lane=True, draw_statistics=True):
        """
        Performs a full lane tracking pipeline on a frame.
        Parameters
        ----------
        frame               : New frame to process.
        draw_lane           : Flag indicating if we need to draw the lane on top of the frame.
        draw_statistics     : Flag indicating if we need to render the debug information on top of the frame.
        Returns
        -------
        Resulting frame.
        """
        edges = get_edges(frame)
        (flat_edges, unwarp_matrix) = flatten_perspective(edges)
        (l_x, l_y) = self.scan_frame_with_windows(flat_edges, self.l_windows)
        self.left.process_points(l_x, l_y)
        (r_x, r_y) = self.scan_frame_with_windows(flat_edges, self.r_windows)
        self.right.process_points(r_x, r_y)

        if draw_statistics:
            edges = get_edges(frame, separate_channels=True)
            debug_overlay = self.draw_debug_overlay(flatten_perspective(edges)[0])
            top_overlay = self.draw_lane_overlay(flatten_perspective(frame)[0])
            debug_overlay = cv2.resize(debug_overlay, (0, 0), fx=0.3, fy=0.3)
            top_overlay = cv2.resize(top_overlay, (0, 0), fx=0.3, fy=0.3)
            frame[:250, :, :] = frame[:250, :, :] * .4
            (h, w, _) = debug_overlay.shape
            frame[20:20 + h, 20:20 + w, :] = debug_overlay
            frame[20:20 + h, 20 + 20 + w:20 + 20 + w + w, :] = top_overlay
            text_x = 20 + 20 + w + w + 20
            self.draw_text(frame, 'Radius of curvature:  {} m'.format(self.radius_of_curvature()), text_x, 80)
            self.draw_text(frame, 'Distance (left):       {:.1f} m'.format(self.left.camera_distance()), text_x, 140)
            self.draw_text(frame, 'Distance (right):      {:.1f} m'.format(self.right.camera_distance()), text_x, 200)

        if draw_lane:
            frame = self.draw_lane_overlay(frame, unwarp_matrix)

        return frame

    def draw_text(self, frame, text, x, y):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)

    def draw_debug_overlay(self, binary, lines=True, windows=True):
        """
        Draws an overlay with debugging information on a bird's-eye view of the road (e.g. after applying perspective
        transform).
        Parameters
        ----------
        binary  : Frame to overlay.
        lines   : Flag indicating if we need to draw lines.
        windows : Flag indicating if we need to draw windows.
        Returns
        -------
        Frame with an debug information overlay.
        """
        if len(binary.shape) == 2:
            image = np.dstack((binary, binary, binary))
        else:
            image = binary
        if windows:
            for window in self.l_windows:
                coordinates = window.coordinates()
                cv2.rectangle(image, coordinates[0], coordinates[1], (1., 1., 0), 2)
            for window in self.r_windows:
                coordinates = window.coordinates()
                cv2.rectangle(image, coordinates[0], coordinates[1], (1., 1., 0), 2)
        if lines:
            cv2.polylines(image, [self.left.get_points()], False, (1., 0, 0), 2)
            cv2.polylines(image, [self.right.get_points()], False, (1., 0, 0), 2)
        return image * 255

    def draw_lane_overlay(self, image, unwarp_matrix=None):
        """
        Draws an overlay with tracked lane applying perspective unwarp to project it on the original frame.
        Parameters
        ----------
        image           : Original frame.
        unwarp_matrix   : Transformation matrix to unwarp the bird's eye view to initial frame. Defaults to `None` (in
        which case no unwarping is applied).
        Returns
        -------
        Frame with a lane overlay.
        """
        # Create an image to draw the lines on
        overlay = np.zeros_like(image).astype(np.uint8)
        points = np.vstack((self.left.get_points(), np.flipud(self.right.get_points())))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(overlay, [points], (0, 255, 0))
        if unwarp_matrix is not None:
            # Warp the blank back to original image space using inverse perspective matrix (Minv)
            overlay = cv2.warpPerspective(overlay, unwarp_matrix, (image.shape[1], image.shape[0]))
        # Combine the result with the original image
        return cv2.addWeighted(image, 1, overlay, 0.3, 0)

    def radius_of_curvature(self):
        """
        Calculates radius of the lane curvature by averaging curvature of the edge lines.
        Returns
        -------
        Radius of the lane curvature in meters.
        """
        return int(np.average([self.left.radius_of_curvature(), self.right.radius_of_curvature()]))


class Window(object):
    """
    Represents a scanning window used to detect points likely to represent lane edge lines.
    """

    def __init__(self, y1, y2, x, m=100, tolerance=50):
        """
        Initialises a window object.
        Parameters
        ----------
        y1          : Top y axis coordinate of the window rect.
        y2          : Bottom y axis coordinate of the window rect.
        x           : X coordinate of the center of the window rect
        m           : X axis span, e.g. window rect width would be m*2..
        tolerance   : Min number of pixels we need to detect within a window in order to adjust its x coordinate.
        """
        self.x = x
        self.mean_x = x
        self.y1 = y1
        self.y2 = y2
        self.m = m
        self.tolerance = tolerance

    def pixels_in(self, nonzero, x=None):
        """
        Returns indices of the pixels in `nonzero` that are located within this window.
        Notes
        -----
        Since this looks a bit tricky, I will go into a bit more detail. `nonzero` contains two arrays of coordinates
        of non-zero pixels. Say, there were 50 non-zero pixels in the image and `nonzero` would contain two arrays of
        shape (50, ) with x and y coordinates of those pixels respectively. What we return here is a array of indices
        within those 50 that are located inside this window. Basically the result would be a 1-dimensional array of
        ints in the [0, 49] range with a size of less than 50.
        Parameters
        ----------
        nonzero : Coordinates of the non-zero pixels in the image.
        Returns
        -------
        Array of indices of the pixels within this window.
        """
        if x is not None:
            self.x = x
        win_indices = (
            (nonzero[0] >= self.y1) & (nonzero[0] < self.y2) &
            (nonzero[1] >= self.x - self.m) & (nonzero[1] < self.x + self.m)
        ).nonzero()[0]
        if len(win_indices) > self.tolerance:
            self.mean_x = np.int(np.mean(nonzero[1][win_indices]))
        else:
            self.mean_x = self.x

        return win_indices

    def coordinates(self):
        """
        Returns coordinates of the bounding rect.
        Returns
        -------
        Tuple of ((x1, y1), (x2, y2))
        """
        return ((self.x - self.m, self.y1), (self.x + self.m, self.y2))


class Line(object):
    """
    Represents a single lane edge line.
    """

    def __init__(self, x, y, h, w):
        """
        Initialises a line object by fitting a 2nd degree polynomial to provided line points.
        Parameters
        ----------
        x   : Array of x coordinates for pixels representing a line.
        y   : Array of y coordinates for pixels representing a line.
        h   : Image height in pixels.
        w   : Image width in pixels.
        """
        # polynomial coefficients for the most recent fit
        self.h = h
        self.w = w
        self.frame_impact = 0
        self.coefficients = deque(maxlen=5)
        self.process_points(x, y)

    def process_points(self, x, y):
        """
        Fits a polynomial if there is enough points to try and approximate a line and updates a queue of coefficients.
        Parameters
        ----------
        x   : Array of x coordinates for pixels representing a line.
        y   : Array of y coordinates for pixels representing a line.
        """
        enough_points = len(y) > 0 and np.max(y) - np.min(y) > self.h * .625
        if enough_points or len(self.coefficients) == 0:
            self.fit(x, y)

    def get_points(self):
        """
        Generates points of the current best fit line.
        Returns
        -------
        Array with x and y coordinates of pixels representing
        current best approximation of a line.
        """
        y = np.linspace(0, self.h - 1, self.h)
        current_fit = self.averaged_fit()
        return np.stack((
            current_fit[0] * y ** 2 + current_fit[1] * y + current_fit[2],
            y
        )).astype(np.int).T

    def averaged_fit(self):
        """
        Returns coefficients for a line averaged across last 5 points' updates.
        Returns
        -------
        Array of polynomial coefficients.
        """
        return np.array(self.coefficients).mean(axis=0)

    def fit(self, x, y):
        """
        Fits a 2nd degree polynomial to provided points and returns its coefficients.
        Parameters
        ----------
        x   : Array of x coordinates for pixels representing a line.
        y   : Array of y coordinates for pixels representing a line.
        """
        self.coefficients.append(np.polyfit(y, x, 2))

    def radius_of_curvature(self):
        """
        Calculates radius of curvature of the line in real world coordinate system (e.g. meters), assuming there are
        27 meters for 720 pixels for y axis and 3.7 meters for 700 pixels for x axis.
        Returns
        -------
        Estimated radius of curvature in meters.
        """
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 27 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        # Fit new polynomials to x,y in world space
        points = self.get_points()
        y = points[:, 1]
        x = points[:, 0]
        fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
        return int(((1 + (2 * fit_cr[0] * 720 * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0]))

    def camera_distance(self):
        """
        Calculates distance to camera in real world coordinate system (e.g. meters), assuming there are 3.7 meters for
        700 pixels for x axis.
        Returns
        -------
        Estimated distance to camera in meters.
        """
        points = self.get_points()
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        x = points[np.max(points[:, 1])][0]
        return np.absolute((self.w // 2 - x) * xm_per_pix)


def flatten_perspective(image):
    """
    Warps the image from the vehicle front-facing camera mapping hte road to a bird view perspective.
    Parameters
    ----------
    image       : Image from the vehicle front-facing camera.
    Returns
    -------
    Warped image.
    """
    # Get image dimensions
    (h, w) = (image.shape[0], image.shape[1])
    # Define source points
    source = np.float32([[w // 2 - 76, h * .625], [w // 2 + 76, h * .625], [-100, h], [w + 100, h]])
    # Define corresponding destination points
    destination = np.float32([[100, 0], [w - 100, 0], [100, h], [w - 100, h]])
    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    unwarp_matrix = cv2.getPerspectiveTransform(destination, source)
    return (cv2.warpPerspective(image, transform_matrix, (w, h)), unwarp_matrix)

def gradient_abs_value_mask(image, sobel_kernel=3, axis='x', threshold=(0, 255)):
    """
    Masks the image based on gradient absolute value.
    Parameters
    ----------
    image           : Image to mask.
    sobel_kernel    : Kernel of the Sobel gradient operation.
    axis            : Axis of the gradient, 'x' or 'y'.
    threshold       : Value threshold for it to make it to appear in the mask.
    Returns
    -------
    Image mask with 1s in activations and 0 in other pixels.
    """
    # Take the absolute value of derivative in x or y given orient = 'x' or 'y'
    if axis == 'x':
        sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if axis == 'y':
        sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    sobel = np.uint8(255 * sobel / np.max(sobel))
    # Create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
    mask = np.zeros_like(sobel)
    # Return this mask as your binary_output image
    mask[(sobel >= threshold[0]) & (sobel <= threshold[1])] = 1
    return mask

def gradient_magnitude_mask(image, sobel_kernel=3, threshold=(0, 255)):
    """
    Masks the image based on gradient magnitude.
    Parameters
    ----------
    image           : Image to mask.
    sobel_kernel    : Kernel of the Sobel gradient operation.
    threshold       : Magnitude threshold for it to make it to appear in the mask.
    Returns
    -------
    Image mask with 1s in activations and 0 in other pixels.
    """
    # Take the gradient in x and y separately
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the magnitude
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    magnitude = (magnitude * 255 / np.max(magnitude)).astype(np.uint8)
    # Create a binary mask where mag thresholds are met
    mask = np.zeros_like(magnitude)
    mask[(magnitude >= threshold[0]) & (magnitude <= threshold[1])] = 1
    # Return this mask as your binary_output image
    return mask

def gradient_direction_mask(image, sobel_kernel=3, threshold=(0, np.pi / 2)):
    """
    Masks the image based on gradient direction.
    Parameters
    ----------
    image           : Image to mask.
    sobel_kernel    : Kernel of the Sobel gradient operation.
    threshold       : Direction threshold for it to make it to appear in the mask.
    Returns
    -------
    Image mask with 1s in activations and 0 in other pixels.
    """
    # Take the gradient in x and y separately
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the x and y gradients and calculate the direction of the gradient
    direction = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
    # Create a binary mask where direction thresholds are met
    mask = np.zeros_like(direction)
    # Return this mask as your binary_output image
    mask[(direction >= threshold[0]) & (direction <= threshold[1])] = 1
    return mask

def color_threshold_mask(image, threshold=(0, 255)):
    """
    Masks the image based on color intensity.
    Parameters
    ----------
    image           : Image to mask.
    threshold       : Color intensity threshold.
    Returns
    -------
    Image mask with 1s in activations and 0 in other pixels.
    """
    mask = np.zeros_like(image)
    mask[(image > threshold[0]) & (image <= threshold[1])] = 1
    return mask

def get_edges(image, separate_channels=False):
    """
    Masks the image based on a composition of edge detectors: gradient value,
    gradient magnitude, gradient direction and color.
    Parameters
    ----------
    image               : Image to mask.
    separate_channels   : Flag indicating if we need to put masks in different color channels.
    Returns
    -------
    Image mask with 1s in activations and 0 in other pixels.
    """
    # Convert to HLS color space and separate required channel
    hls = cv2.cvtColor(np.copy(image), cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:, :, 2]
    # Get a combination of all gradient thresholding masks
    gradient_x = gradient_abs_value_mask(s_channel, axis='x', sobel_kernel=3, threshold=(20, 100))
    gradient_y = gradient_abs_value_mask(s_channel, axis='y', sobel_kernel=3, threshold=(20, 100))
    magnitude = gradient_magnitude_mask(s_channel, sobel_kernel=3, threshold=(20, 100))
    direction = gradient_direction_mask(s_channel, sobel_kernel=3, threshold=(0.7, 1.3))
    gradient_mask = np.zeros_like(s_channel)
    gradient_mask[((gradient_x == 1) & (gradient_y == 1)) | ((magnitude == 1) & (direction == 1))] = 1
    # Get a color thresholding mask
    color_mask = color_threshold_mask(s_channel, threshold=(170, 255))

    if separate_channels:
        return np.dstack((np.zeros_like(s_channel), gradient_mask, color_mask))
    else:
        mask = np.zeros_like(gradient_mask)
        mask[(gradient_mask == 1) | (color_mask == 1)] = 1
        return mask