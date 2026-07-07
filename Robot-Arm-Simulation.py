import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, RadioButtons, Slider
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon, Circle
from matplotlib.lines import Line2D
import csv

matplotlib.use("tkAgg")


# ==========================================
# Pre-Launch: Ask User for Joint Limits
# ==========================================
def get_joint_limits():
    print("=====================================================")
    print("   Robot Joint Limits Configuration                  ")
    print("=====================================================")

    # NEW: Quick default option
    choice = input("Use default limits (-180 to 180) for all joints? (Y/n): ").strip().lower()
    if choice == '' or choice == 'y':
        print("Using defaults: -180 to 180 degrees.")
        return [(-np.pi, np.pi) for _ in range(3)]

    limits = []
    for i in range(3):
        try:
            s_min = input(f"Joint {i + 1} Lower Bound [-180]: ")
            min_deg = float(s_min) if s_min.strip() else -180.0

            s_max = input(f"Joint {i + 1} Upper Bound [180]: ")
            max_deg = float(s_max) if s_max.strip() else 180.0

            if min_deg > max_deg:
                min_deg, max_deg = max_deg, min_deg

            limits.append((np.deg2rad(min_deg), np.deg2rad(max_deg)))
        except ValueError:
            print("Invalid input! Defaulting to -180 to 180 for this joint.")
            limits.append((-np.pi, np.pi))
    return limits


joint_limits = get_joint_limits()

# ==========================================
# Global Parameters & State
# ==========================================
dt = 0.05
current_time = 0.0
tau = 0.15  # Time constant for motor dynamics

# Physical Constraints (Strength Enhancement)
joint_speed_limit = np.deg2rad(90.0)  # Max joint rate in rad/s (90 deg/s)

# Store defaults for the reset button
DEFAULT_THETA = np.array([np.pi / 4, -np.pi / 4, -np.pi / 4])
DEFAULT_L = [2.0, 1.5, 1.0]

theta_current = np.copy(DEFAULT_THETA)
theta_dot_actual = np.array([0.0, 0.0, 0.0])

current_mode = "Auto: Circle"
frame_idx = 0
trace_history = []
recorded_data = []
jog_directions = [0, 0, 0]

# Vibrant colors for visibility
COLOR_LOW = '#00FF00'  # Bright Green
COLOR_MED = '#FFA500'  # Vivid Orange
COLOR_HIGH = '#FF0000'  # Bright Red


# ==========================================
# Aesthetic Update: Custom Drawing Functions
# ==========================================
def create_link_patch(width, color):
    """Creates an empty polygon patch with given aesthetic properties."""
    polygon = Polygon([[0, 0], [0, 0]], closed=True, color=color, alpha=0.9, zorder=10)
    return polygon


def create_base_patch():
    """Creates the patches that represent the substantial robot base."""
    base_circle = Circle((0, 0), radius=0.3, color='black', alpha=1.0, zorder=5)
    stand_poly = Polygon([[-0.2, -0.1], [0.2, -0.1], [0.2, 0.1], [-0.2, 0.1]],
                         closed=True, color='#444444', alpha=1.0, zorder=5)
    return base_circle, stand_poly


def update_link_patches(patches, points, widths):
    """Updates the vertices of link polygon patches based on joint positions."""
    for i in range(len(patches)):
        p1 = points[:, i]
        p2 = points[:, i + 1]

        delta = p2 - p1
        length = np.linalg.norm(delta)
        angle = np.arctan2(delta[1], delta[0])

        width = widths[i]
        verts = np.array([
            [0, -width / 2],
            [length, -width / 2],
            [length, width / 2],
            [0, width / 2]
        ])

        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        transformed_verts = (R @ verts.T).T + p1
        patches[i].set_xy(transformed_verts)


# ==========================================
# Kinematic and Jacobian functions
# ==========================================
def forward_kinematics(theta, L):
    th1, th2, th3 = theta
    l1, l2, l3 = L
    x0, y0 = 0.0, 0.0
    x1, y1 = l1 * np.cos(th1), l1 * np.sin(th1)
    x2, y2 = x1 + l2 * np.cos(th1 + th2), y1 + l2 * np.sin(th1 + th2)
    x3, y3 = x2 + l3 * np.cos(th1 + th2 + th3), y2 + l3 * np.sin(th1 + th2 + th3)
    return np.array([[x0, x1, x2, x3], [y0, y1, y2, y3]])


def get_jacobian(theta, L):
    th1, th2, th3 = theta
    l1, l2, l3 = L
    s1, c1 = np.sin(th1), np.cos(th1)
    s12, c12 = np.sin(th1 + th2), np.cos(th1 + th2)
    s123, c123 = np.sin(th1 + th2 + th3), np.cos(th1 + th2 + th3)

    J = np.zeros((2, 3))
    J[0, 0] = -l1 * s1 - l2 * s12 - l3 * s123
    J[0, 1] = -l2 * s12 - l3 * s123
    J[0, 2] = -l3 * s123
    J[1, 0] = l1 * c1 + l2 * c12 + l3 * c123
    J[1, 1] = l2 * c12 + l3 * c123
    J[1, 2] = l3 * c123
    return J


# ==========================================
# Trajectory Generation
# ==========================================
def generate_trajectory(mode):
    path_x, path_y = [], []
    if mode == "line":
        start_pt, end_pt = np.array([2.0, 2.0]), np.array([-2.0, 3.0])
        steps = 150
        for i in range(steps):
            t = i / (steps - 1)
            pt = start_pt + t * (end_pt - start_pt)
            path_x.append(pt[0])
            path_y.append(pt[1])
    elif mode == "circle":
        center, R = np.array([1.5, 1.5]), 1.0
        steps = 200
        for i in range(steps):
            th = 2 * np.pi * i / steps
            path_x.append(center[0] + R * np.cos(th))
            path_y.append(center[1] + R * np.sin(th))
    return np.array(path_x), np.array(path_y)


circle_x, circle_y = generate_trajectory("circle")
line_x, line_y = generate_trajectory("line")

# ==========================================
# GUI Setup
# ==========================================
fig, ax = plt.subplots(figsize=(14, 8))
plt.subplots_adjust(left=0.05, right=0.45, bottom=0.05)

ax.set_facecolor('#EAEAF2')
ax.grid(True, linestyle='-', color='white', alpha=1.0, zorder=0)
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_aspect('equal')
ax.set_title("3-Link Robot Arm")

# --- Robot Visual Elements ---
base_circle, base_stand = create_base_patch()
ax.add_patch(base_stand)
ax.add_patch(base_circle)

LINK_COLORS = ['#CCCCCC', '#999999', '#666666']
LINK_WIDTHS = [0.25, 0.2, 0.15]
link_patches = [create_link_patch(LINK_WIDTHS[i], LINK_COLORS[i]) for i in range(3)]
for patch in link_patches:
    ax.add_patch(patch)

JOINT_COLOR = 'black'
JOINT_SIZE_BASE = 0.08
joint_circles = [Circle((0, 0), radius=JOINT_SIZE_BASE, color=JOINT_COLOR, zorder=15) for _ in range(4)]
for circle in joint_circles:
    ax.add_patch(circle)

target_path_plot, = ax.plot([], [], 'r--', label='Target Path', zorder=20)
reticle_plot, = ax.plot([], [], 'rx', markersize=8, label='Target Reticle', zorder=25)

trace_coll = LineCollection([], linewidths=3, zorder=10)
ax.add_collection(trace_coll)

info_text = fig.text(0.48, 0.95, '', fontsize=10, verticalalignment='top', family='monospace',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor='#f8f9fa', edgecolor='black', alpha=0.9))

# --- NEW: Improved Legend ---
legend_elements = [
    Line2D([0], [0], color='r', linestyle='--', label='Target Path'),
    Line2D([0], [0], marker='x', color='r', linestyle='None', label='Target Reticle', markersize=8),
    Line2D([0], [0], marker='o', color='black', linestyle='None', label='Robot Joints', markersize=8),
    Line2D([0], [0], marker='o', color=COLOR_LOW, linestyle='None', label='EE: Low Pressure', markersize=8),
    Line2D([0], [0], marker='o', color=COLOR_MED, linestyle='None', label='EE: Med Pressure', markersize=8),
    Line2D([0], [0], marker='o', color=COLOR_HIGH, linestyle='None', label='EE: High Pressure', markersize=8)
]
ax.legend(handles=legend_elements, loc='lower left', framealpha=0.9, edgecolor='black')

# Initialize visuals with default pose
points_init = forward_kinematics(theta_current, DEFAULT_L)
update_link_patches(link_patches, points_init, LINK_WIDTHS)
for i, circle in enumerate(joint_circles):
    circle.center = points_init[:, i]

# --- Widgets ---
WIDGET_START_X = 0.72
WIDGET_START_Y = 0.88
WIDGET_WIDTH = 0.22
WIDGET_HEIGHT = 0.025
WIDGET_SPACING = 0.04


def next_widget_y(curr_y): return curr_y - WIDGET_SPACING


curr_y = WIDGET_START_Y

# E-STOP BUTTON (Bonus Feature)
ax_estop = plt.axes([WIDGET_START_X, curr_y, WIDGET_WIDTH, 0.05])
btn_estop = Button(ax_estop, 'EMERGENCY STOP', color='#FF3333', hovercolor='#FF0000')
btn_estop.label.set_color('white')
btn_estop.label.set_fontweight('bold')


def estop_callback(event):
    global current_mode, theta_dot_actual, jog_directions
    print("\n!!! EMERGENCY STOP ACTIVATED !!!\n")
    current_mode = 'Manual'
    radio_mode.set_active(2)  # Update radio dial to Manual
    theta_dot_actual = np.array([0.0, 0.0, 0.0])
    jog_directions = [0, 0, 0]


btn_estop.on_clicked(estop_callback)
curr_y -= 0.08  # Space after big button

# Mode Radio Buttons
ax_radio = plt.axes([WIDGET_START_X, curr_y - 0.1, WIDGET_WIDTH, 0.12])
radio_mode = RadioButtons(ax_radio, ('Auto: Circle', 'Auto: Line', 'Manual'))


def mode_changed(label):
    global current_mode, frame_idx, trace_history
    current_mode = label
    frame_idx = 0
    trace_history = []
    trace_coll.set_segments([])
    if label == 'Auto: Circle':
        target_path_plot.set_data(circle_x, circle_y)
    elif label == 'Auto: Line':
        target_path_plot.set_data(line_x, line_y)
    else:
        target_path_plot.set_data([], [])
        reticle_plot.set_data([], [])


radio_mode.on_clicked(mode_changed)
target_path_plot.set_data(circle_x, circle_y)

curr_y -= (0.1 + WIDGET_SPACING)

ax_l1 = plt.axes([WIDGET_START_X, curr_y, WIDGET_WIDTH, WIDGET_HEIGHT])
slider_l1 = Slider(ax_l1, 'L1', 0.5, 4.0, valinit=DEFAULT_L[0])

curr_y = next_widget_y(curr_y)
ax_l2 = plt.axes([WIDGET_START_X, curr_y, WIDGET_WIDTH, WIDGET_HEIGHT])
slider_l2 = Slider(ax_l2, 'L2', 0.5, 4.0, valinit=DEFAULT_L[1])

curr_y = next_widget_y(curr_y)
ax_l3 = plt.axes([WIDGET_START_X, curr_y, WIDGET_WIDTH, WIDGET_HEIGHT])
slider_l3 = Slider(ax_l3, 'L3', 0.5, 4.0, valinit=DEFAULT_L[2])

curr_y = next_widget_y(curr_y)
ax_auto_speed = plt.axes([WIDGET_START_X, curr_y, WIDGET_WIDTH, WIDGET_HEIGHT])
slider_auto_speed = Slider(ax_auto_speed, 'Auto Vel', 0.5, 10.0, valinit=3.0)

curr_y = next_widget_y(curr_y)
ax_velocity_manual = plt.axes([WIDGET_START_X, curr_y, WIDGET_WIDTH, WIDGET_HEIGHT])
slider_velocity_manual = Slider(ax_velocity_manual, 'Man Vel', 0.1, 2.0, valinit=0.5)

curr_y = next_widget_y(curr_y)
ax_damping = plt.axes([WIDGET_START_X, curr_y, WIDGET_WIDTH, WIDGET_HEIGHT])
slider_damping = Slider(ax_damping, 'DLS Damp', 0.05, 0.5, valinit=0.2)

curr_y = next_widget_y(curr_y)
ax_pressure = plt.axes([WIDGET_START_X, curr_y, WIDGET_WIDTH, WIDGET_HEIGHT])
slider_pressure = Slider(ax_pressure, 'Pressure', 1, 3, valinit=1, valstep=1)

curr_y = next_widget_y(curr_y)
curr_y -= 0.02  # Spacer

ax_save = plt.axes([WIDGET_START_X, curr_y, WIDGET_WIDTH, 0.035])
btn_save = Button(ax_save, 'Save Data to CSV', color='lightgreen')


def save_to_csv(event):
    try:
        with open('robot_output.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['Time(s)', 'X', 'Y', 'Th1(rad)', 'Th2(rad)', 'Th3(rad)', 'Vx', 'Vy', 'Vz', 'Wx', 'Wy', 'Wz'])
            for row in recorded_data:
                writer.writerow(row)
        print(">>> SUCCESS: Data saved to 'robot_output.csv'")
    except Exception as e:
        print(f"Error saving file: {e}")


btn_save.on_clicked(save_to_csv)

curr_y -= 0.045

ax_reset = plt.axes([WIDGET_START_X, curr_y, WIDGET_WIDTH, 0.035])
btn_reset = Button(ax_reset, 'Reset to Defaults', color='lightgray')


def reset_robot(event):
    global theta_current, theta_dot_actual, current_time, trace_history, frame_idx
    theta_current = np.copy(DEFAULT_THETA)
    theta_dot_actual = np.array([0.0, 0.0, 0.0])
    current_time = 0.0
    trace_history = []
    frame_idx = 0
    trace_coll.set_segments([])

    slider_l1.set_val(DEFAULT_L[0])
    slider_l2.set_val(DEFAULT_L[1])
    slider_l3.set_val(DEFAULT_L[2])
    slider_auto_speed.set_val(3.0)
    slider_velocity_manual.set_val(0.5)
    slider_damping.set_val(0.2)
    slider_pressure.set_val(1)

    if current_mode == 'Manual':
        radio_mode.set_active(0)


btn_reset.on_clicked(reset_robot)

curr_y -= 0.05

# Manual Control Buttons
button_axes = [
    plt.axes([WIDGET_START_X, curr_y, 0.10, 0.035]), plt.axes([WIDGET_START_X + 0.12, curr_y, 0.10, 0.035]),
    plt.axes([WIDGET_START_X, curr_y - 0.04, 0.10, 0.035]),
    plt.axes([WIDGET_START_X + 0.12, curr_y - 0.04, 0.10, 0.035]),
    plt.axes([WIDGET_START_X, curr_y - 0.08, 0.10, 0.035]),
    plt.axes([WIDGET_START_X + 0.12, curr_y - 0.08, 0.10, 0.035])
]

buttons = [
    Button(button_axes[0], 'Th1 -'), Button(button_axes[1], 'Th1 +'),
    Button(button_axes[2], 'Th2 -'), Button(button_axes[3], 'Th2 +'),
    Button(button_axes[4], 'Th3 -'), Button(button_axes[5], 'Th3 +')
]


def on_press(event):
    for i, ax_btn in enumerate(button_axes):
        if event.inaxes == ax_btn:
            joint = i // 2
            direction = -1 if i % 2 == 0 else 1
            jog_directions[joint] = direction


def on_release(event):
    for i in range(3):
        jog_directions[i] = 0


fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)


# ==========================================
# Main Update Loop
# ==========================================
def update(frame):
    global theta_current, theta_dot_actual, frame_idx, trace_history, current_time

    current_time += dt
    L_vals = [slider_l1.val, slider_l2.val, slider_l3.val]
    theta_dot_target = np.array([0.0, 0.0, 0.0])

    if current_mode.startswith('Auto'):
        t_x, t_y = (circle_x, circle_y) if current_mode == 'Auto: Circle' else (line_x, line_y)

        target_pos = np.array([t_x[frame_idx], t_y[frame_idx]])
        reticle_plot.set_data([target_pos[0]], [target_pos[1]])

        points = forward_kinematics(theta_current, L_vals)
        ee_pos = points[:, -1]

        error = target_pos - ee_pos

        max_error_step = 2.0
        error_norm = np.linalg.norm(error)
        if error_norm > max_error_step:
            error = (error / error_norm) * max_error_step

        v_ee = slider_auto_speed.val * error

        J = get_jacobian(theta_current, L_vals)

        damping_factor = slider_damping.val
        J_dls = J.T @ np.linalg.inv(J @ J.T + (damping_factor ** 2) * np.eye(2))

        theta_dot_target = J_dls.dot(v_ee)
        frame_idx = (frame_idx + 1) % len(t_x)
    else:
        theta_dot_target = np.array(jog_directions) * slider_velocity_manual.val

    theta_ddot = (theta_dot_target - theta_dot_actual) / tau
    theta_dot_actual += theta_ddot * dt

    for i in range(3):
        theta_dot_actual[i] = np.clip(theta_dot_actual[i], -joint_speed_limit, joint_speed_limit)

    next_theta = theta_current + theta_dot_actual * dt

    warnings = ["", "", ""]
    for i in range(3):
        limit_min, limit_max = joint_limits[i]
        if next_theta[i] <= limit_min + 0.001:
            next_theta[i] = limit_min
            warnings[i] = " ⚠ MIN LIMIT"
            if theta_dot_actual[i] < 0: theta_dot_actual[i] = 0
        elif next_theta[i] >= limit_max - 0.001:
            next_theta[i] = limit_max
            warnings[i] = " ⚠ MAX LIMIT"
            if theta_dot_actual[i] > 0: theta_dot_actual[i] = 0

    theta_current = next_theta

    points = forward_kinematics(theta_current, L_vals)
    ee_x, ee_y = points[0, -1], points[1, -1]

    # Visual Updates
    update_link_patches(link_patches, points, LINK_WIDTHS)
    for i, circle in enumerate(joint_circles):
        circle.center = points[:, i]

    # --- NEW: High Visibility Pressure Colors ---
    ee_pressure = int(slider_pressure.val)
    ee_joint_circle = joint_circles[-1]

    if ee_pressure == 1:
        ee_joint_circle.set_color(COLOR_LOW)
    elif ee_pressure == 2:
        ee_joint_circle.set_color(COLOR_MED)
    elif ee_pressure == 3:
        ee_joint_circle.set_color(COLOR_HIGH)

    ee_joint_circle.set_radius(JOINT_SIZE_BASE + (ee_pressure - 1) * 0.02)

    J_actual = get_jacobian(theta_current, L_vals)
    V_ee_actual = J_actual.dot(theta_dot_actual)
    vx, vy, vz = V_ee_actual[0], V_ee_actual[1], 0.0
    wx, wy, wz = 0.0, 0.0, sum(theta_dot_actual)

    recorded_data.append([
        round(current_time, 2), ee_x, ee_y,
        theta_current[0], theta_current[1], theta_current[2],
        vx, vy, vz, wx, wy, wz
    ])

    if current_mode.startswith('Auto'):
        current_pressure = slider_pressure.val
        trace_history.append((ee_x, ee_y, current_pressure))

        if len(trace_history) > 150:
            trace_history.pop(0)

        segments = []
        colors_rgba = []

        # Mapping to RGB arrays for the trace line collection
        trace_color_map = {
            1: np.array([0.0, 1.0, 0.0]),  # Bright Green
            2: np.array([1.0, 0.65, 0.0]),  # Vivid Orange
            3: np.array([1.0, 0.0, 0.0])  # Bright Red
        }

        total_pts = len(trace_history)
        for i in range(1, total_pts):
            pt1 = trace_history[i - 1]
            pt2 = trace_history[i]
            segments.append([[pt1[0], pt1[1]], [pt2[0], pt2[1]]])

            alpha = (i / total_pts) ** 1.5
            rgb = trace_color_map.get(pt2[2], trace_color_map[1])
            colors_rgba.append(np.append(rgb, alpha))

        trace_coll.set_segments(segments)
        trace_coll.set_colors(colors_rgba)

    # UI Update
    table_str = f"Mode: {current_mode} | Time: {current_time:.1f}s\n\n"
    table_str += "--- Joint Angles (rad) ---\n"
    for i in range(3):
        lim_str = f"[{round(joint_limits[i][0], 2)}, {round(joint_limits[i][1], 2)}]"
        table_str += f"Th{i + 1}: {theta_current[i]:>6.3f} {lim_str}{warnings[i]}\n"

    table_str += "\n--- Realtime Speeds ---\n"
    table_str += f"Vx: {vx:>6.3f} | Vy: {vy:>6.3f}\n"
    table_str += f"Wz: {wz:>6.3f} rad/s\n\n"

    info_text.set_text(table_str)

    return link_patches + joint_circles + [base_circle, base_stand, trace_coll, info_text, reticle_plot]


ani = animation.FuncAnimation(fig, update, interval=dt * 1000, blit=False, cache_frame_data=False)
plt.show()