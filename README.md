# Interactive 3-DOF Planar Robotic Arm Simulation

An advanced, interactive 3-Degree of Freedom (3-DOF) planar robot arm simulation built in Python using `matplotlib`. This repository serves as a comprehensive platform for visualizing and analyzing core robotics engineering principles, including forward/differential kinematics, singularity avoidance via Damped Least Squares (DLS), safety hard-clamping, and physical motor lag emulation.



---

## 🚀 Detailed Feature Walkthrough

### 1. Interactive GUI & Morphological Control
The simulation features a robust real-time graphical user interface built on top of the Matplotlib event loop, allowing users to modify physical properties on the fly without restarting the simulation:
* **Dynamic Morphological Scaling:** Adjust the lengths of links ($L_1, L_2, L_3$) via sliders. The analytical Jacobian matrices and forward kinematics equations update their physical state mappings instantly.
* **Multi-Mode Trajectory Sequencer:** A `RadioButtons` state machine allows immediate hot-swapping between three distinct behaviors:
  - **Manual Jogging:** Actuates joints independently using incremental directional buttons.
  - **Automated Circle Generation:** Drives the end-effector along an ideal continuous parametric circle.
  - **Automated Straight-Line Tracking:** Commands a linear path vector between discrete cartesian waypoints.

### 2. High-Fidelity Aesthetic & Force Telemetry Rendering
Designed to look like an industrial control application rather than a primitive line plot:
* **Polygonal Link Geometry:** Robot links are rendered using custom polygon patches that taper symmetrically from base to tip, creating an authentic mechanical aesthetic.
* **Visual Force Feedback:** The end-effector node dynamically reacts to an external "Pressure Slider." As pressure values escalate, the end-effector changes color along a vivid gradient (Green $\rightarrow$ Orange $\rightarrow$ Red) and scales up in size, providing intuitive visual feedback of simulated physical strain.
* **Fading Trace History:** Utilizes a specialized `LineCollection` to maintain a rolling window of the last 150 coordinates traversed by the end-effector. The path utilizes alpha-blending (gradient transparency) to create a fading motion trail that maps the color history of the applied pressure telemetry.

### 3. Safety Subsystems & Industrial Soft-Limits
To mimic physical hardware protection, the control loop enforces three distinct layers of programmatic safeguards:
* **Joint Space Bounds Clamping:** The system accepts hard maximum and minimum angle configurations upon launch (e.g., $-180^\circ$ to $+180^\circ$). If an integration step calculates a value exceeding these soft limits, the state vector is clamped at the ceiling value and its command velocity is zeroed out.
* **Velocity Saturation Filters:** To protect against numerical anomalies, maximum joint rotational speeds are bounded to an operational limit of $90^\circ/\text{s}$.
* **Hardware E-Stop Routine:** A prominent, dedicated Emergency Stop UI button instantly breaks the execution loop, commands all target joint velocities to zero, overrides automated tracking modes back to manual, and prints a diagnostic warning dump to the system log.

### 4. Advanced Joint Motor Emulation (First-Order Lag Circuit)
Rather than instantaneous, unnatural software position stepping, the joints in this simulation model the true physics of physical DC motors. The torque-speed response and rotational inertia of the joints are simulated using a **First-Order Dynamic Circuit Model** (mathematically identical to an electrical low-pass $RC$ filter circuit).



When a step command for a target velocity ($\dot{\Theta}_{\text{target}}$) is dispatched by the kinematics controller, the actual joint velocity ($\dot{\Theta}_{\text{actual}}$) does not change instantly. Instead, it behaves like voltage charging across a capacitor in a first-order circuit, governed by the following differential equation:

$$\ddot{\Theta} = \frac{\dot{\Theta}_{\text{target}} - \dot{\Theta}_{\text{actual}}}{\tau}$$

Where $\tau$ represents the motor's mechanical time-constant. This differential is solved at every time-step ($dt$) using explicit Euler Integration:

$$\dot{\Theta}_{\text{actual}}(t + dt) = \dot{\Theta}_{\text{actual}}(t) + \ddot{\Theta} \cdot dt$$

$$\Theta(t + dt) = \Theta(t) + \dot{\Theta}_{\text{actual}}(t + dt) \cdot dt$$

This ensures that the robotic joints exhibit natural physical properties such as smooth acceleration curves, momentum deceleration, and realistic tracking lag.

---

## 🧮 Core Mathematical Implementations

### 1. Forward Kinematics (Analytical Position)
Maps the individual joint space vectors $\Theta = [\theta_1, \theta_2, \theta_3]^T$ into global Cartesian coordinates $(x, y)$ for the tip of each link:

$$x = L_1 \cos(\theta_1) + L_2 \cos(\theta_1 + \theta_2) + L_3 \cos(\theta_1 + \theta_2 + \theta_3)$$

$$y = L_1 \sin(\theta_1) + L_2 \sin(\theta_1 + \theta_2) + L_3 \sin(\theta_1 + \theta_2 + \theta_3)$$

### 2. Differential Kinematics & The Jacobian Matrix
Linearizes the relationship between the joint velocity vector $\dot{\Theta}$ and the resulting Cartesian velocity vector $V = [v_x, v_y]^T$. The analytical Jacobian ($J$) is a $2 \times 3$ matrix derived from partial derivatives of the forward kinematics equations:

$$J(\Theta) = \begin{bmatrix} 
\frac{\partial x}{\partial \theta_1} & \frac{\partial x}{\partial \theta_2} & \frac{\partial x}{\partial \theta_3} \\ 
\frac{\partial y}{\partial \theta_1} & \frac{\partial y}{\partial \theta_2} & \frac{\partial y}{\partial \theta_3} 
\end{bmatrix}$$

Using the shorthand notations $s_1 = \sin(\theta_1)$, $c_{12} = \cos(\theta_1 + \theta_2)$, and $s_{123} = \sin(\theta_1 + \theta_2 + \theta_3)$, the compiled matrix evaluates to:

$$J(\Theta) = \begin{bmatrix}
-L_1 s_1 - L_2 s_{12} - L_3 s_{123} & -L_2 s_{12} - L_3 s_{123} & -L_3 s_{123} \\
L_1 c_1 + L_2 c_{12} + L_3 c_{123} & L_2 c_{12} + L_3 c_{123} & L_3 c_{123}
\end{bmatrix}$$

### 3. Singularity Handling via Damped Least Squares (DLS)
When tracking paths, the controller must perform Inverse Kinematics to find required joint velocities: $\dot{\Theta} = J^{-1}V$. Because the robot has redundant degrees of freedom ($2 \times 3$ non-square matrix) and encounters singularities (e.g., when the arm is fully extended and $\det(JJ^T) \to 0$), standard pseudo-inverse calculations yield joint speeds approaching infinity.

To prevent this, this repository implements the **Damped Least Squares (Levenberg-Marquardt)** optimization algorithm. It minimizes a multi-objective cost function balancing tracking error against high joint speeds, resulting in the following inverted control law:

$$\dot{\Theta}_{\text{target}} = J^T \left( J J^T + \lambda^2 I \right)^{-1} V_{\text{target}}$$

Where:
- $I$ is a $2 \times 2$ identity matrix.
- $\lambda$ is the **Damping Factor**, controllable via the GUI (`DLS Damp` slider). 

When the robot approaches a singularity boundary, $\lambda^2 I$ populates the denominator, mathematically preventing division-by-zero. This smoothly bounds the joint velocities to a safe operating envelope by sacrificing marginal tracking precision in favor of absolute numerical and physical stability.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   
   git clone [https://github.com/Adel1054/Robot-Arm-Simulation.git](https://github.com/Adel1054/Robot-Arm-Simulation.git)
   cd Robot-Arm-Simulation

2. **Install dependencies:**

   pip install -r requirements.txt

3. **Run the simulation:**

   python Robot-Arm-Simulation.py


## 📊 Automated Telemetry Logging

Clicking the "Save Data to CSV" button dumps the entire operational memory array into a clean, time-series .csv spreadsheet. The logged state-space includes:

- Timestamp (s) — Exact historical clock steps.

- $EE_X$, $EE_Y$ — Cartesian position coordinates.

- $\Theta_1$, $\Theta_2$, $\Theta_3$ — Joint angles (radians).

- $Vel_X$, $Vel_Y$ — Resulting end-effector Cartesian velocity tracking performance.
