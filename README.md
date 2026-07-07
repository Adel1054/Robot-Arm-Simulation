# 3-DOF Planar Robotic Arm Simulation

An interactive 3-Degree of Freedom (3-DOF) planar robot arm simulation built in Python using `matplotlib`. This project demonstrates core robotics concepts including forward kinematics, differential kinematics via the Jacobian matrix, singularity handling using Damped Least Squares (DLS), and basic motor dynamics.

## 🚀 Key Features
- **Interactive GUI Control:** Real-time adjustments of link lengths, target velocities, and operating modes (Manual Jogging vs. Automated Tracking).
- **Advanced Inverse Kinematics:** Implementation of Damped Least Squares (DLS) to safely navigate kinematic singularities without joint speed explosions.
- **Visual Force Feedback:** Dynamic end-effector scaling and color-shifting based on applied pressure metrics.
- **Fading Trace History:** High-performance rendering of the last 150 end-effector coordinates with alpha-blended paths mimicking physical motion trails.
- **Safety Soft-Limits:** Real-time joint limit clamping and an Instant Emergency Stop (E-Stop) safety routine.
- **Telemetry System:** Real-time data logging exporting multi-variable parameters (time, positions, angles, velocities) straight to a `.csv` file.

## 🧮 Mathematical Foundations

### 1. Forward Kinematics
Calculates the absolute Cartesian coordinates $(x_3, y_3)$ of the end-effector given the joint configuration $\Theta = [\theta_1, \theta_2, \theta_3]^T$:
$$x = L_1 \cos(\theta_1) + L_2 \cos(\theta_1 + \theta_2) + L_3 \cos(\theta_1 + \theta_2 + \theta_3)$$
$$y = L_1 \sin(\theta_1) + L_2 \sin(\theta_1 + \theta_2) + L_3 \sin(\theta_1 + \theta_2 + \theta_3)$$

### 2. Singularity Handling via Damped Least Squares (DLS)
To map Cartesian target velocities to joint speeds near fully extended singularities, standard pseudo-inverse calculations collapse. This simulation introduces a damping factor $\lambda$ to stabilize the inversion process:
$$\dot{\Theta}_{target} = J^T (J J^T + \lambda^2 I)^{-1} V_{target}$$

The damping factor prevents joint speeds from spiking to infinity near workspace boundaries by trading off perfect path accuracy for mechanical stability.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   
   git clone [https://github.com/Adel1054/your-repo-name.git](https://github.com/Adel1054/Robot-Arm-Simulation.git)
   cd Robot-Arm-Simulation

2. **Install dependencies:**

   pip install -r requirements.txt

3. **Run the simulation:**

   python Robot-Arm-Simulation.py