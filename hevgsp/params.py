"""Vehicle parameters for a mid-size, front-wheel-drive, power-split hybrid.

Values are representative of published mid-size power-split HEVs (planetary
e-CVT, ~73 kW Atkinson engine, ~60 kW traction motor, ~1 kWh Li-ion pack).
They are not the specification of any particular production vehicle.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Chassis:
    m: float = 1470.0          # kg, curb + driver
    m_rot: float = 45.0        # kg, equivalent rotating mass (wheels, driveline)
    g: float = 9.81
    rho_air: float = 1.20
    Cd: float = 0.26
    A_f: float = 2.20          # m^2
    Crr0: float = 0.0085       # rolling-resistance coefficient at nominal pressure
    Crr_v: float = 6.0e-5      # speed dependence, per (m/s)
    wheelbase: float = 2.70
    a: float = 1.08            # CG to front axle (m)
    b: float = 1.62            # CG to rear axle (m)
    track: float = 1.53
    h_cg: float = 0.55
    roll_front_share: float = 0.62
    r_free: float = 0.315      # free tire radius (m)
    steer_ratio: float = 14.5
    K_us: float = 0.0025       # understeer gradient, rad/(m/s^2)
    tau_yaw: float = 0.15      # s


@dataclass
class Tires:
    p0: float = 2.40           # bar, cold placard pressure
    k_t0: float = 210e3        # N/m vertical stiffness at p0
    slip_stiff: float = 15.0   # F_x = slip_stiff * F_z * slip
    tau_T: float = 400.0       # s, tire temperature time constant


@dataclass
class PowerSplit:
    S: int = 30                # sun teeth (MG1)
    R: int = 78                # ring teeth (output)
    k_mg2: float = 2.636       # MG2 -> ring reduction
    FD: float = 3.267          # final drive (ring -> wheel)
    eta_gear: float = 0.97
    J_eng: float = 0.18        # kg m^2
    tau_eng_speed: float = 0.45  # s, MG1 speed-control closed loop


@dataclass
class Engine:
    P_max: float = 73e3
    rpm_tq: tuple = (0, 1000, 2000, 3000, 4000, 5200, 5600)
    tq_max: tuple = (0, 102, 125, 138, 142, 134, 120)
    # optimal operating line (power W -> rpm)
    P_opt: tuple = (0, 5e3, 10e3, 20e3, 30e3, 45e3, 60e3, 73e3)
    rpm_opt: tuple = (1000, 1100, 1300, 1800, 2400, 3400, 4400, 5200)
    eta_ind: float = 0.415     # Willans indicated efficiency
    Tf0: float = 8.0           # friction torque (N m)
    Tf1: float = 0.020         # friction torque per rad/s
    LHV: float = 43.0e6        # J/kg
    q_cool_frac: float = 0.36  # fraction of fuel power to coolant (part load)
    C_th: float = 55e3         # J/K engine + coolant effective capacity
    UA_amb: float = 9.0        # W/K
    thermo_open: tuple = (82.0, 95.0)
    C_rad: float = 8e3         # J/K radiator
    mdot_rad_max: float = 0.60  # kg/s at full thermostat opening
    cp_cool: float = 3600.0    # J/(kg K), glycol mix
    UA_rad0: float = 120.0     # W/K air side
    UA_rad_v: float = 8.0      # W/K per m/s
    UA_fan: float = 600.0      # W/K when fan on
    T_fan_on: float = 98.0
    P_on: float = 15e3
    P_off: float = 8e3
    v_ev_max: float = 19.5     # m/s EV-mode speed limit
    min_on: float = 4.0
    min_off: float = 3.0


@dataclass
class Motor:
    T_max: float
    P_max: float
    w_max: float
    k_cu: float
    k_fe: float
    k_fe2: float
    c0: float
    C_th: float
    G_th: float


MG1 = Motor(T_max=60.0, P_max=42e3, w_max=1100.0, k_cu=0.12, k_fe=0.3, k_fe2=3e-4,
            c0=80.0, C_th=5e3, G_th=40.0)
MG2 = Motor(T_max=163.0, P_max=60e3, w_max=1450.0, k_cu=0.08, k_fe=0.4, k_fe2=4e-4,
            c0=100.0, C_th=8e3, G_th=60.0)


@dataclass
class Inverter:
    c0: float = 60.0
    c1: float = 0.015
    c2: float = 2e-7
    C_th: float = 3e3
    G_th: float = 80.0


@dataclass
class EDUCooling:
    C_th: float = 12e3
    UA0: float = 80.0
    UA_v: float = 12.0
    UA_fan: float = 300.0


@dataclass
class Battery:
    n_series: int = 60
    Q_Ah: float = 5.0
    soc_pts: tuple = (0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0)
    ocv_pts: tuple = (3.30, 3.50, 3.58, 3.63, 3.67, 3.72, 3.79, 3.87, 3.96, 4.06, 4.18)
    R0_cell: float = 1.2e-3    # ohm at 25 C
    R1_cell: float = 0.8e-3
    tau1: float = 15.0
    Ea_R: float = 3500.0       # Arrhenius, K
    C_th: float = 25e3
    hA0: float = 5.0
    hA_fan: float = 20.0
    P_dis_max: float = 25e3
    P_chg_max: float = 20e3
    soc_target: float = 0.60
    K_soc: float = 60e3        # W per unit SOC error


@dataclass
class LowVoltage:
    V_set: float = 14.2
    R_out: float = 0.004
    eta: float = 0.93
    P_base: float = 330.0


@dataclass
class Cabin:
    C_th: float = 90e3
    UA0: float = 55.0
    UA_v: float = 2.5
    T_set: float = 22.0
    Q_max: float = 5.0e3
    k_heater_core: float = 150.0   # W/K
    P_ptc_max: float = 1.5e3
    Kp: float = 900.0
    Ki: float = 6.0


@dataclass
class Brakes:
    front_share: float = 0.70
    N_per_bar: float = 180.0
    C_front: float = 6440.0
    C_rear: float = 4600.0
    h0: float = 8.0
    h_v: float = 2.4


@dataclass
class Suspension:
    ms: float = 1320.0         # sprung mass
    Iy: float = 2200.0         # pitch inertia
    Ix: float = 520.0          # roll inertia
    mu_f: float = 38.0         # unsprung mass per corner
    mu_r: float = 36.0
    ks_f: float = 26e3
    ks_r: float = 22e3
    cs_f: float = 2400.0
    cs_r: float = 2100.0
    k_arb_f: float = 18e3      # anti-roll bar (N/m equivalent at wheel)
    k_arb_r: float = 8e3
    fs: float = 400.0          # Hz, integration rate


@dataclass
class Vehicle:
    ch: Chassis = field(default_factory=Chassis)
    ty: Tires = field(default_factory=Tires)
    ps: PowerSplit = field(default_factory=PowerSplit)
    eng: Engine = field(default_factory=Engine)
    mg1: Motor = field(default_factory=lambda: MG1)
    mg2: Motor = field(default_factory=lambda: MG2)
    inv: Inverter = field(default_factory=Inverter)
    edu: EDUCooling = field(default_factory=EDUCooling)
    bat: Battery = field(default_factory=Battery)
    lv: LowVoltage = field(default_factory=LowVoltage)
    cab: Cabin = field(default_factory=Cabin)
    brk: Brakes = field(default_factory=Brakes)
    sus: Suspension = field(default_factory=Suspension)


RPM = np.pi / 30.0
