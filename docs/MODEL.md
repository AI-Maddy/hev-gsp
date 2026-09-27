# Model description

This file describes every equation the simulator integrates. Symbols follow the code (`hevgsp/`).
The physical plant is integrated with explicit Euler at 50 Hz. The ride model runs at 400 Hz. All outputs are block-averaged to 10 Hz.

## 1. Vehicle and road

**Longitudinal (forward-facing).**

$(m+m_{rot})\dot v = F_w - \tfrac12\rho C_d A v^2 - \sum_i C_{rr,i} F_{z,i}\cos\theta - m g\sin\theta - F_{fric} - F_{drag}$

- $C_{rr,i}=(C_{rr0}+C_{rr,v}v)(p_0/p_i)^{0.4}$, so an under-inflated tire raises rolling resistance.
- $F_{drag}$ is the brake-drag fault force.
- The IMU reports specific force, $a_{long}=\dot v + g\sin\theta$.

**Driver.** A PI controller with feed-forward tracks $v_{ref}$:
$F_{req}=m_{eff}\dot v_{ref}+\hat F_{res}+K_p e+K_i\!\int e$, with $K_p=1400$ and $K_i=180$.
- $F_{req}>0$ maps to the accelerator pedal, $pedal=F_{req}/F_{max}(v)$, rate-limited to 150 %/s.
- $F_{req}<0$ maps to the brake demand.

**Lateral (single-track, steady-state yaw with lag).**
- Curvature is $\kappa=a_{lat}^{des}/v_{ref}^2$, clipped to $|\kappa|\le0.12$ m⁻¹.
- Yaw rate follows $\tau\dot r = v\kappa - r$, and $a_{lat}=vr$.
- Steering-wheel angle is $\delta_{sw}=i_s(L\kappa + K_{us}a_{lat})$.

**Loads and tires.**
- Static axle loads plus longitudinal transfer $m a h/L$ and lateral transfer $m a_y h/t$, split front/rear by roll-stiffness share.
- Tire vertical stiffness is $k_t=k_{t0}(0.3+0.7p/p_0)$.
- Effective rolling radius is $r_e=r_{free}-F_z/(3k_t)$, which is the basis of indirect TPMS.
- Tire temperature is first order toward $T_{amb}+c_1v+c_2|a_y|$. Pressure follows the gas law on absolute pressure, $p+p_{atm}=(p_{cold}+p_{atm})(T_{tire}+273)/(T_{amb}+273)$ (gauge pressures reported).
- Wheel speeds are $\omega_i = v_i(1+s_i)/r_{e,i}$, with $v_{L/R}=v\mp r t/2$ and longitudinal slip $s_i=F_{x,i}/(C_s F_{z,i})$.

**Drive cycles.** Randomized micro-trips (urban, suburban, highway, hilly), plus a four-phase "mixed" cycle shaped like WLTC Class 3. The mixed cycle is **not** the regulatory speed table.
- Road grade is spatially correlated noise, ±1.2 % (±5.5 % on hilly).
- Road roughness follows the ISO 8608 PSD $G_d(n)=G_d(n_0)(n/n_0)^{-2}$ with patchwise class A/B/C. Left/right track coherence is 0.6, and the rear wheels see the front profile delayed by the wheelbase.
- Speed bumps and potholes are added.

## 2. Power-split hybrid transmission (planetary e-CVT)

**Kinematics.**
- Willis equation: $(S+R)\omega_e = S\omega_1 + R\omega_r$.
- Ring speed: $\omega_r = FD\,\bar\omega_{front}$ (mean front-wheel speed, including slip).
- MG2 speed: $\omega_2=k_{MG2}\omega_r$.

**Torques.**
- Carrier torque: $T_c=T_e - J_e\dot\omega_e$.
- Engine share at the ring: $T_{r,e}=\frac{R}{S+R}T_c$.
- MG1 reaction torque: $T_1=-\frac{S}{S+R}T_c$.
- MG2 torque: $T_2=(T_{r}^{dem}-T_{r,e})/k_{MG2}$, clipped to $\min(T_{max},P_{max}/|\omega_2|)$.

**Energy management (HCU function, hosted on the CVC).**
- Power request: $P_{req}=P_{wheel}+P_{chg}(SOC)+P_{aux}+P_{lim}$, with charge-sustaining term $P_{chg}=K_{SOC}(SOC^*-SOC)$.
- The engine starts when $P_{req}>15$ kW, $v>70$ km/h, SOC $<0.48$, or on warm-up (coolant $<45$ °C) or cabin-heat request. It stops with hysteresis (SOC $>0.52$) and minimum on/off times. $P_{chg}$ is capped at 14 kW.
- Engine speed follows the optimal operating line $\omega^*(P)$ through a first-order MG1 speed loop ($\tau=0.45$ s). It is bounded so that $|\omega_1|\le\omega_{1,max}$.
- Engine torque is $\min(P/\omega_e, T_{max}(\omega_e))$.

**Engine.**
- Willans fuel model: $\dot m_f=(T_{cmd}+T_f(\omega))\omega/(\eta_i\,LHV)$.
- Friction $T_f=(T_{f0}+T_{f1}\omega)$ rises when the engine is cold.
- The ECM torque signal is the air-path estimate $T_{cmd}$. A misfire lowers only the delivered torque, so the ECM estimate and the MG1 reaction torque become inconsistent.

**Motors.**
- Losses: $P_{loss}=k_{cu}T^2+k_{fe}|\omega|+k_{fe2}\omega^2+c_0$.
- Electrical power: $P_{el}=T\omega+P_{loss}$ (the same loss heats the winding node, so the electrical and thermal paths are energy-consistent).
- The MG2 inter-turn fault adds $500\,f(\omega/1000)^2$ W and scales $k_{cu}$.

**Inverter.** Losses are $c_0+c_1(|P_1|+|P_2|)+c_2(P_1^2+P_2^2)$.

## 3. Battery and low-voltage system

**Battery (60s Li-ion, 5 Ah, first-order Thevenin model).**
- Current from power: $I=\frac{(V_{oc}-V_1)-\sqrt{(V_{oc}-V_1)^2-4R_0P}}{2R_0}$.
- RC branch: $\dot V_1=-V_1/\tau_1+I/C_1$.
- SOC: $\dot{SOC}=-I/Q$.
- Resistance: $R_0(T)=R_{0,25}e^{E_a/R(1/T-1/298)}$, times $(1+1.5f)$ under the resistance fault.
- Power limits are derated when the pack is cold.

**Low-voltage system.**
- DC-DC input power: $P_{12}/\eta$.
- Bus voltage: $V_{12}=V_{set}-R_{out}I$.
- The DC-DC fault lowers $V_{set}$ and $\eta$ and adds ripple.
- The 12 V load combines a base load, stochastic consumers, and cold-weather defrost and seat heating.

## 4. Thermal networks (lumped capacitances)

| Node | Equation |
|---|---|
| Engine coolant | $C\dot T=q_{cool}\dot Q_{fuel}+0.3P_{fric,motored}-\dot m c_p(T-T_{rad})-\dot Q_{heater}-UA(T-T_{amb})$, $C$ = 55 kJ/K, $q_{cool}$ = 0.36. Radiator flow $\dot m=0.6\,\text{open}+0.002$ kg/s; opening ramps from 82 to 95 °C (stuck-open fault holds it at 60 %/100 %). |
| Radiator | $C\dot T_{rad}=\dot m c_p(T-T_{rad})-UA_{air}(v,fan)(T_{rad}-T_{amb})$ |
| Catalyst | First order toward $T_{amb}+350+400\sqrt{P/P_{max}}+600\,\text{misfire}$ ($\tau$ = 35 s heating, 300 s cooling) |
| MG windings, inverter | $C\dot T_w=P_{loss}-G(T_w-T_{edu})$. The pump fault scales $G$ and $UA$. |
| Electric-drive coolant | $C\dot T=\sum G(T_k-T)-pump\,(UA_{lt}(v)+UA_{fan})(T-T_{amb})$ |
| Battery | $C\dot T=I^2R_0+V_1^2/R_1-hA(fan)(T-T_{cabin})$ |
| Cabin | $C\dot T=UA(v)(T_{amb}-T)+\dot Q_{solar}+\dot Q_{HVAC}+90$ W. A PI controller sets $\dot Q_{HVAC}$: heating comes from the heater core, limited by coolant temperature, plus a PTC heater; cooling comes from the electric compressor, $P_{ac}=\dot Q/COP(T_{amb})$. |
| Brakes | Front/rear discs take the friction power split 70/30. Cooling is $h_0+h_vv$ (rear 0.8×). |

## 5. Ride (7-DOF full vehicle, 400 Hz)

**States:**
- body heave $z_s$, pitch $\theta$, roll $\phi$;
- four unsprung masses $z_{u,i}$.

**Suspension forces:**
- corner body displacement $z_{b,i}=z_s+x_i\theta+y_i\phi$;
- spring/damper force $F_i=-k_i d_i - c_i\dot d_i$, with deflection $d_i=z_{b,i}-z_{u,i}$;
- anti-roll bars couple the left/right deflections on each axle;
- tire force $k_{t,i}(z_{r,i}-z_{u,i})$.

**Inertial moments:**
- pitch $M_p=m_s a_x h$;
- roll $M_r=m_s a_y h$.

**Integration:**
- The system is discretized with zero-order hold and simulated piecewise-LTI in 1 s blocks (parameters evaluated at the block start, so a fault effect never precedes its label), because $k_t$ changes with tire pressure and $c_{RR}$ with the damper fault.

**Outputs at 10 Hz:**
- mean suspension travel (mm);
- RMS body acceleration per corner.

## 6. E/E layer (frame-level discrete-event simulation)

### Classic Platform ECUs (COM, OS, CanIf)

**I-PDU transmission:**
- Every I-PDU in `comm/comm_matrix.csv` uses `ComTxModeMode` PERIODIC or MIXED. MIXED is periodic plus `TRIGGERED_ON_CHANGE`.
- The triggered part uses filter `MASKED_NEW_DIFFERS_MASKED_OLD` on the raw signal $\lfloor(x-\text{offset})/\text{factor}\rfloor$, with the low `mask_bits` cleared.
- Triggered transmissions land on `Com_MainFunctionTx` ticks; `ComMinimumDelayTime` is enforced across periodic and triggered sends.

**Release jitter:**
- Frames are released by `Com_MainFunctionTx`.
- Response time is $C+\Gamma(2,\,C\,u/(2(1-u)))$, where $u$ is the ECU CPU load. The load grows with the frame rate. Releases of one PDU stay in FIFO order.

**CanIf buffering:**
- The buffer holds one instance per PDU. A newer instance overwrites an unsent one, and the loss is counted.

### CAN FD (ISO 11898-1:2015)

- Bit rates are 500 kbit/s arbitration and 2 Mbit/s data.
- Base-format frame length:
  - arbitration phase (SOF through BRS): 17 bits at the nominal rate;
  - data phase: ESI, DLC, data, 4-bit stuff count, CRC-17 (≤16 B) or CRC-21, plus fixed stuff bits (one before the stuff count and one every 4 bits of the CRC field);
  - dynamic stuff bits in SOF..data drawn as Binomial(n, 1/30), capped at the worst case;
  - tail (CRC delimiter, ACK, ACK delimiter, EOF, IFS): 13 bits.
- Arbitration is non-preemptive: at each bus-idle instant, the pending frame with the lowest CAN ID wins.

### E2E protection (as implemented in CAPI `isoft/e2e`)

- **P11:** CRC-8 in byte 0, 4-bit counter at bit 8 (the counter wraps mod 15).
- **P05:** CRC-16 in bytes 0–1, 8-bit counter in byte 2.
- The simulation reserves these header bits in the PDU layout (DBC/ARXML) but does not compute CRCs. It counts what the receiver-side E2E check would flag: lost instances (counter jump > MaxDeltaCounter = 1) and reception gaps longer than `ComTimeout` = 2.5 × period.
- The counts are stored in `aux/e2e_errors`.

### Zonal gateways (CP)

- PduR routing plus signal-to-service translation, modeled as a FIFO server.
- Service time is $(40+8n_{sig})$ µs $/(1-u_{bg})$.
- The zonal-overload fault injects a DoIP (ISO 13400) UDS request stream.

### Backbone (100BASE-T1)

- SOME/IP notification per event: 16-byte header plus UDP/IPv4/Ethernet overhead.
- Background streams have OU-varying load.
- Queueing is M/D/1.

### CVC (Adaptive Platform, CAPI)

- The ara::com event dispatcher is a FIFO server whose speed depends on the other applications' load.
- The graph signal is the end-to-end latency, from sensor sample to consumer.

**PHM** (CAPI `phm` `DeadlineSupervision`, `maxDeadline` = 12 ms) supervises the energy-management inputs only (ECM_EngSts, PCU_MgSts, ESC_VehDyn, ESC_WhlSpd, EPS_Sts, BMS_PackSts, pedal): sensor sample → ara::com consumer. Violations per window are in `aux/phm_deadline_fail`. Alive and logical supervision are not simulated.

### TCU (Adaptive Platform)

- Telemetry uplink as an M/M/1 queue over an OU-varying cellular rate, plus half the RTT.

### Graph signals per 100 ms window

| Node | Signal |
|---|---|
| CP ECU | mean end-to-end latency (ms) |
| Zonal gateway | mean sojourn time (ms) |
| CVC, TCU | latency (ms) |
| Buses | utilization (%) |

## 7. Sensors

- Each measured value is clean + Gaussian noise, quantized to the CAN signal resolution (the CompuMethod factor) and clipped to the CAN signal range.
- Where SAE J1979 defines a PID, the scaling follows it: engine speed 0.25 rpm/bit, coolant temperature 1 °C with a −40 °C offset.
- E/E window means get 1 % multiplicative measurement noise and ~1 µs resolution; bus utilization gets ±0.2 % noise.
- Measured values are clipped to the CAN signal range, so noise on non-negative signals (fuel rate, P_ac, speed, brake pressure) is half-rectified near zero, as in a real encoder.

**Sensor faults:**
- yaw-rate bias drift;
- battery temperature stuck at its last value with an offset;
- wheel-speed dropouts plus noise.

## 8. Graph

**Physical layer: structural edges.**
- An edge $i$–$j$ exists when the governing equation of one quantity contains the other directly.
- Edges are typed kinematic, mechanical, control, electrical, or thermal (99 edges).

**E/E layer:**
- CAN FD attachment, 100BASE-T1 links, and ara::com provider–consumer relations (20 edges).

**Interlayer:**
- Each sensor node links to the ECU that samples and transmits it (47 edges).

**Edge weights provided:**
- binary $A$;
- $W_{signed}=\rho_{ij}$, the nominal correlation, on every graph edge (all three layers);
- $|W|$.

The signed Laplacian $L=D_{|W|}-W$ is PSD, with $x^\top Lx=\sum|w_{ij}|(x_i-\mathrm{sgn}(w_{ij})x_j)^2$.
