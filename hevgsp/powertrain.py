"""Forward-facing longitudinal/lateral vehicle + power-split HEV + thermal model.

Integrated with explicit Euler at 50 Hz, then block-averaged to the output
rate. All equations are in docs/MODEL.md.
"""
import math
import numpy as np
from .params import Vehicle, RPM
from .faults import lvl

PHYS_OUT = [
    "v_veh", "a_long", "a_lat", "yaw_rate", "steer_angle", "pedal_acc",
    "w_FL", "w_FR", "w_RL", "w_RR",
    "p_tire_FL", "p_tire_FR", "p_tire_RL", "p_tire_RR",
    "brake_pressure", "T_brake_F", "T_brake_R",
    "n_eng", "tq_eng", "fuel_rate", "T_cool_eng", "T_cat",
    "n_MG1", "tq_MG1", "n_MG2", "tq_MG2", "T_MG1", "T_MG2", "T_inv",
    "I_batt", "V_batt", "SOC", "T_batt",
    "V_12", "I_dcdc",
    "T_cool_edu", "T_rad_out", "T_cabin", "P_ac",
]
AUX_OUT = ["s", "grade", "engine_on", "tq_eng_actual", "P_batt", "F_fric",
           "Fz_FL", "Fz_FR", "Fz_RL", "Fz_RR", "k_t_FL", "k_t_FR", "k_t_RL", "k_t_RR",
           "a_long_kin", "a_lat_kin", "Q_hvac", "fault_level"]


def _motor_loss(m, T, w, kcu_mult=1.0):
    return m.k_cu * kcu_mult * T * T + m.k_fe * abs(w) + m.k_fe2 * w * w + m.c0


def _clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def simulate_powertrain(route, faults=(), veh=None, dt=0.02, out_dt=0.1, seed=0,
                        soc0=0.60):
    V = veh or Vehicle()
    ch, ty, ps, eng, m1, m2 = V.ch, V.ty, V.ps, V.eng, V.mg1, V.mg2
    inv, edu, bat, lv, cab, brk = V.inv, V.edu, V.bat, V.lv, V.cab, V.brk
    rng = np.random.default_rng(seed + 7)
    Tamb = route.T_amb

    n = int(round(route.t[-1] / dt)) + 1
    t = np.arange(n) * dt
    v_ref = np.interp(t, route.t, route.v_ref)
    a_ref = np.gradient(v_ref, dt)
    a_ref = np.convolve(a_ref, np.ones(25) / 25, mode="same")
    alat_des = np.interp(t, route.t, route.a_lat_des)
    P12 = lv.P_base + np.interp(t, route.t, route.P_aux12)
    solar = np.interp(t, route.t, route.solar)
    noise_u = rng.standard_normal(n)

    names = [f.name for f in faults]
    def F(name):
        out_ = np.zeros(n)
        for f in faults:
            if f.name == name:
                out_ = f.severity * np.asarray(f.level(t), float)
        return out_
    f_bat, f_pump, f_mg2 = F("bat_resistance"), F("edu_pump"), F("mg2_winding")
    f_mis, f_th, f_tire = F("misfire"), F("thermostat_stuck"), F("tire_leak_FL")
    f_brk, f_dcdc = F("brake_drag_F"), F("dcdc_degradation")
    if "tire_leak_FL" in names:                   # smooth exponential, not stepwise
        fl = [f for f in faults if f.name == "tire_leak_FL"][0]
        f_tire = fl.severity * fl.level(t)
    fault_level = np.maximum.reduce([f_bat, f_pump, f_mg2, f_mis, f_th, f_tire, f_brk, f_dcdc])

    # constants
    m_eff = ch.m + ch.m_rot
    L = ch.wheelbase
    S, R = ps.S, ps.R
    kS, kR = S / (S + R), R / (S + R)
    Q_As = bat.Q_Ah * 3600.0
    ns = bat.n_series
    R1p = bat.R1_cell * ns
    C1 = bat.tau1 / R1p
    soc_pts, ocv_pts = np.array(bat.soc_pts), np.array(bat.ocv_pts) * ns
    rpm_tq = np.array(eng.rpm_tq) * RPM
    tq_max_tab = np.array(eng.tq_max)
    P_opt, w_opt = np.array(eng.P_opt), np.array(eng.rpm_opt) * RPM
    Fz_static_f = ch.m * ch.g * ch.b / L / 2
    Fz_static_r = ch.m * ch.g * ch.a / L / 2
    r_nom = ch.r_free - Fz_static_f / ty.k_t0 / 3
    grade_arr, ds = route.grade, route.ds
    ns_grid = len(grade_arr)

    # states
    v = 0.0; s_pos = 0.0; I_int = 0.0
    yaw = 0.0
    w_e = 0.0; eng_on = False; t_switch = -100.0
    soc = soc0; V1 = 0.0
    T_bat = Tamb + (3.0 if Tamb > 25 else 0.0)
    T_w1 = T_w2 = T_inv = T_edu = Tamb
    T_cool = Tamb; T_rad = Tamb; T_cat = Tamb
    T_cab = Tamb + (15.0 if Tamb > 28 else 3.0 if Tamb > 10 else 0.0)
    hv_int = 0.0
    T_bf = T_br = Tamb
    T_tf = T_tr = Tamb
    p_cold = np.full(4, ty.p0)
    P_lim_bias = 0.0
    a_prev = 0.0
    fan_on = False
    pedal_prev = 0.0
    w_front_prev = 0.0

    out = {k: np.zeros(n) for k in PHYS_OUT + AUX_OUT}
    O = out

    for i in range(n):
        ti = t[i]
        # ---------------- road / geometry
        gi = int(s_pos / ds)
        theta = grade_arr[gi if gi < ns_grid else ns_grid - 1]
        sin_th, cos_th = math.sin(theta), math.cos(theta)

        # ---------------- lateral (single-track, steady-state yaw with lag)
        vr_ = max(v_ref[i], 3.0)
        kappa = _clip(alat_des[i] / (vr_ * vr_), -0.12, 0.12)
        yaw += (v * kappa - yaw) * dt / ch.tau_yaw
        a_lat = v * yaw
        steer = ch.steer_ratio * (L * kappa + ch.K_us * a_lat) * 180 / math.pi

        # ---------------- tires: temperature, pressure, stiffness, loads
        T_tf += ((Tamb + 0.55 * v + 2.0 * abs(a_lat)) - T_tf) * dt / ty.tau_T
        T_tr += ((Tamb + 0.45 * v + 1.6 * abs(a_lat)) - T_tr) * dt / ty.tau_T
        p_cold[0] = ty.p0 * (1 - 0.45 * f_tire[i])
        pf = (T_tf + 273.15) / (Tamb + 273.15)
        pr = (T_tr + 273.15) / (Tamb + 273.15)
        PATM = 1.013                                   # gas law on absolute pressure
        p = tuple((p_cold[j] + PATM) * (pf if j < 2 else pr) - PATM for j in range(4))
        dFx = ch.m * a_prev * ch.h_cg / L / 2
        dFy = ch.m * a_lat * ch.h_cg / ch.track
        Fz = (Fz_static_f - dFx - dFy * ch.roll_front_share,
              Fz_static_f - dFx + dFy * ch.roll_front_share,
              Fz_static_r + dFx - dFy * (1 - ch.roll_front_share),
              Fz_static_r + dFx + dFy * (1 - ch.roll_front_share))
        kt = [ty.k_t0 * (0.3 + 0.7 * pj / ty.p0) for pj in p]
        re = [ch.r_free - Fz[j] / kt[j] / 3 for j in range(4)]
        crr_v = ch.Crr0 + ch.Crr_v * v
        F_roll = sum(crr_v * (ty.p0 / p[j]) ** 0.4 * Fz[j] for j in range(4)) * cos_th if v > 0.05 else 0.0
        F_aero = 0.5 * ch.rho_air * ch.Cd * ch.A_f * v * v
        F_grade = ch.m * ch.g * sin_th
        F_drag = 250.0 * f_brk[i] if v > 0.05 else 0.0

        # ---------------- driver (PI + feedforward)
        F_res_est = F_aero + (ch.Crr0 + ch.Crr_v * v) * ch.m * ch.g + F_grade
        err = v_ref[i] - v
        I_int = _clip(I_int + err * dt, -8.0, 8.0)
        F_req = m_eff * a_ref[i] + F_res_est + 1400.0 * err + 180.0 * I_int
        if v_ref[i] < 0.05 and v < 0.3:
            F_req = min(F_req, -1500.0)
        F_max = min(4200.0, 90e3 / max(v, 1.0))
        if F_req >= 0:
            pedal_cmd = min(F_req / F_max, 1.0)
            dp = 1.5 * dt                                   # pedal rate limit 150 %/s
            pedal = pedal_prev + _clip(pedal_cmd - pedal_prev, -2 * dp, dp)
            F_b = 0.0
        else:
            pedal = max(pedal_prev - 3.0 * dt, 0.0)
            F_b = -F_req

        # ---------------- HCU: demand, energy management
        r_f = 0.5 * (re[0] + re[1])
        w_wheel = w_front_prev if v > 0.3 else v / r_f
        w_r = w_wheel * ps.FD
        w2 = w_r * ps.k_mg2
        T_wheel_dem = pedal * F_max * r_f
        P_wheel_dem = pedal * F_max * v
        P_aux = P12[i] / lv.eta
        P_chg = _clip(bat.K_soc * (bat.soc_target - soc), -5e3, 14e3)
        # climate demand (computed below, use previous step's hv_int based heat need)
        heat_need = (Tamb < 12) and (T_cab < cab.T_set - 1.0) and (T_cool < 55.0)
        warmup = T_cool < 45.0 and ti > 5.0          # hold engine on until coolant warm
        P_req = P_wheel_dem + P_chg + P_aux + P_lim_bias
        if eng_on:
            if (ti - t_switch > eng.min_on and P_req < eng.P_off and v < eng.v_ev_max - 1.5
                    and soc > 0.52 and not heat_need and not warmup):
                eng_on = False; t_switch = ti
        else:
            if (ti - t_switch > eng.min_off and (P_req > eng.P_on or v > eng.v_ev_max
                    or soc < 0.48 or ((heat_need or warmup) and v > 0.5))):
                eng_on = True; t_switch = ti
        if eng_on:
            P_min = 10e3 if (heat_need or warmup) else 4e3
            P_e_tgt = _clip(P_req, P_min, eng.P_max) if P_req > 0 else 0.0
            if (heat_need or warmup) and P_req <= 0 and v > 0.5:
                P_e_tgt = 4e3
            w_tgt = float(np.interp(P_e_tgt, P_opt, w_opt))
            # MG1 speed limit bounds the admissible engine speed
            w_hi = (S * m1.w_max + R * w_r) / (S + R)
            w_lo = (R * w_r - S * m1.w_max) / (S + R)
            w_tgt = _clip(w_tgt, max(w_lo, 0.0), w_hi)
        else:
            P_e_tgt, w_tgt = 0.0, 0.0
            w_lo = (R * w_r - S * m1.w_max) / (S + R)
            if w_lo > 0:                              # engine must spin at very high road speed
                w_tgt = w_lo
        dw_e = (w_tgt - w_e) / ps.tau_eng_speed
        mis = f_mis[i] * 0.25
        if mis > 0 and eng_on:
            dw_e += mis * 40.0 * noise_u[i]
        w_e_new = max(w_e + dw_e * dt, 0.0)
        dw_e = (w_e_new - w_e) / dt
        w_e = w_e_new
        firing = eng_on and w_e > 800 * RPM
        Tf = (eng.Tf0 + eng.Tf1 * w_e) * (1 + 0.5 * _clip((80 - T_cool) / 90, 0, 1)) if w_e > 1 else 0.0
        if firing:
            T_cmd = min(P_e_tgt / w_e, float(np.interp(w_e, rpm_tq, tq_max_tab)))
            T_act = T_cmd * (1 - mis * (0.7 + 0.6 * abs(noise_u[i]))) if mis > 0 else T_cmd
            P_ind = (T_cmd + Tf) * w_e             # fuel follows commanded (air-based) torque
            fuel = P_ind / eng.eta_ind / eng.LHV * 1000.0   # g/s
        else:
            T_cmd = 0.0
            T_act = -Tf
            fuel = 0.0
        T_carrier = T_act - ps.J_eng * dw_e
        T_r_eng = kR * T_carrier
        T1 = -kS * T_carrier
        w1 = ((S + R) * w_e - R * w_r) / S
        # MG2 torque to meet ring demand
        if T_wheel_dem > 0:
            T_r_dem = T_wheel_dem / (ps.FD * ps.eta_gear)
        else:
            T_r_dem = 0.0
        T_regen_req = 0.0
        if F_b > 0:
            fade = _clip((v - 1.0) / 3.0, 0, 1)
            T_regen_req = -F_b * r_f * ps.eta_gear / ps.FD * fade      # ring torque
        T_r_dem += T_regen_req
        T2_lim = min(m2.T_max, m2.P_max / max(abs(w2), 1.0))
        T2 = _clip((T_r_dem - T_r_eng) / ps.k_mg2, -T2_lim, T2_lim)
        # electrical
        kcu2 = 1 + 2.0 * f_mg2[i]
        loss1 = _motor_loss(m1, T1, w1)
        # inter-turn short: circulating-current loss grows with back-EMF^2 (speed^2)
        loss2 = _motor_loss(m2, T2, w2, kcu2) + 500.0 * f_mg2[i] * (w2 / 1000.0) ** 2
        P1 = T1 * w1 + loss1
        P2 = T2 * w2 + loss2
        P_inv_loss = inv.c0 + inv.c1 * (abs(P1) + abs(P2)) + inv.c2 * (P1 * P1 + P2 * P2)
        # climate
        Q_req = _clip(cab.Kp * (cab.T_set - T_cab) + cab.Ki * hv_int, -cab.Q_max, cab.Q_max)
        hv_int = _clip(hv_int + (cab.T_set - T_cab) * dt, -800, 800)
        if Q_req > 0:
            valve = _clip((T_cool - 30.0) / 30.0, 0.1, 1.0)
            Q_hc = min(Q_req, max(valve * cab.k_heater_core * (T_cool - T_cab), 0.0))
            P_ptc = min(Q_req - Q_hc, cab.P_ptc_max) if T_cool < 50 else 0.0
            Q_hvac = Q_hc + P_ptc
            P_ac = 0.0
        else:
            Q_hc, P_ptc = 0.0, 0.0
            cop = max(3.0 - 0.04 * (Tamb - 25), 1.6)
            P_ac = -Q_req / cop
            Q_hvac = Q_req
        eta_dc = lv.eta - 0.08 * f_dcdc[i]
        V12 = lv.V_set - 0.8 * f_dcdc[i] + 0.25 * f_dcdc[i] * noise_u[i]
        I_dc = P12[i] / V12
        V12 -= lv.R_out * I_dc
        P_b = P1 + P2 + P_inv_loss + P12[i] / eta_dc + P_ac + P_ptc
        # battery limits
        tmult = _clip((T_bat + 15) / 35, 0.4, 1.0)
        P_dis, P_chg_lim = bat.P_dis_max * tmult, bat.P_chg_max * tmult
        if P_b > P_dis and T2 > 0 and abs(w2) > 1:
            dT = min((P_b - P_dis) / abs(w2), T2)
            T2 -= dT; P2 -= dT * w2; P_b -= dT * w2
        if P_b < -P_chg_lim and T2 < 0 and abs(w2) > 1:
            dT = min((-P_chg_lim - P_b) / abs(w2), -T2)
            T2 += dT; P2 += dT * w2; P_b += dT * w2
        P_lim_bias = _clip(P_lim_bias + (max(P_b - 0.85 * P_dis, 0) - max(-P_b - 0.85 * P_chg_lim, 0)
                                          - 0.2 * P_lim_bias) * dt, -10e3, 20e3)
        R0 = bat.R0_cell * ns * math.exp(bat.Ea_R * (1 / (T_bat + 273.15) - 1 / 298.15)) \
            * (1 + 1.5 * f_bat[i])
        ocv = float(np.interp(soc, soc_pts, ocv_pts))
        Ve = ocv - V1
        disc = Ve * Ve - 4 * R0 * P_b
        if disc < 0:
            disc = 0.0
        I_b = (Ve - math.sqrt(disc)) / (2 * R0)
        V_b = Ve - R0 * I_b
        V1 += (-V1 / bat.tau1 + I_b / C1) * dt
        soc -= I_b * dt / Q_As
        Q_bat = I_b * I_b * R0 + V1 * V1 / R1p
        fan = _clip((T_bat - 30) / 10, 0, 1)
        T_bat += (Q_bat - (bat.hA0 + bat.hA_fan * fan) * (T_bat - T_cab)) * dt / bat.C_th

        # ---------------- wheel force, brakes, longitudinal dynamics
        T_ring = T_r_eng + ps.k_mg2 * T2
        T_w = T_ring * ps.FD * (ps.eta_gear if T_ring > 0 else 1 / ps.eta_gear)
        F_w = T_w / r_f
        F_regen = max(-F_w, 0.0) if F_b > 0 else 0.0
        F_fric = max(F_b - F_regen, 0.0) if F_b > 0 else 0.0
        if v < 0.05 and F_b > 0:
            F_fric = F_b
        F_net = F_w - F_aero - F_roll - F_grade - F_fric - F_drag
        a = F_net / m_eff
        if v <= 0.0 and a < 0:
            a = 0.0
        if v < 0.05 and F_b > 0 and a < 0:
            a = 0.0
        v = max(v + a * dt, 0.0)
        s_pos += v * dt
        a_prev = a
        pedal_prev = pedal

        # brake thermal
        P_fric = F_fric * v
        hB = brk.h0 + brk.h_v * v
        T_bf += (brk.front_share * P_fric + F_drag * v - hB * (T_bf - Tamb)) * dt / brk.C_front
        T_br += ((1 - brk.front_share) * P_fric - 0.8 * hB * (T_br - Tamb)) * dt / brk.C_rear

        # ---------------- engine thermal and exhaust
        th_open = _clip((T_cool - eng.thermo_open[0]) / (eng.thermo_open[1] - eng.thermo_open[0]), 0, 1)
        if f_th[i] > 0:                         # stuck at 60 % (sev 0.5) / 100 % (sev 1)
            th_open = max(th_open, 0.2 + 0.8 * f_th[i])
        mdot = eng.mdot_rad_max * th_open + 0.002
        if T_cool > eng.T_fan_on: fan_on = True
        elif T_cool < eng.T_fan_on - 5: fan_on = False
        UA_air = eng.UA_rad0 + eng.UA_rad_v * v + (eng.UA_fan if fan_on else 0.0)
        Q_rad = mdot * eng.cp_cool * (T_cool - T_rad)
        Q_in = eng.q_cool_frac * fuel / 1000 * eng.LHV + (0.3 * Tf * w_e if not firing else 0.0)
        T_cool += (Q_in - Q_rad - Q_hc - eng.UA_amb * (T_cool - Tamb)) * dt / eng.C_th
        T_rad += (Q_rad - UA_air * (T_rad - Tamb)) * dt / eng.C_rad
        if firing:
            T_ss = Tamb + 350 + 400 * math.sqrt(max(P_e_tgt, 0) / eng.P_max) + 600 * mis
            T_cat += (T_ss - T_cat) * dt / 35.0
        else:
            T_cat += (Tamb + 20 - T_cat) * dt / 300.0

        # ---------------- electric drive thermal
        pump = 1 - 0.75 * f_pump[i]
        gfac = 0.45 + 0.55 * pump
        G1, G2, Gi = m1.G_th * gfac, m2.G_th * gfac, inv.G_th * gfac
        q1, q2, qi = G1 * (T_w1 - T_edu), G2 * (T_w2 - T_edu), Gi * (T_inv - T_edu)
        T_w1 += (loss1 - q1) * dt / m1.C_th
        T_w2 += (loss2 - q2) * dt / m2.C_th
        T_inv += (P_inv_loss - qi) * dt / inv.C_th
        UA_lt = pump * (edu.UA0 + edu.UA_v * v) + (edu.UA_fan if fan_on else 0.0) * pump
        T_edu += (q1 + q2 + qi - UA_lt * (T_edu - Tamb)) * dt / edu.C_th

        # ---------------- cabin
        UA_c = cab.UA0 + cab.UA_v * v
        T_cab += (UA_c * (Tamb - T_cab) + solar[i] + Q_hvac + 90.0) * dt / cab.C_th

        # ---------------- wheel speeds
        half = ch.track / 2
        vw = (v - yaw * half, v + yaw * half, v - yaw * half, v + yaw * half)
        Ffront = F_w - brk.front_share * F_fric - F_drag
        Frear = -(1 - brk.front_share) * F_fric
        Fx = (Ffront / 2, Ffront / 2, Frear / 2, Frear / 2)
        for j, key in enumerate(("w_FL", "w_FR", "w_RL", "w_RR")):
            slip = _clip(Fx[j] / (ty.slip_stiff * Fz[j]), -0.2, 0.2) if v > 0.3 else 0.0
            O[key][i] = max(vw[j], 0) * (1 + slip) / re[j]
        w_front_prev = 0.5 * (O["w_FL"][i] + O["w_FR"][i])

        # ---------------- record
        O["v_veh"][i] = v
        O["a_long"][i] = a + ch.g * sin_th
        O["a_long_kin"][i] = a
        O["a_lat"][i] = a_lat
        O["a_lat_kin"][i] = a_lat
        O["yaw_rate"][i] = yaw * 180 / math.pi
        O["steer_angle"][i] = steer
        O["pedal_acc"][i] = pedal * 100
        for j, sfx in enumerate(("FL", "FR", "RL", "RR")):
            O["p_tire_" + sfx][i] = p[j]
            O["Fz_" + sfx][i] = Fz[j]
            O["k_t_" + sfx][i] = kt[j]
        O["brake_pressure"][i] = F_b / brk.N_per_bar
        O["T_brake_F"][i] = T_bf
        O["T_brake_R"][i] = T_br
        O["n_eng"][i] = w_e / RPM
        O["tq_eng"][i] = T_cmd
        O["tq_eng_actual"][i] = T_act
        O["fuel_rate"][i] = fuel
        O["T_cool_eng"][i] = T_cool
        O["T_cat"][i] = T_cat
        O["n_MG1"][i] = w1 / RPM
        O["tq_MG1"][i] = T1
        O["n_MG2"][i] = w2 / RPM
        O["tq_MG2"][i] = T2
        O["T_MG1"][i] = T_w1
        O["T_MG2"][i] = T_w2
        O["T_inv"][i] = T_inv
        O["I_batt"][i] = I_b
        O["V_batt"][i] = V_b
        O["SOC"][i] = soc * 100
        O["T_batt"][i] = T_bat
        O["V_12"][i] = V12
        O["I_dcdc"][i] = I_dc
        O["T_cool_edu"][i] = T_edu
        O["T_rad_out"][i] = T_rad
        O["T_cabin"][i] = T_cab
        O["P_ac"][i] = P_ac
        O["s"][i] = s_pos
        O["grade"][i] = theta
        O["engine_on"][i] = float(eng_on)
        O["P_batt"][i] = P_b
        O["F_fric"][i] = F_fric
        O["Q_hvac"][i] = Q_hvac
        O["fault_level"][i] = fault_level[i]

    # high-rate series needed by the suspension model
    hi = {"t": t, "v": out["v_veh"].copy(), "s": out["s"].copy(),
          "a_x": out["a_long_kin"].copy(), "a_y": out["a_lat_kin"].copy(),
          "k_t": np.stack([out["k_t_" + k] for k in ("FL", "FR", "RL", "RR")], 1)}
    # block-average to output rate
    k = int(round(out_dt / dt))
    m = n // k
    dec = {key: val[: m * k].reshape(m, k).mean(1) for key, val in out.items()}
    dec["t"] = t[: m * k].reshape(m, k)[:, 0]
    return dec, hi
