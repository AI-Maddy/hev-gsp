"""Export the communication description:
  * comm_matrix.csv       - COM signal / I-PDU / CAN FD frame matrix
  * CANFD_F.dbc, CANFD_R.dbc
  * hev_zonal_system.arxml - AUTOSAR R4 system description (CAN clusters,
    frames, I-SIGNAL-I-PDUs with timing, I-SIGNALs, COMPU-METHODs,
    ECU-INSTANCEs with I-PDU groups, E2E protection set)
  * someip_deployment.csv - ara::com service/event -> SOME/IP IDs
"""
import csv, os
from xml.sax.saxutils import escape
import cantools
from cantools.database.can import Message, Signal, Node
from cantools.database.conversion import BaseConversion
from . import ee_arch as EA

E2E_CAT = {"P11": "PROFILE_11", "P05": "PROFILE_05"}


def _e2e_signals(p):
    if p.e2e == "P11":
        return [("CRC", 0, 8), ("Counter", 8, 4)]
    if p.e2e == "P05":
        return [("CRC", 0, 16), ("Counter", 16, 8)]
    return []


def write_csv(pdus, out):
    with open(os.path.join(out, "comm_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pdu", "sender", "bus", "can_id_hex", "frame_format", "dlc_bytes", "ComTxModeMode",
                    "ComTxModeTimePeriod_ms", "ComMinimumDelayTime_ms", "e2e_profile", "ComTimeout_ms",
                    "routed_to_someip", "signal", "start_bit", "length_bits", "factor", "offset", "unit",
                    "ComTransferProperty", "ComFilterAlgorithm", "filter_mask_hex", "J1979"])
        for p in pdus:
            rows = p.signals or [None]
            for s in rows:
                base = [p.name, p.sender, p.bus, f"0x{p.can_id:03X}", "CAN FD base (11-bit)", p.payload,
                        p.mode, p.period_ms, p.mdt_ms, p.e2e, p.timeout_ms, int(p.routed)]
                if s is None:
                    w.writerow(base + [""] * 10)
                    continue
                mask = ((1 << s.length) - 1) & ~((1 << s.mask_bits) - 1)
                w.writerow(base + [s.name, s.start, s.length, s.factor, s.offset, s.unit,
                                   "TRIGGERED_ON_CHANGE" if s.trigger else "PENDING",
                                   "MASKED_NEW_DIFFERS_MASKED_OLD" if s.trigger else "ALWAYS",
                                   f"0x{mask:X}" if s.trigger else "",
                                   EA.J1979.get(s.name, ("",))[0]])


def write_dbc(pdus, out):
    for bus in EA.CAN_BUSES:
        msgs = []
        nodes = sorted({p.sender for p in pdus if p.bus == bus} | {z for z, b in EA.ZONES.items() if b == bus})
        for p in pdus:
            if p.bus != bus:
                continue
            sigs = [Signal(f"{p.name}_{n}", st, ln) for n, st, ln in _e2e_signals(p)]
            for s in p.signals:
                sigs.append(Signal(s.name, s.start, s.length,
                                   conversion=BaseConversion.factory(s.factor, s.offset),
                                   minimum=s.offset, maximum=s.offset + s.factor * ((1 << s.length) - 1),
                                   unit=s.unit, receivers=[EA.ZONE_OF.get(p.sender, "")] if p.routed else []))
            if not sigs:
                sigs = [Signal(f"{p.name}_Payload", 0, min(64, 8 * p.payload))]
            msgs.append(Message(p.can_id, p.name, p.payload, sigs, senders=[p.sender],
                                cycle_time=int(p.period_ms), is_fd=True, bus_name=bus,
                                comment=f"ComTxModeMode={p.mode}; MDT={p.mdt_ms} ms; E2E={p.e2e or 'none'}"))
        db = cantools.database.Database(messages=msgs, nodes=[Node(n) for n in nodes])
        cantools.database.dump_file(db, os.path.join(out, f"{bus}.dbc"))


def write_arxml(pdus, out):
    X = []
    a = X.append
    a('<?xml version="1.0" encoding="UTF-8"?>')
    a('<AUTOSAR xmlns="http://autosar.org/schema/r4.0" '
      'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
      'xsi:schemaLocation="http://autosar.org/schema/r4.0 AUTOSAR_00051.xsd">')
    a('<AR-PACKAGES>')
    def pkg(name, elems):
        a(f'<AR-PACKAGE><SHORT-NAME>{name}</SHORT-NAME><ELEMENTS>')
        elems()
        a('</ELEMENTS></AR-PACKAGE>')
    frames_by_bus = {b: [p for p in pdus if p.bus == b] for b in EA.CAN_BUSES}

    def clusters():
        for b, spec in EA.CAN_BUSES.items():
            a(f'<CAN-CLUSTER><SHORT-NAME>{b}</SHORT-NAME><CAN-CLUSTER-VARIANTS><CAN-CLUSTER-CONDITIONAL>')
            a(f'<BAUDRATE>{int(spec["nominal"])}</BAUDRATE><PHYSICAL-CHANNELS><CAN-PHYSICAL-CHANNEL>'
              f'<SHORT-NAME>{b}_CH</SHORT-NAME><FRAME-TRIGGERINGS>')
            for p in frames_by_bus[b]:
                a(f'<CAN-FRAME-TRIGGERING><SHORT-NAME>FT_{p.name}</SHORT-NAME>'
                  f'<FRAME-PORT-REFS><FRAME-PORT-REF DEST="FRAME-PORT">/ECUs/{p.sender}/CN_{b}/FP_{p.name}</FRAME-PORT-REF></FRAME-PORT-REFS>'
                  f'<FRAME-REF DEST="CAN-FRAME">/Frames/{p.name}</FRAME-REF>'
                  f'<PDU-TRIGGERINGS><PDU-TRIGGERING-REF-CONDITIONAL><PDU-TRIGGERING-REF DEST="PDU-TRIGGERING">/Clusters/{b}/{b}_CH/PT_{p.name}</PDU-TRIGGERING-REF></PDU-TRIGGERING-REF-CONDITIONAL></PDU-TRIGGERINGS>'
                  f'<CAN-ADDRESSING-MODE>STANDARD</CAN-ADDRESSING-MODE><CAN-FRAME-RX-BEHAVIOR>CAN-FD</CAN-FRAME-RX-BEHAVIOR>'
                  f'<CAN-FRAME-TX-BEHAVIOR>CAN-FD</CAN-FRAME-TX-BEHAVIOR><IDENTIFIER>{p.can_id}</IDENTIFIER></CAN-FRAME-TRIGGERING>')
            a('</FRAME-TRIGGERINGS><PDU-TRIGGERINGS>')
            for p in frames_by_bus[b]:
                a(f'<PDU-TRIGGERING><SHORT-NAME>PT_{p.name}</SHORT-NAME>'
                  f'<I-PDU-REF DEST="I-SIGNAL-I-PDU">/PDUs/{p.name}</I-PDU-REF></PDU-TRIGGERING>')
            a('</PDU-TRIGGERINGS></CAN-PHYSICAL-CHANNEL></PHYSICAL-CHANNELS>')
            a(f'<CAN-FD-BAUDRATE>{int(spec["data"])}</CAN-FD-BAUDRATE>')
            a('</CAN-CLUSTER-CONDITIONAL></CAN-CLUSTER-VARIANTS></CAN-CLUSTER>')

    def frames():
        for p in pdus:
            a(f'<CAN-FRAME><SHORT-NAME>{p.name}</SHORT-NAME><FRAME-LENGTH>{p.payload}</FRAME-LENGTH>'
              f'<PDU-TO-FRAME-MAPPINGS><PDU-TO-FRAME-MAPPING><SHORT-NAME>PFM_{p.name}</SHORT-NAME>'
              f'<PACKING-BYTE-ORDER>MOST-SIGNIFICANT-BYTE-LAST</PACKING-BYTE-ORDER>'
              f'<PDU-REF DEST="I-SIGNAL-I-PDU">/PDUs/{p.name}</PDU-REF><START-POSITION>0</START-POSITION>'
              f'</PDU-TO-FRAME-MAPPING></PDU-TO-FRAME-MAPPINGS></CAN-FRAME>')

    def sig_list(p):
        out_ = [(n, st, ln, None) for n, st, ln in _e2e_signals(p)]
        out_ += [(s.name, s.start, s.length, s) for s in p.signals]
        if not out_:
            out_ = [("Payload", 0, min(64, 8 * p.payload), None)]
        return out_

    def ipdus():
        for p in pdus:
            a(f'<I-SIGNAL-I-PDU><SHORT-NAME>{p.name}</SHORT-NAME><LENGTH>{p.payload}</LENGTH>'
              f'<I-PDU-TIMING-SPECIFICATIONS><I-PDU-TIMING>')
            if p.mdt_ms:
                a(f'<MINIMUM-DELAY>{p.mdt_ms / 1000:g}</MINIMUM-DELAY>')
            a('<TRANSMISSION-MODE-DECLARATION><TRANSMISSION-MODE-TRUE-TIMING>'
              f'<CYCLIC-TIMING><TIME-PERIOD><VALUE>{p.period_ms / 1000:g}</VALUE></TIME-PERIOD></CYCLIC-TIMING>')
            if p.mode == "MIXED":
                a('<EVENT-CONTROLLED-TIMING><NUMBER-OF-REPETITIONS>0</NUMBER-OF-REPETITIONS></EVENT-CONTROLLED-TIMING>')
            a('</TRANSMISSION-MODE-TRUE-TIMING></TRANSMISSION-MODE-DECLARATION></I-PDU-TIMING></I-PDU-TIMING-SPECIFICATIONS>')
            a('<I-SIGNAL-TO-PDU-MAPPINGS>')
            for n, st, ln, s in sig_list(p):
                tp = "TRIGGERED-ON-CHANGE" if (s is not None and s.trigger) else "PENDING"
                a(f'<I-SIGNAL-TO-I-PDU-MAPPING><SHORT-NAME>M_{n}</SHORT-NAME>'
                  f'<I-SIGNAL-REF DEST="I-SIGNAL">/ISignals/{f"{p.name}_{n}" if s is None else s.name}</I-SIGNAL-REF>'
                  f'<PACKING-BYTE-ORDER>MOST-SIGNIFICANT-BYTE-LAST</PACKING-BYTE-ORDER>'
                  f'<START-POSITION>{st}</START-POSITION><TRANSFER-PROPERTY>{tp}</TRANSFER-PROPERTY>'
                  f'</I-SIGNAL-TO-I-PDU-MAPPING>')
            a('</I-SIGNAL-TO-PDU-MAPPINGS><UNUSED-BIT-PATTERN>255</UNUSED-BIT-PATTERN></I-SIGNAL-I-PDU>')

    def isignals():
        for p in pdus:
            for n, st, ln, s in sig_list(p):
                sname = f"{p.name}_{n}" if s is None else s.name
                a(f'<I-SIGNAL><SHORT-NAME>{sname}</SHORT-NAME>')
                a('<INIT-VALUE><NUMERICAL-VALUE-SPECIFICATION><VALUE>0</VALUE></NUMERICAL-VALUE-SPECIFICATION></INIT-VALUE>')
                a(f'<LENGTH>{ln}</LENGTH><NETWORK-REPRESENTATION-PROPS><SW-DATA-DEF-PROPS-VARIANTS><SW-DATA-DEF-PROPS-CONDITIONAL>'
                  f'<BASE-TYPE-REF DEST="SW-BASE-TYPE">/BaseTypes/{"UINT64" if ln > 32 else "UINT32" if ln > 16 else "UINT16" if ln > 8 else "UINT8"}</BASE-TYPE-REF>')
                if s is not None:
                    a(f'<COMPU-METHOD-REF DEST="COMPU-METHOD">/CompuMethods/CM_{s.name}</COMPU-METHOD-REF>')
                a('</SW-DATA-DEF-PROPS-CONDITIONAL></SW-DATA-DEF-PROPS-VARIANTS></NETWORK-REPRESENTATION-PROPS>')
                a(f'<SYSTEM-SIGNAL-REF DEST="SYSTEM-SIGNAL">/SystemSignals/{sname}</SYSTEM-SIGNAL-REF></I-SIGNAL>')

    def syssignals():
        for p in pdus:
            for n, st, ln, s in sig_list(p):
                sname = f"{p.name}_{n}" if s is None else s.name
                a(f'<SYSTEM-SIGNAL><SHORT-NAME>{sname}</SHORT-NAME>')
                if s is not None:
                    a('<PHYSICAL-PROPS><SW-DATA-DEF-PROPS-VARIANTS><SW-DATA-DEF-PROPS-CONDITIONAL>'
                      f'<COMPU-METHOD-REF DEST="COMPU-METHOD">/CompuMethods/CM_{s.name}</COMPU-METHOD-REF>'
                      f'<UNIT-REF DEST="UNIT">/Units/U_{s.name}</UNIT-REF>'
                      '</SW-DATA-DEF-PROPS-CONDITIONAL></SW-DATA-DEF-PROPS-VARIANTS></PHYSICAL-PROPS>')
                a('</SYSTEM-SIGNAL>')

    def compu():
        for p in pdus:
            for s in p.signals:
                hi = (1 << s.length) - 1
                a(f'<COMPU-METHOD><SHORT-NAME>CM_{s.name}</SHORT-NAME><CATEGORY>LINEAR</CATEGORY>'
                  f'<UNIT-REF DEST="UNIT">/Units/U_{s.name}</UNIT-REF>'
                  f'<COMPU-INTERNAL-TO-PHYS><COMPU-SCALES><COMPU-SCALE>'
                  f'<LOWER-LIMIT INTERVAL-TYPE="CLOSED">0</LOWER-LIMIT><UPPER-LIMIT INTERVAL-TYPE="CLOSED">{hi}</UPPER-LIMIT>'
                  f'<COMPU-RATIONAL-COEFFS><COMPU-NUMERATOR><V>{s.offset:g}</V><V>{s.factor:g}</V></COMPU-NUMERATOR>'
                  f'<COMPU-DENOMINATOR><V>1</V></COMPU-DENOMINATOR></COMPU-RATIONAL-COEFFS>'
                  f'</COMPU-SCALE></COMPU-SCALES></COMPU-INTERNAL-TO-PHYS></COMPU-METHOD>')

    def units():
        for p in pdus:
            for s in p.signals:
                a(f'<UNIT><SHORT-NAME>U_{s.name}</SHORT-NAME><DISPLAY-NAME>{escape(s.unit)}</DISPLAY-NAME></UNIT>')

    def basetypes():
        for n, bits in (("UINT8", 8), ("UINT16", 16), ("UINT32", 32), ("UINT64", 64)):
            a(f'<SW-BASE-TYPE><SHORT-NAME>{n}</SHORT-NAME><CATEGORY>FIXED_LENGTH</CATEGORY>'
              f'<BASE-TYPE-SIZE>{bits}</BASE-TYPE-SIZE><BASE-TYPE-ENCODING>NONE</BASE-TYPE-ENCODING></SW-BASE-TYPE>')

    ecus = EA.CP_ECUS + list(EA.ZONES)

    def ecu_instances():
        for e in ecus:
            a(f'<ECU-INSTANCE><SHORT-NAME>{e}</SHORT-NAME><ASSOCIATED-COM-I-PDU-GROUP-REFS>'
              f'<ASSOCIATED-COM-I-PDU-GROUP-REF DEST="I-SIGNAL-I-PDU-GROUP">/PduGroups/{e}_Tx</ASSOCIATED-COM-I-PDU-GROUP-REF>')
            if e in EA.ZONES:
                a(f'<ASSOCIATED-COM-I-PDU-GROUP-REF DEST="I-SIGNAL-I-PDU-GROUP">/PduGroups/{e}_Rx</ASSOCIATED-COM-I-PDU-GROUP-REF>')
            a('</ASSOCIATED-COM-I-PDU-GROUP-REFS><CONNECTORS>')
            bus = EA.BUS_OF.get(e) or EA.ZONES[e]
            a(f'<CAN-COMMUNICATION-CONNECTOR><SHORT-NAME>CN_{bus}</SHORT-NAME><ECU-COMM-PORT-INSTANCES>')
            for p in pdus:
                if p.sender == e:
                    a(f'<FRAME-PORT><SHORT-NAME>FP_{p.name}</SHORT-NAME><COMMUNICATION-DIRECTION>OUT</COMMUNICATION-DIRECTION></FRAME-PORT>')
            a('</ECU-COMM-PORT-INSTANCES></CAN-COMMUNICATION-CONNECTOR></CONNECTORS></ECU-INSTANCE>')

    def pdu_groups():
        for e in ecus:
            groups = [("Tx", "OUT", [p for p in pdus if p.sender == e])]
            if e in EA.ZONES:
                groups.append(("Rx", "IN", [p for p in pdus if p.routed and EA.ZONE_OF.get(p.sender) == e]))
            for g, d, lst in groups:
                a(f'<I-SIGNAL-I-PDU-GROUP><SHORT-NAME>{e}_{g}</SHORT-NAME><COMMUNICATION-DIRECTION>{d}</COMMUNICATION-DIRECTION>'
                  f'<I-SIGNAL-I-PDUS>')
                for p in lst:
                    a(f'<I-SIGNAL-I-PDU-REF-CONDITIONAL><I-SIGNAL-I-PDU-REF DEST="I-SIGNAL-I-PDU">/PDUs/{p.name}'
                      f'</I-SIGNAL-I-PDU-REF></I-SIGNAL-I-PDU-REF-CONDITIONAL>')
                a('</I-SIGNAL-I-PDUS></I-SIGNAL-I-PDU-GROUP>')

    def e2e():
        a('<END-TO-END-PROTECTION-SET><SHORT-NAME>E2E_Set</SHORT-NAME><END-TO-END-PROTECTIONS>')
        for k, p in enumerate([p for p in pdus if p.e2e]):
            a(f'<END-TO-END-PROTECTION><SHORT-NAME>E2E_{p.name}</SHORT-NAME><END-TO-END-PROFILE>'
              f'<CATEGORY>{E2E_CAT[p.e2e]}</CATEGORY><DATA-IDS><DATA-ID>{0x100 + k}</DATA-ID></DATA-IDS>'
              f'<MAX-DELTA-COUNTER>1</MAX-DELTA-COUNTER></END-TO-END-PROFILE>'
              f'<END-TO-END-PROTECTION-I-SIGNAL-I-PDUS><END-TO-END-PROTECTION-I-SIGNAL-I-PDU>'
              f'<SHORT-NAME>E2EP_{p.name}</SHORT-NAME><I-SIGNAL-I-PDU-REF DEST="I-SIGNAL-I-PDU">/PDUs/{p.name}</I-SIGNAL-I-PDU-REF>'
              f'</END-TO-END-PROTECTION-I-SIGNAL-I-PDU></END-TO-END-PROTECTION-I-SIGNAL-I-PDUS></END-TO-END-PROTECTION>')
        a('</END-TO-END-PROTECTIONS></END-TO-END-PROTECTION-SET>')

    for name, fn in [("Clusters", clusters), ("Frames", frames), ("PDUs", ipdus), ("ISignals", isignals),
                     ("SystemSignals", syssignals), ("CompuMethods", compu), ("Units", units),
                     ("BaseTypes", basetypes), ("ECUs", ecu_instances), ("PduGroups", pdu_groups),
                     ("E2E", e2e)]:
        pkg(name, fn)
    a('</AR-PACKAGES></AUTOSAR>')
    path = os.path.join(out, "hev_zonal_system.arxml")
    with open(path, "w") as f:
        f.write("\n".join(X))
    return path


def write_someip(pdus, out):
    bypdu = {p.name: p for p in pdus}
    with open(os.path.join(out, "someip_deployment.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["service", "service_id_hex", "instance_id", "provider", "consumers", "eventgroup_id",
                    "event", "event_id_arxml_hex", "event_id_on_wire_hex", "source_ipdu", "cycle_ms", "payload_bytes", "transport",
                    "someip_message_type", "graph_signals"])
        for sname, sid, prov, eg, plist in EA.SERVICES:
            cons = "CVC;TCU" if prov != "CVC" else "ZC_FRONT;ZC_REAR"
            items = [(n, bypdu[n]) for n in plist] or [(sname + "_Evt", None)]
            for k, (n, p) in enumerate(items):
                sig = ";".join(s.name for s in p.signals) if p else ("pedal_acc" if sname == "DriverInput" else "")
                w.writerow([sname, f"0x{sid:04X}", 1, prov, cons, eg, n, f"0x{0x0001 + k:04X}", f"0x{0x8001 + k:04X}", n if p else "",
                            p.period_ms if p else 10, (p.payload if p else 8), "UDP (unicast/multicast)",
                            "NOTIFICATION (0x02)", sig])


def export_all(pdus, out):
    os.makedirs(out, exist_ok=True)
    write_csv(pdus, out)
    write_dbc(pdus, out)
    ax = write_arxml(pdus, out)
    write_someip(pdus, out)
    return ax
