"""Adaptive Platform (CAPI) model of the service-oriented part of the E/E layer.

Writes ARXML in the structure used by the AUTOSAR CAPI v1.0.0 samples
(samples/helloworld-cm/model): STD-CPP-IMPLEMENTATION-DATA-TYPEs,
SERVICE-INTERFACEs, SOMEIP-SERVICE-INTERFACE-DEPLOYMENTs,
PROVIDED/REQUIRED-SOMEIP-SERVICE-INSTANCEs, and EXECUTABLEs with
ADAPTIVE-APPLICATION-SW-COMPONENT-TYPE ports for the CVC energy-management
application and the TCU telemetry application. The files can be fed to
CAPI's generator: aragen -g CODE -e /HEV/Apps/exe/energy_mgmt ...
"""
import os
from . import ee_arch as EA

HDR = ('<?xml version="1.0" encoding="UTF-8"?>\n'
       '<AUTOSAR xmlns="http://autosar.org/schema/r4.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
       'xsi:schemaLocation="http://autosar.org/schema/r4.0 AUTOSAR_00049.xsd">\n<AR-PACKAGES>\n')
FTR = '</AR-PACKAGES>\n</AUTOSAR>\n'
NS = ('<NAMESPACES><SYMBOL-PROPS><SHORT-NAME>hev</SHORT-NAME><SYMBOL>hev</SYMBOL></SYMBOL-PROPS>'
      '<SYMBOL-PROPS><SHORT-NAME>zonal</SHORT-NAME><SYMBOL>zonal</SYMBOL></SYMBOL-PROPS></NAMESPACES>')
FLOAT = "/AUTOSAR/StdTypes/float"

# CVC -> zones command events (EnergyManagement service)
EM_EVENTS = [("EngineRequest", ["tq_eng_req", "n_eng_req", "engine_on_req"]),
             ("MotorRequest", ["tq_MG1_req", "tq_MG2_req"]),
             ("ThermalRequest", ["edu_pump_duty", "fan_duty", "heater_valve"]),
             ("BatteryRequest", ["P_dis_limit", "P_chg_limit"]),
             ("ClimateRequest", ["T_set", "Q_req"])]


def _services(pdus):
    bypdu = {p.name: p for p in pdus}
    out = []
    for sname, sid, prov, eg, plist in EA.SERVICES:
        if sname == "DriverInput":
            evs = [("PedalEvent", ["pedal_acc"])]
        elif sname == "EnergyManagement":
            evs = EM_EVENTS
        else:
            evs = [(p + "Event", [s.name for s in bypdu[p].signals]) for p in plist]
        out.append((sname, sid, prov, eg, evs))
    return out


def _pkg(name, body):
    return f'<AR-PACKAGE><SHORT-NAME>{name}</SHORT-NAME><ELEMENTS>\n{body}</ELEMENTS></AR-PACKAGE>\n'


def write_adaptive(pdus, out):
    os.makedirs(out, exist_ok=True)
    svcs = _services(pdus)
    # ---- data types
    body = ""
    for sname, _, _, _, evs in svcs:
        for ev, fields in evs:
            body += (f'<STD-CPP-IMPLEMENTATION-DATA-TYPE><SHORT-NAME>{ev}Type</SHORT-NAME><CATEGORY>STRUCTURE</CATEGORY>'
                     f'{NS}<SUB-ELEMENTS>')
            for f in fields:
                body += (f'<CPP-IMPLEMENTATION-DATA-TYPE-ELEMENT><SHORT-NAME>{f}</SHORT-NAME><TYPE-REFERENCE>'
                         f'<TYPE-REFERENCE-REF DEST="STD-CPP-IMPLEMENTATION-DATA-TYPE">{FLOAT}</TYPE-REFERENCE-REF>'
                         f'</TYPE-REFERENCE></CPP-IMPLEMENTATION-DATA-TYPE-ELEMENT>')
            body += '</SUB-ELEMENTS><TYPE-EMITTER>TYPE_EMITTER_ARA</TYPE-EMITTER></STD-CPP-IMPLEMENTATION-DATA-TYPE>\n'
    open(os.path.join(out, "hev_datatypes.arxml"), "w").write(HDR + _pkg("HEVTypes", body) + FTR)

    # ---- service interfaces
    body = ""
    for sname, _, _, _, evs in svcs:
        body += (f'<SERVICE-INTERFACE><SHORT-NAME>{sname}</SHORT-NAME>{NS}'
                 '<MAJOR-VERSION>1</MAJOR-VERSION><MINOR-VERSION>0</MINOR-VERSION><EVENTS>')
        for ev, _ in evs:
            body += (f'<VARIABLE-DATA-PROTOTYPE><SHORT-NAME>{ev}</SHORT-NAME>'
                     f'<TYPE-TREF DEST="STD-CPP-IMPLEMENTATION-DATA-TYPE">/HEVTypes/{ev}Type</TYPE-TREF>'
                     '</VARIABLE-DATA-PROTOTYPE>')
        body += '</EVENTS></SERVICE-INTERFACE>\n'
    open(os.path.join(out, "hev_service_interfaces.arxml"), "w").write(HDR + _pkg("HEVServiceInterfaces", body) + FTR)

    # ---- SOME/IP service interface deployments
    body = ""
    for sname, sid, _, eg, evs in svcs:
        body += (f'<SOMEIP-SERVICE-INTERFACE-DEPLOYMENT UUID=""><SHORT-NAME>{sname}</SHORT-NAME><EVENT-DEPLOYMENTS>')
        for k, (ev, _) in enumerate(evs):
            body += (f'<SOMEIP-EVENT-DEPLOYMENT><SHORT-NAME>{ev}</SHORT-NAME>'
                     f'<EVENT-REF DEST="VARIABLE-DATA-PROTOTYPE">/HEVServiceInterfaces/{sname}/{ev}</EVENT-REF>'
                     f'<EVENT-ID>0x{0x0001 + k:04X}</EVENT-ID><TRANSPORT-PROTOCOL>UDP</TRANSPORT-PROTOCOL>'
                     '</SOMEIP-EVENT-DEPLOYMENT>')
        body += (f'</EVENT-DEPLOYMENTS><SERVICE-INTERFACE-REF DEST="SERVICE-INTERFACE">/HEVServiceInterfaces/{sname}'
                 f'</SERVICE-INTERFACE-REF><EVENT-GROUPS><SOMEIP-EVENT-GROUP><SHORT-NAME>Eventgroup{eg}</SHORT-NAME>'
                 f'<EVENT-GROUP-ID>{eg}</EVENT-GROUP-ID><EVENT-REFS>')
        for ev, _ in evs:
            body += f'<EVENT-REF DEST="SOMEIP-EVENT-DEPLOYMENT">/SomeipDeploymentsHEV/{sname}/{ev}</EVENT-REF>'
        body += (f'</EVENT-REFS></SOMEIP-EVENT-GROUP></EVENT-GROUPS><SERVICE-INTERFACE-ID>{sid}</SERVICE-INTERFACE-ID>'
                 '<SERVICE-INTERFACE-VERSION><MAJOR-VERSION>1</MAJOR-VERSION><MINOR-VERSION>0</MINOR-VERSION>'
                 '</SERVICE-INTERFACE-VERSION></SOMEIP-SERVICE-INTERFACE-DEPLOYMENT>\n')
    open(os.path.join(out, "hev_someip_deployments.arxml"), "w").write(HDR + _pkg("SomeipDeploymentsHEV", body) + FTR)

    # ---- provided / required service instances
    body = ""
    consumers = {"CVC": [s for s in svcs if s[2] != "CVC"], "TCU": [s for s in svcs if s[2] != "CVC"],
                 "ZC_FRONT": [s for s in svcs if s[2] == "CVC"], "ZC_REAR": [s for s in svcs if s[2] == "CVC"]}
    for sname, sid, prov, eg, _ in svcs:
        body += (f'<PROVIDED-SOMEIP-SERVICE-INSTANCE><SHORT-NAME>P_{prov}_{sname}</SHORT-NAME>'
                 f'<SERVICE-INTERFACE-DEPLOYMENT-REF DEST="SOMEIP-SERVICE-INTERFACE-DEPLOYMENT">/SomeipDeploymentsHEV/{sname}'
                 f'</SERVICE-INTERFACE-DEPLOYMENT-REF><PROVIDED-EVENT-GROUPS><SOMEIP-PROVIDED-EVENT-GROUP>'
                 f'<SHORT-NAME>PEG{eg}</SHORT-NAME><EVENT-GROUP-REF DEST="SOMEIP-EVENT-GROUP">/SomeipDeploymentsHEV/{sname}/Eventgroup{eg}'
                 f'</EVENT-GROUP-REF><MULTICAST-THRESHOLD>2</MULTICAST-THRESHOLD></SOMEIP-PROVIDED-EVENT-GROUP>'
                 f'</PROVIDED-EVENT-GROUPS><SERVICE-INSTANCE-ID>1</SERVICE-INSTANCE-ID></PROVIDED-SOMEIP-SERVICE-INSTANCE>\n')
    for cons, lst in consumers.items():
        for sname, sid, prov, eg, _ in lst:
            body += (f'<REQUIRED-SOMEIP-SERVICE-INSTANCE><SHORT-NAME>R_{cons}_{sname}</SHORT-NAME>'
                     f'<SERVICE-INTERFACE-DEPLOYMENT-REF DEST="SOMEIP-SERVICE-INTERFACE-DEPLOYMENT">/SomeipDeploymentsHEV/{sname}'
                     f'</SERVICE-INTERFACE-DEPLOYMENT-REF><REQUIRED-EVENT-GROUPS><SOMEIP-REQUIRED-EVENT-GROUP>'
                     f'<SHORT-NAME>REG{eg}</SHORT-NAME><EVENT-GROUP-REF DEST="SOMEIP-EVENT-GROUP">/SomeipDeploymentsHEV/{sname}/Eventgroup{eg}'
                     f'</EVENT-GROUP-REF></SOMEIP-REQUIRED-EVENT-GROUP></REQUIRED-EVENT-GROUPS>'
                     f'<REQUIRED-MINOR-VERSION>0</REQUIRED-MINOR-VERSION><REQUIRED-SERVICE-INSTANCE-ID>1</REQUIRED-SERVICE-INSTANCE-ID>'
                     f'<VERSION-DRIVEN-FIND-BEHAVIOR>MINIMUM-MINOR-VERSION</VERSION-DRIVEN-FIND-BEHAVIOR>'
                     f'</REQUIRED-SOMEIP-SERVICE-INSTANCE>\n')
    open(os.path.join(out, "hev_someip_instances.arxml"), "w").write(HDR + _pkg("SomeipInstancesHEV", body) + FTR)

    # ---- executables (Adaptive Applications on CVC and TCU)
    apps = [("energy_mgmt", "EnergyManagementApp",
             [s[0] for s in svcs if s[2] != "CVC"], ["EnergyManagement"]),
            ("telemetry", "TelemetryApp", [s[0] for s in svcs if s[2] != "CVC"], [])]
    exe, swc, design = "", "", ""
    for ename, cname, req, prov in apps:
        design += (f'<PROCESS-DESIGN><SHORT-NAME>{ename}_design</SHORT-NAME>'
                   f'<EXECUTABLE-REF DEST="EXECUTABLE">/HEVApps/exe/{ename}</EXECUTABLE-REF></PROCESS-DESIGN>\n')
        exe += (f'<EXECUTABLE><SHORT-NAME>{ename}</SHORT-NAME><CATEGORY>APPLICATION_LEVEL</CATEGORY>'
                '<LOGGING-BEHAVIOR>USES-LOGGING</LOGGING-BEHAVIOR><REPORTING-BEHAVIOR>REPORTS-EXECUTION-STATE</REPORTING-BEHAVIOR>'
                f'<ROOT-SW-COMPONENT-PROTOTYPE><SHORT-NAME>{ename}_root</SHORT-NAME>'
                f'<APPLICATION-TYPE-TREF DEST="ADAPTIVE-APPLICATION-SW-COMPONENT-TYPE">/HEVApps/swc/{cname}</APPLICATION-TYPE-TREF>'
                '</ROOT-SW-COMPONENT-PROTOTYPE></EXECUTABLE>\n')
        swc += f'<ADAPTIVE-APPLICATION-SW-COMPONENT-TYPE><SHORT-NAME>{cname}</SHORT-NAME><PORTS>'
        for r in req:
            swc += (f'<R-PORT-PROTOTYPE><SHORT-NAME>{r}_RPort</SHORT-NAME>'
                    f'<REQUIRED-INTERFACE-TREF DEST="SERVICE-INTERFACE">/HEVServiceInterfaces/{r}</REQUIRED-INTERFACE-TREF>'
                    '</R-PORT-PROTOTYPE>')
        for p in prov:
            swc += (f'<P-PORT-PROTOTYPE><SHORT-NAME>{p}_PPort</SHORT-NAME>'
                    f'<PROVIDED-INTERFACE-TREF DEST="SERVICE-INTERFACE">/HEVServiceInterfaces/{p}</PROVIDED-INTERFACE-TREF>'
                    '</P-PORT-PROTOTYPE>')
        swc += '</PORTS></ADAPTIVE-APPLICATION-SW-COMPONENT-TYPE>\n'
    txt = (HDR + f'<AR-PACKAGE><SHORT-NAME>HEVApps</SHORT-NAME><ELEMENTS>\n{design}</ELEMENTS><AR-PACKAGES>'
           + _pkg("exe", exe) + _pkg("swc", swc) + '</AR-PACKAGES></AR-PACKAGE>\n' + FTR)
    open(os.path.join(out, "hev_apps.arxml"), "w").write(txt)
    return [s[0] for s in svcs]
