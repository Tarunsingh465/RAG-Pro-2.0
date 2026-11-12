# Industrial PLC and SCADA Network Diagnostics
## 1. Modbus/TCP Communication Timeouts
Modbus/TCP operates over port 502. When communication timeouts exceed 250 ms repeatedly:
1. Check Ethernet collision rate and packet drop counters on managed industrial switches.
2. Confirm slave unit IDs match transaction query headers.
3. Verify TCP keep-alive interval is configured to 5000 ms on slave field devices.

## 2. 4-20mA Analog Current Loop Grounding Faults
Analog process loops (pressure transmitters, thermocouples) are vulnerable to common-mode electrical ground loops.
- Symptoms: Analog signal oscillates or reads above 21.0 mA (saturation) with zero process load.
- Diagnostic measure: Disconnect shield wire at transmitter end; API/ISA standards require analog shields "
to be grounded at ONE single end only (typically at the PLC I/O cabinet marshaling terminal).
- Corrective remedy: Install a galvanic loop isolator with 1500V dielectric isolation.
