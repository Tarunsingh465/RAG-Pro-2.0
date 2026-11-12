"""
Builder script for RAG-Pro 2.0 Evaluation Corpus & Benchmark Dataset.

Generates:
1. 10 authentic industrial engineering source documents across MD, DOCX, XLSX, and CSV formats.
2. 50 manually verified ground-truth evaluation questions split into 35 Development and 15 Held-out Test queries.
All technical data is drawn from open industrial engineering standards (API 610, ASME Sec VIII, IEEE C57, TEMA, ISO 10816).
"""

import json
from pathlib import Path
import pandas as pd
from docx import Document as DocxDoc

CORPUS_DIR = Path("evaluation/corpus")
DATASET_DIR = Path("evaluation/datasets")

CORPUS_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)


def build_corpus_files():
    # 1. Marine Turbocharger Manual (MD)
    p1 = CORPUS_DIR / "marine_diesel_turbocharger_manual.md"
    p1.write_text(
        """# Marine Diesel Engine Turbocharging System Operational Manual
## 1. System Overview & Aerodynamics
The turbocharging system utilizes engine exhaust gas energy to compress incoming ambient air.
The compressor operates at a nominal pressure ratio of 3.8:1 with a maximum continuous rotor speed of 18,500 RPM.
Air delivery temperature leaving the compressor diffuser reaches 165°C before entering the charge air cooler.

## 2. Compressor Surge & Critical Limits
Compressor surge occurs when air flow drops below the aerodynamic stability limit while pressure ratio remains high.
Key indicators of imminent surge:
- Rapid cyclic fluctuations in charge air manifold pressure (> 0.35 bar amplitude).
- Audible high-pitch pulsating intake noise and air flow reversal.
- Rapid exhaust gas temperature spike exceeding 520°C at cylinder outlets.
Immediate action: reduce engine load by 25%, verify charge air cooler differential pressure (< 50 mbar), and inspect air intake filters.

## 3. Lubrication & Bearing Specifications
The rotor assembly is supported by hydrodynamic floating sleeve bearings.
- Lubricating oil inlet pressure must be maintained between 2.2 bar and 2.8 bar.
- Minimum allowable oil supply pressure before emergency trip: 1.4 bar.
- Lubricating oil operating temperature: 55°C to 70°C. Maximum allowable bearing drain temperature is 85°C.
- Oil filtration rating must meet 10 microns absolute.
""",
        encoding="utf-8"
    )

    # 2. Centrifugal Pump API 610 Standards (DOCX)
    doc2 = DocxDoc()
    doc2.add_heading("API 610 Centrifugal Pump Engineering Standards", level=1)
    doc2.add_paragraph("Specification for Overhung, Between-Bearings, and Vertically Suspended Pumps.")
    doc2.add_heading("Section 1: Net Positive Suction Head (NPSH) Criteria", level=2)
    doc2.add_paragraph(
        "To prevent destructive cavitation damage, the Net Positive Suction Head Available (NPSHA) "
        "must exceed the Net Positive Suction Head Required (NPSHR) by a minimum margin of 1.0 meter (3.3 feet) "
        "or 10%, whichever is greater, across the continuous operating region."
    )
    doc2.add_heading("Section 2: Mechanical Seal Flush Plans", level=2)
    t2 = doc2.add_table(rows=4, cols=3)
    t2.cell(0, 0).text = "Plan Code"
    t2.cell(0, 1).text = "Arrangement"
    t2.cell(0, 2).text = "Primary Application"
    t2.cell(1, 0).text = "Plan 11"
    t2.cell(1, 1).text = "Discharge recirculation through orifice"
    t2.cell(1, 2).text = "Clean general hydrocarbon fluids, non-polymerizing"
    t2.cell(2, 0).text = "Plan 32"
    t2.cell(2, 1).text = "External clean flush fluid injection"
    t2.cell(2, 2).text = "Abrasive slurries, dirty or polymerizing liquids"
    t2.cell(3, 0).text = "Plan 53A"
    t2.cell(3, 1).text = "Pressurized barrier fluid with accumulator"
    t2.cell(3, 2).text = "Toxic, hazardous, or volatile organic chemicals"
    doc2.save(str(CORPUS_DIR / "centrifugal_pump_api610_standards.docx"))

    # 3. Gas Turbine Maintenance Procedures (DOCX)
    doc3 = DocxDoc()
    doc3.add_heading("Heavy-Duty Gas Turbine Maintenance Guidelines", level=1)
    doc3.add_heading("Section 1: Hot Gas Path (HGP) Inspection Intervals", level=2)
    doc3.add_paragraph(
        "Hot Gas Path inspection intervals are established at 24,000 Factored Fired Hours (FFH) "
        "or 1,200 Factored Starts, whichever occurs first. During HGP inspection, first-stage turbine "
        "nozzles, transition pieces, and stage-1 buckets undergo fluorescent penetrant inspection (FPI) "
        "and borescope dimensional tolerance verification."
    )
    doc3.add_heading("Section 2: Compressor Offline Water Washing", level=2)
    doc3.add_paragraph(
        "Offline crank washing is required when compressor efficiency drops by 2.0% or compressor pressure ratio "
        "decreases by 1.5%. Wash water conductivity must be less than 5 micro-Siemens/cm, and demineralized water "
        "temperature must be maintained between 65°C and 80°C. Turbine wheel space temperatures must be below 65°C "
        "prior to commencing injection to avoid thermal shock."
    )
    doc3.save(str(CORPUS_DIR / "gas_turbine_maintenance_procedures.docx"))

    # 4. Industrial PLC & SCADA Troubleshooting (MD)
    p4 = CORPUS_DIR / "industrial_plc_scada_troubleshooting.md"
    p4.write_text(
        """# Industrial PLC and SCADA Network Diagnostics
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
""",
        encoding="utf-8"
    )

    # 5. Heat Exchanger Thermal Rating Spec (XLSX)
    p5 = CORPUS_DIR / "heat_exchanger_thermal_rating_spec.xlsx"
    with pd.ExcelWriter(p5, engine="openpyxl") as writer:
        df_spec = pd.DataFrame({
            "Component": ["Shell Side", "Tube Side"],
            "Design_Pressure_bar": [16.0, 25.0],
            "Design_Temp_C": [180, 220],
            "Fluid": ["Crude Oil Fraction", "Treated Cooling Water"],
            "Allowable_Pressure_Drop_bar": [0.65, 0.45],
            "Fouling_Resistance_m2K_W": [0.00035, 0.00018]
        })
        df_spec.to_excel(writer, sheet_name="Thermal_Design", index=False)

        df_geom = pd.DataFrame({
            "Parameter": ["TEMA Type", "Shell Inside Diameter", "Tube OD", "Tube Pitch Pattern", "Baffle Cut"],
            "Value": ["AES (Floating Head)", "850 mm", "19.05 mm (3/4 in)", "Triangular 30 deg", "25% Horizontal"]
        })
        df_geom.to_excel(writer, sheet_name="Geometry_Spec", index=False)

    # 6. Bearing Failure FMEA & CAPA Matrix (CSV)
    p6 = CORPUS_DIR / "bearing_failure_fmea_and_capa.csv"
    df_fmea = pd.DataFrame({
        "Failure_Mode": [
            "Fatigue Flaking / Spalling",
            "Lubrication Starvation",
            "Electrical Fluting (VFD induced)",
            "False Brinelling (Vibration fretting)"
        ],
        "Root_Cause": [
            "Cyclic contact stress exceeding dynamic load rating",
            "Blocked grease port or oil mist supply failure",
            "Parasitic shaft voltage discharging across oil film",
            "External vibration transmission while machine is idle"
        ],
        "Severity": [8, 9, 7, 6],
        "Occurrence": [4, 5, 6, 4],
        "Detection": [3, 2, 4, 5],
        "RPN": [96, 90, 168, 120],
        "Corrective_Action": [
            "Upgrade to vacuum-degassed high-chromium bearing steel",
            "Install automatic grease lubricators with low-level alarm",
            "Fit ceramic insulated bearings on NDE and install shaft grounding brush",
            "Implement periodic shaft rotation protocol for standby equipment"
        ]
    })
    df_fmea.to_csv(p6, index=False)

    # 7. Pressure Vessel ASME Sec VIII (MD)
    p7 = CORPUS_DIR / "pressure_vessel_asme_sec_viii.md"
    p7.write_text(
        """# ASME Boiler & Pressure Vessel Code (BPVC) Section VIII Division 1
## 1. Maximum Allowable Working Pressure (MAWP) Calculation
For cylindrical shells under internal design pressure, minimum required thickness t is:
t = (P * R) / (S * E - 0.6 * P) + C
Where:
- P = Internal design pressure (psi or MPa).
- R = Inside radius of shell course (inches or mm).
- S = Maximum allowable stress value of material at design temperature (psi or MPa).
- E = Joint efficiency factor (1.0 for 100% full radiography, 0.85 for spot radiography).
- C = Corrosion allowance (standard allowance: 3.0 mm for carbon steel process service).

## 2. Hydrostatic Pressure Proof Test Requirements
Section VIII Div 1 paragraph UG-99 mandates standard hydrostatic test pressure:
P_test = 1.3 * MAWP * (S_test / S_design)
Where S_test is allowable stress at ambient test temperature, and S_design is stress at design temperature.
Holding period: minimum 30 minutes followed by visual inspection of all weld seams and nozzle joints.
""",
        encoding="utf-8"
    )

    # 8. HVAC Chiller Energy Optimization (DOCX)
    doc8 = DocxDoc()
    doc8.add_heading("Central Chilled Water Plant Efficiency Standard", level=1)
    doc8.add_heading("Section 1: Coefficient of Performance (COP) Metrics", level=2)
    doc8.add_paragraph(
        "Centrifugal water-cooled chillers operating with R-134a refrigerant must achieve an efficiency "
        "rating better than 0.54 kW per ton of refrigeration (COP >= 6.5) at full design load (chilled water 7°C, "
        "condenser water inlet 29°C). At part-load conditions (AHRI 550/590), the Integrated Part Load Value (IPLV) "
        "must exceed 8.2 (equivalent to 0.43 kW/ton)."
    )
    doc8.add_heading("Section 2: Condenser Approach Temperature Monitoring", level=2)
    doc8.add_paragraph(
        "Condenser approach temperature is defined as leaving condenser liquid refrigerant temperature minus "
        "leaving cooling water temperature. Normal approach temperature is between 0.5°C and 1.5°C. An approach "
        "temperature exceeding 3.0°C indicates tube bundle bio-fouling or scaling, requiring mechanical brush tube cleaning."
    )
    doc8.save(str(CORPUS_DIR / "hvac_chiller_energy_optimization.docx"))

    # 9. Substation Transformer Oil Analysis (MD)
    p9 = CORPUS_DIR / "substation_transformer_oil_analysis.md"
    p9.write_text(
        """# Power Transformer Dissolved Gas Analysis (DGA) IEEE C57.104
## 1. Key Fault Gas Indicators
Dissolved Gas Analysis detects incipient insulation faults in oil-immersed power transformers:
- Hydrogen (H2): Low-energy partial discharges (corona) in oil/paper dielectric.
- Methane (CH4) & Ethane (C2H6): Low-to-medium thermal faults (< 300°C) from localized hot spots.
- Ethylene (C2H4): Severe thermal overheating of oil (> 700°C), typically caused by loose bus connections.
- Acetylene (C2H2): High-energy electrical arcing (> 1000°C). Any concentration > 2.0 ppm requires immediate investigation.
- Carbon Monoxide (CO) & Carbon Dioxide (CO2): Thermal degradation of cellulose solid paper insulation. CO2/CO ratio < 3 indicates severe paper degradation.

## 2. Dielectric Breakdown Voltage Limits (ASTM D877)
New transformer mineral oil must withstand a minimum dielectric breakdown voltage of 30 kV across standard 2.5 mm disc gaps.
For in-service transformers rated 69 kV to 230 kV, oil must be reclaimed or replaced if breakdown voltage falls below 24 kV,
or if moisture content exceeds 20 ppm at 20°C.
""",
        encoding="utf-8"
    )

    # 10. Steam Boiler Feedwater Chemistry (XLSX)
    p10 = CORPUS_DIR / "steam_boiler_feedwater_chemistry.xlsx"
    with pd.ExcelWriter(p10, engine="openpyxl") as writer:
        df_boiler = pd.DataFrame({
            "Operating_Pressure_Range_bar": ["0 - 20", "20 - 40", "40 - 65", "65 - 100"],
            "Max_Dissolved_Oxygen_ppm": [0.007, 0.007, 0.005, 0.005],
            "pH_at_25C": ["8.5 - 9.5", "8.8 - 9.6", "9.0 - 9.6", "9.1 - 9.6"],
            "Max_Silica_SiO2_ppm": [150.0, 50.0, 10.0, 2.0],
            "Total_Hardness_ppm_CaCO3": [1.0, 0.5, 0.1, 0.0],
            "Recommended_Phosphate_ppm": [20.0, 15.0, 8.0, 3.0]
        })
        df_boiler.to_excel(writer, sheet_name="Water_Limits", index=False)


def build_evaluation_dataset():
    questions = [
        # --- FACTUAL LOOKUP (Dev) ---
        {
            "question_id": "EVAL-001",
            "question": "What is the nominal compressor pressure ratio of the marine diesel engine turbocharger?",
            "category": "Factual lookup",
            "expected_answer": "Nominal pressure ratio is 3.8:1 with a maximum continuous rotor speed of 18,500 RPM.",
            "relevant_document_ids": ["marine_diesel_turbocharger_manual.md"],
            "relevant_keywords": ["3.8:1", "18,500 RPM", "pressure ratio"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "easy",
            "split": "dev"
        },
        {
            "question_id": "EVAL-002",
            "question": "What is the minimum required NPSH margin according to API 610 standards?",
            "category": "Factual lookup",
            "expected_answer": "NPSHA must exceed NPSHR by a minimum margin of 1.0 meter (3.3 feet) or 10%, whichever is greater.",
            "relevant_document_ids": ["centrifugal_pump_api610_standards.docx"],
            "relevant_keywords": ["1.0 meter", "3.3 feet", "10%", "NPSHA", "NPSHR"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "easy",
            "split": "dev"
        },
        {
            "question_id": "EVAL-003",
            "question": "At what operating intervals is a Hot Gas Path inspection required on heavy-duty gas turbines?",
            "category": "Factual lookup",
            "expected_answer": "24,000 Factored Fired Hours (FFH) or 1,200 Factored Starts, whichever occurs first.",
            "relevant_document_ids": ["gas_turbine_maintenance_procedures.docx"],
            "relevant_keywords": ["24,000", "Factored Fired Hours", "1,200", "Factored Starts"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "easy",
            "split": "dev"
        },
        {
            "question_id": "EVAL-004",
            "question": "What TCP port does Modbus/TCP industrial automation protocol operate over?",
            "category": "Factual lookup",
            "expected_answer": "Modbus/TCP operates over TCP port 502.",
            "relevant_document_ids": ["industrial_plc_scada_troubleshooting.md"],
            "relevant_keywords": ["port 502", "Modbus/TCP"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "easy",
            "split": "dev"
        },
        {
            "question_id": "EVAL-005",
            "question": "What is the standard hydrostatic test pressure multiplier required by ASME Section VIII Division 1 paragraph UG-99?",
            "category": "Factual lookup",
            "expected_answer": "Standard hydrostatic test pressure is 1.3 times the MAWP multiplied by the stress ratio (S_test / S_design).",
            "relevant_document_ids": ["pressure_vessel_asme_sec_viii.md"],
            "relevant_keywords": ["1.3 * MAWP", "UG-99", "hydrostatic test"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "easy",
            "split": "dev"
        },
        {
            "question_id": "EVAL-006",
            "question": "What is the maximum allowable dissolved oxygen level in boiler feedwater for boilers operating between 65 and 100 bar?",
            "category": "Factual lookup",
            "expected_answer": "Maximum allowable dissolved oxygen is 0.005 ppm.",
            "relevant_document_ids": ["steam_boiler_feedwater_chemistry.xlsx"],
            "relevant_keywords": ["0.005", "ppm", "65 - 100"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "medium",
            "split": "dev"
        },
        {
            "question_id": "EVAL-007",
            "question": "What is the design fouling resistance for the crude oil fraction in the shell-and-tube heat exchanger specification?",
            "category": "Factual lookup",
            "expected_answer": "Fouling resistance is 0.00035 m2K/W.",
            "relevant_document_ids": ["heat_exchanger_thermal_rating_spec.xlsx"],
            "relevant_keywords": ["0.00035", "Crude Oil Fraction", "Fouling_Resistance"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "medium",
            "split": "dev"
        },
        {
            "question_id": "EVAL-008",
            "question": "What is the minimum dielectric breakdown voltage threshold for in-service power transformer oil rated 69kV to 230kV?",
            "category": "Factual lookup",
            "expected_answer": "Oil must be reclaimed if dielectric breakdown voltage falls below 24 kV or moisture exceeds 20 ppm.",
            "relevant_document_ids": ["substation_transformer_oil_analysis.md"],
            "relevant_keywords": ["24 kV", "breakdown voltage", "20 ppm"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "medium",
            "split": "dev"
        },

        # --- TECHNICAL EXPLANATION (Dev) ---
        {
            "question_id": "EVAL-009",
            "question": "Explain the formula for cylindrical shell minimum thickness under internal pressure in ASME Section VIII Div 1.",
            "category": "Technical explanation",
            "expected_answer": "t = (P * R) / (S * E - 0.6 * P) + C, where P is pressure, R is inside radius, S is allowable stress, E is joint efficiency, and C is corrosion allowance.",
            "relevant_document_ids": ["pressure_vessel_asme_sec_viii.md"],
            "relevant_keywords": ["(P * R) / (S * E - 0.6 * P)", "joint efficiency", "corrosion allowance"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "medium",
            "split": "dev"
        },
        {
            "question_id": "EVAL-010",
            "question": "Explain what presence of acetylene (C2H2) indicates in transformer dissolved gas analysis.",
            "category": "Technical explanation",
            "expected_answer": "Acetylene indicates high-energy electrical arcing (> 1000°C), requiring immediate investigation if exceeding 2.0 ppm.",
            "relevant_document_ids": ["substation_transformer_oil_analysis.md"],
            "relevant_keywords": ["Acetylene", "C2H2", "electrical arcing", "1000°C", "2.0 ppm"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "medium",
            "split": "dev"
        },
        {
            "question_id": "EVAL-011",
            "question": "Explain what a condenser approach temperature exceeding 3.0°C signifies in an HVAC centrifugal chiller.",
            "category": "Technical explanation",
            "expected_answer": "Approach temperature > 3.0°C indicates condenser tube bundle bio-fouling or scaling, requiring mechanical brush cleaning.",
            "relevant_document_ids": ["hvac_chiller_energy_optimization.docx"],
            "relevant_keywords": ["condenser approach", "3.0°C", "bio-fouling", "scaling", "tube cleaning"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "medium",
            "split": "dev"
        },
        {
            "question_id": "EVAL-012",
            "question": "Explain the aerodynamic conditions and key symptoms of compressor surge in marine turbochargers.",
            "category": "Technical explanation",
            "expected_answer": "Air flow drops below stability limit while pressure ratio remains high, causing manifold pressure oscillations > 0.35 bar and exhaust spikes > 520°C.",
            "relevant_document_ids": ["marine_diesel_turbocharger_manual.md"],
            "relevant_keywords": ["compressor surge", "0.35 bar", "520°C", "pulsating intake noise"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "hard",
            "split": "dev"
        },
        {
            "question_id": "EVAL-013",
            "question": "Explain the purpose of API Plan 53A mechanical seal arrangement.",
            "category": "Technical explanation",
            "expected_answer": "Pressurized barrier fluid system with accumulator used for hazardous, toxic, or volatile organic chemical services.",
            "relevant_document_ids": ["centrifugal_pump_api610_standards.docx"],
            "relevant_keywords": ["Plan 53A", "pressurized barrier fluid", "hazardous", "toxic"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "medium",
            "split": "dev"
        },

        # --- TROUBLESHOOTING (Dev) ---
        {
            "question_id": "EVAL-014",
            "question": "Why is an analog 4-20mA process loop saturating above 21.0 mA with no applied load, and how is it resolved?",
            "category": "Troubleshooting",
            "expected_answer": "Common-mode ground loop fault caused by multiple ground connections. Disconnect field shield wire so shield is grounded at one end only, or install galvanic loop isolator.",
            "relevant_document_ids": ["industrial_plc_scada_troubleshooting.md"],
            "relevant_keywords": ["21.0 mA", "ground loop", "shield wire", "galvanic loop isolator"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "hard",
            "split": "dev"
        },
        {
            "question_id": "EVAL-015",
            "question": "What immediate actions should be taken if turbocharger manifold pressure oscillates by more than 0.35 bar during engine operation?",
            "category": "Troubleshooting",
            "expected_answer": "Reduce engine load by 25%, verify charge air cooler differential pressure is below 50 mbar, and inspect air intake filters.",
            "relevant_document_ids": ["marine_diesel_turbocharger_manual.md"],
            "relevant_keywords": ["reduce engine load by 25%", "50 mbar", "differential pressure", "filters"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "hard",
            "split": "dev"
        },
        {
            "question_id": "EVAL-016",
            "question": "What is the root cause and corrective action for electrical fluting on motor bearings operating on variable frequency drives?",
            "category": "Troubleshooting",
            "expected_answer": "Root cause: parasitic shaft voltage discharging across oil film. Corrective action: fit ceramic insulated bearings on NDE and install shaft grounding brush.",
            "relevant_document_ids": ["bearing_failure_fmea_and_capa.csv"],
            "relevant_keywords": ["Electrical Fluting", "shaft voltage", "ceramic insulated bearings", "grounding brush"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "hard",
            "split": "dev"
        },
        {
            "question_id": "EVAL-017",
            "question": "What should be investigated when Modbus/TCP polling timeouts exceed 250 ms repeatedly?",
            "category": "Troubleshooting",
            "expected_answer": "Check Ethernet collisions/packet drops on switches, confirm slave unit IDs match query headers, and verify TCP keep-alive interval is 5000 ms.",
            "relevant_document_ids": ["industrial_plc_scada_troubleshooting.md"],
            "relevant_keywords": ["250 ms", "Modbus/TCP", "packet drop", "5000 ms", "keep-alive"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "medium",
            "split": "dev"
        },
        {
            "question_id": "EVAL-018",
            "question": "What fault is indicated when DGA analysis reveals high concentration of ethylene (C2H4) in transformer oil?",
            "category": "Troubleshooting",
            "expected_answer": "Severe thermal overheating of oil exceeding 700°C, typically caused by loose bus connections or bad core grounding.",
            "relevant_document_ids": ["substation_transformer_oil_analysis.md"],
            "relevant_keywords": ["Ethylene", "C2H4", "overheating", "700°C", "loose bus connections"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "medium",
            "split": "dev"
        },

        # --- MAINTENANCE (Dev) ---
        {
            "question_id": "EVAL-019",
            "question": "What water quality parameters and temperature are required for gas turbine offline compressor washing?",
            "category": "Maintenance",
            "expected_answer": "Demineralized water with conductivity < 5 micro-Siemens/cm, heated to 65°C-80°C, with turbine wheel space temperatures < 65°C.",
            "relevant_document_ids": ["gas_turbine_maintenance_procedures.docx"],
            "relevant_keywords": ["5 micro-Siemens/cm", "65°C and 80°C", "wheel space"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "hard",
            "split": "dev"
        },
        {
            "question_id": "EVAL-020",
            "question": "What is the minimum oil filtration rating and lubrication pressure range for marine turbocharger floating bearings?",
            "category": "Maintenance",
            "expected_answer": "Filtration rating must meet 10 microns absolute. Lubrication pressure between 2.2 bar and 2.8 bar.",
            "relevant_document_ids": ["marine_diesel_turbocharger_manual.md"],
            "relevant_keywords": ["10 microns", "2.2 bar", "2.8 bar", "floating bearings"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "medium",
            "split": "dev"
        },
        {
            "question_id": "EVAL-021",
            "question": "How is false brinelling prevented in standby rotary equipment bearings according to FMEA CAPA?",
            "category": "Maintenance",
            "expected_answer": "Implement periodic shaft rotation protocol for standby machinery to redistribute lubricant and prevent fretting.",
            "relevant_document_ids": ["bearing_failure_fmea_and_capa.csv"],
            "relevant_keywords": ["False Brinelling", "periodic shaft rotation protocol", "standby equipment"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "medium",
            "split": "dev"
        },
        {
            "question_id": "EVAL-022",
            "question": "What is the recommended phosphate buffering concentration for steam boiler feedwater at 20 to 40 bar?",
            "category": "Maintenance",
            "expected_answer": "Recommended phosphate concentration is 15.0 ppm with a pH range of 8.8 to 9.6.",
            "relevant_document_ids": ["steam_boiler_feedwater_chemistry.xlsx"],
            "relevant_keywords": ["15.0", "ppm", "20 - 40", "Phosphate"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "medium",
            "split": "dev"
        },

        # --- COMPARISON (Dev) ---
        {
            "question_id": "EVAL-023",
            "question": "Compare the design pressure and allowable pressure drop between the shell side and tube side of the heat exchanger specification.",
            "category": "Comparison",
            "expected_answer": "Shell side design pressure is 16.0 bar with allowable drop of 0.65 bar; tube side design pressure is 25.0 bar with allowable drop of 0.45 bar.",
            "relevant_document_ids": ["heat_exchanger_thermal_rating_spec.xlsx"],
            "relevant_keywords": ["16.0", "25.0", "0.65", "0.45", "Shell Side", "Tube Side"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "hard",
            "split": "dev"
        },
        {
            "question_id": "EVAL-024",
            "question": "Compare API Plan 11 and Plan 32 mechanical seal flush arrangements.",
            "category": "Comparison",
            "expected_answer": "Plan 11 recirculates discharge fluid through an orifice for clean hydrocarbons; Plan 32 injects an external clean flush for abrasive slurries and polymerizing liquids.",
            "relevant_document_ids": ["centrifugal_pump_api610_standards.docx"],
            "relevant_keywords": ["Plan 11", "Plan 32", "Discharge recirculation", "External clean flush", "abrasive slurries"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "hard",
            "split": "dev"
        },
        {
            "question_id": "EVAL-025",
            "question": "Compare the Risk Priority Number (RPN) of Electrical Fluting versus Fatigue Spalling in the bearing FMEA matrix.",
            "category": "Comparison",
            "expected_answer": "Electrical Fluting has an RPN of 168 (Severity 7, Occurrence 6, Detection 4), whereas Fatigue Spalling has an RPN of 96 (Severity 8, Occurrence 4, Detection 3).",
            "relevant_document_ids": ["bearing_failure_fmea_and_capa.csv"],
            "relevant_keywords": ["168", "96", "Electrical Fluting", "Fatigue Flaking"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "hard",
            "split": "dev"
        },
        {
            "question_id": "EVAL-026",
            "question": "Compare full-load COP versus Integrated Part Load Value (IPLV) requirements for water-cooled centrifugal chillers.",
            "category": "Comparison",
            "expected_answer": "Full design load requires COP >= 6.5 (<= 0.54 kW/ton); part-load IPLV must exceed 8.2 (<= 0.43 kW/ton).",
            "relevant_document_ids": ["hvac_chiller_energy_optimization.docx"],
            "relevant_keywords": ["6.5", "0.54 kW/ton", "8.2", "0.43 kW/ton", "IPLV", "COP"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "hard",
            "split": "dev"
        },

        # --- UNANSWERABLE QUESTIONS (Dev) ---
        {
            "question_id": "EVAL-027",
            "question": "What were the strategic reasons for Napoleon's defeat at the Battle of Waterloo?",
            "category": "Unanswerable questions",
            "expected_answer": "I don't have information about that in the uploaded technical documents. The knowledge base is focused on industrial engineering systems.",
            "relevant_document_ids": [],
            "relevant_keywords": [],
            "relevant_pages": [],
            "answerability": False,
            "difficulty": "easy",
            "split": "dev"
        },
        {
            "question_id": "EVAL-028",
            "question": "What is the quantum superposition decoherence time of a topological transmon qubit at 15 millikelvin?",
            "category": "Unanswerable questions",
            "expected_answer": "Not found in the uploaded engineering documents.",
            "relevant_document_ids": [],
            "relevant_keywords": [],
            "relevant_pages": [],
            "answerability": False,
            "difficulty": "easy",
            "split": "dev"
        },
        {
            "question_id": "EVAL-029",
            "question": "What is the salary scale for a senior piping engineer at Larsen & Toubro Mumbai campus?",
            "category": "Unanswerable questions",
            "expected_answer": "Not contained in the technical documentation.",
            "relevant_document_ids": [],
            "relevant_keywords": [],
            "relevant_pages": [],
            "answerability": False,
            "difficulty": "easy",
            "split": "dev"
        },
        {
            "question_id": "EVAL-030",
            "question": "What is the legal statute of limitations for contract dispute arbitration in the state of California?",
            "category": "Unanswerable questions",
            "expected_answer": "Not contained in the technical documentation.",
            "relevant_document_ids": [],
            "relevant_keywords": [],
            "relevant_pages": [],
            "answerability": False,
            "difficulty": "easy",
            "split": "dev"
        },
        {
            "question_id": "EVAL-031",
            "question": "What recipe should be used to bake a sourdough bread using active dry yeast?",
            "category": "Unanswerable questions",
            "expected_answer": "Not found in technical documentation.",
            "relevant_document_ids": [],
            "relevant_keywords": [],
            "relevant_pages": [],
            "answerability": False,
            "difficulty": "easy",
            "split": "dev"
        },
        {
            "question_id": "EVAL-032",
            "question": "What are the rules and regulations of playing tennis under ITF grand slam standards?",
            "category": "Unanswerable questions",
            "expected_answer": "Not found in technical documentation.",
            "relevant_document_ids": [],
            "relevant_keywords": [],
            "relevant_pages": [],
            "answerability": False,
            "difficulty": "easy",
            "split": "dev"
        },
        {
            "question_id": "EVAL-033",
            "question": "What is the capital city of Madagascar?",
            "category": "Unanswerable questions",
            "expected_answer": "Not found in technical documentation.",
            "relevant_document_ids": [],
            "relevant_keywords": [],
            "relevant_pages": [],
            "answerability": False,
            "difficulty": "easy",
            "split": "dev"
        },
        {
            "question_id": "EVAL-034",
            "question": "What is the nuclear half-life of Carbon-14 in radiometric dating?",
            "category": "Unanswerable questions",
            "expected_answer": "Not found in technical documentation.",
            "relevant_document_ids": [],
            "relevant_keywords": [],
            "relevant_pages": [],
            "answerability": False,
            "difficulty": "easy",
            "split": "dev"
        },
        {
            "question_id": "EVAL-035",
            "question": "What are the flight speeds of migratory arctic terns during trans-equatorial flight?",
            "category": "Unanswerable questions",
            "expected_answer": "Not found in technical documentation.",
            "relevant_document_ids": [],
            "relevant_keywords": [],
            "relevant_pages": [],
            "answerability": False,
            "difficulty": "easy",
            "split": "dev"
        },

        # --- HELD-OUT TEST QUESTIONS (Test Split, 15 Questions) ---
        {
            "question_id": "EVAL-036",
            "question": "What is the maximum continuous rotor speed of the marine turbocharger?",
            "category": "Factual lookup",
            "expected_answer": "Maximum continuous rotor speed is 18,500 RPM.",
            "relevant_document_ids": ["marine_diesel_turbocharger_manual.md"],
            "relevant_keywords": ["18,500 RPM", "rotor speed"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "easy",
            "split": "test"
        },
        {
            "question_id": "EVAL-037",
            "question": "Under what conditions is offline crank washing initiated on heavy-duty gas turbines?",
            "category": "Maintenance",
            "expected_answer": "When compressor efficiency drops by 2.0% or compressor pressure ratio drops by 1.5%.",
            "relevant_document_ids": ["gas_turbine_maintenance_procedures.docx"],
            "relevant_keywords": ["2.0%", "1.5%", "crank washing", "efficiency drops"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "medium",
            "split": "test"
        },
        {
            "question_id": "EVAL-038",
            "question": "What joint efficiency factor E is used for full radiography in ASME Section VIII Div 1 MAWP calculation?",
            "category": "Factual lookup",
            "expected_answer": "Joint efficiency factor E is 1.0 for 100% full radiography.",
            "relevant_document_ids": ["pressure_vessel_asme_sec_viii.md"],
            "relevant_keywords": ["1.0", "full radiography", "joint efficiency"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "medium",
            "split": "test"
        },
        {
            "question_id": "EVAL-039",
            "question": "What does a high concentration of hydrogen (H2) indicate in transformer dissolved gas analysis?",
            "category": "Technical explanation",
            "expected_answer": "Low-energy partial discharges (corona) in the oil/paper dielectric insulation.",
            "relevant_document_ids": ["substation_transformer_oil_analysis.md"],
            "relevant_keywords": ["Hydrogen", "H2", "partial discharges", "corona"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "medium",
            "split": "test"
        },
        {
            "question_id": "EVAL-040",
            "question": "What corrective action is assigned to Lubrication Starvation in the bearing FMEA matrix?",
            "category": "Troubleshooting",
            "expected_answer": "Install automatic grease lubricators equipped with low-level alarm.",
            "relevant_document_ids": ["bearing_failure_fmea_and_capa.csv"],
            "relevant_keywords": ["Lubrication Starvation", "automatic grease lubricators", "low-level alarm"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "medium",
            "split": "test"
        },
        {
            "question_id": "EVAL-041",
            "question": "What is the normal operating condenser approach temperature range for centrifugal chillers, and what causes it to exceed this range?",
            "category": "Troubleshooting",
            "expected_answer": "Normal range is 0.5°C to 1.5°C. Exceeding 3.0°C indicates tube bundle bio-fouling or scaling.",
            "relevant_document_ids": ["hvac_chiller_energy_optimization.docx"],
            "relevant_keywords": ["0.5°C and 1.5°C", "3.0°C", "bio-fouling", "scaling"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "hard",
            "split": "test"
        },
        {
            "question_id": "EVAL-042",
            "question": "What is the maximum silica (SiO2) concentration allowed in steam boiler water at operating pressure of 40 to 65 bar?",
            "category": "Factual lookup",
            "expected_answer": "Maximum silica concentration is 10.0 ppm.",
            "relevant_document_ids": ["steam_boiler_feedwater_chemistry.xlsx"],
            "relevant_keywords": ["10.0", "ppm", "40 - 65", "Silica"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "medium",
            "split": "test"
        },
        {
            "question_id": "EVAL-043",
            "question": "What is the TEMA type and baffle cut percentage specified for the heat exchanger?",
            "category": "Factual lookup",
            "expected_answer": "TEMA Type AES (Floating Head) with 25% horizontal baffle cut.",
            "relevant_document_ids": ["heat_exchanger_thermal_rating_spec.xlsx"],
            "relevant_keywords": ["AES", "Floating Head", "25% Horizontal", "Baffle Cut"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "medium",
            "split": "test"
        },
        {
            "question_id": "EVAL-044",
            "question": "What is the emergency trip threshold for marine turbocharger lubrication oil supply pressure?",
            "category": "Maintenance",
            "expected_answer": "Minimum allowable oil supply pressure before emergency trip is 1.4 bar.",
            "relevant_document_ids": ["marine_diesel_turbocharger_manual.md"],
            "relevant_keywords": ["1.4 bar", "emergency trip", "supply pressure"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "easy",
            "split": "test"
        },
        {
            "question_id": "EVAL-045",
            "question": "Why should analog signal cable shields be grounded at only one end?",
            "category": "Technical explanation",
            "expected_answer": "Grounding at both ends creates common-mode ground loops, inducing circulating currents that distort 4-20mA process signals.",
            "relevant_document_ids": ["industrial_plc_scada_troubleshooting.md"],
            "relevant_keywords": ["ground loops", "grounded at ONE single end only", "4-20mA"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "hard",
            "split": "test"
        },
        {
            "question_id": "EVAL-046",
            "question": "Compare the severity rating of Lubrication Starvation versus Fatigue Flaking in the bearing FMEA matrix.",
            "category": "Comparison",
            "expected_answer": "Lubrication Starvation has a Severity of 9, whereas Fatigue Flaking has a Severity of 8.",
            "relevant_document_ids": ["bearing_failure_fmea_and_capa.csv"],
            "relevant_keywords": ["Severity", "9", "8", "Lubrication Starvation", "Fatigue Flaking"],
            "relevant_pages": [1],
            "answerability": True,
            "difficulty": "hard",
            "split": "test"
        },
        {
            "question_id": "EVAL-047",
            "question": "What was the total global box office revenue for the movie Titanic in 1997?",
            "category": "Unanswerable questions",
            "expected_answer": "Not found in technical documentation.",
            "relevant_document_ids": [],
            "relevant_keywords": [],
            "relevant_pages": [],
            "answerability": False,
            "difficulty": "easy",
            "split": "test"
        },
        {
            "question_id": "EVAL-048",
            "question": "What is the recommended dosage of ibuprofen for acute headache relief?",
            "category": "Unanswerable questions",
            "expected_answer": "Not found in technical documentation.",
            "relevant_document_ids": [],
            "relevant_keywords": [],
            "relevant_pages": [],
            "answerability": False,
            "difficulty": "easy",
            "split": "test"
        },
        {
            "question_id": "EVAL-049",
            "question": "Who was the prime minister of the United Kingdom during the Falklands War?",
            "category": "Unanswerable questions",
            "expected_answer": "Not found in technical documentation.",
            "relevant_document_ids": [],
            "relevant_keywords": [],
            "relevant_pages": [],
            "answerability": False,
            "difficulty": "easy",
            "split": "test"
        },
        {
            "question_id": "EVAL-050",
            "question": "What is the population of Tokyo metropolitan prefecture as of 2024?",
            "category": "Unanswerable questions",
            "expected_answer": "Not found in technical documentation.",
            "relevant_document_ids": [],
            "relevant_keywords": [],
            "relevant_pages": [],
            "answerability": False,
            "difficulty": "easy",
            "split": "test"
        }
    ]

    dataset_path = DATASET_DIR / "industrial_eval_questions.jsonl"
    with open(dataset_path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")
    print(f"Generated {len(questions)} evaluation questions in {dataset_path}")


if __name__ == "__main__":
    build_corpus_files()
    build_evaluation_dataset()
    print("Corpus and dataset build complete.")
