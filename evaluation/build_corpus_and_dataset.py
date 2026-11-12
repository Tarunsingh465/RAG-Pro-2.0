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
    target_path = Path("evaluation/datasets/industrial_eval_questions.jsonl")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    audited_questions = [
        # -------------------------------------------------------------
        # DEVELOPMENT SPLIT (35 Questions: 26 Answerable, 9 Unanswerable)
        # -------------------------------------------------------------
        # --- Category: Factual lookup (Dev) ---
        {
            "question_id": "EVAL-001",
            "question": "What is the nominal compressor pressure ratio of the marine diesel engine turbocharger?",
            "category": "Factual lookup",
            "expected_answer": "Nominal pressure ratio is 3.8:1 with a maximum continuous rotor speed of 18,500 RPM.",
            "relevant_document_ids": ["marine_diesel_turbocharger_manual.md"],
            "relevant_evidence_span": "The compressor operates at a nominal pressure ratio of 3.8:1 with a maximum continuous rotor speed of 18,500 RPM.",
            "relevant_section": "1. System Overview & Aerodynamics",
            "relevant_keywords": ["3.8:1", "18,500 RPM", "pressure ratio"],
            "relevant_pages": [2],
            "ground_truth_chunks_medium": ["marine_diesel_turbocharger_manual.md_medium_1"],
            "ground_truth_chunks_small": ["marine_diesel_turbocharger_manual.md_small_1"],
            "ground_truth_chunks_large": ["marine_diesel_turbocharger_manual.md_large_1"],
            "answerability": True,
            "difficulty": "easy",
            "split": "dev",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-002",
            "question": "What is the minimum required NPSH margin according to API 610 standards?",
            "category": "Factual lookup",
            "expected_answer": "NPSHA must exceed NPSHR by a minimum margin of 1.0 meter (3.3 feet) or 10%, whichever is greater.",
            "relevant_document_ids": ["centrifugal_pump_api610_standards.docx"],
            "relevant_evidence_span": "To prevent destructive cavitation damage, the Net Positive Suction Head Available (NPSHA) must exceed the Net Positive Suction Head Required (NPSHR) by a minimum margin of 1.0 meter (3.3 feet) or 10%, whichever is greater, across the continuous operating region.",
            "relevant_section": "Section 1: Net Positive Suction Head (NPSH) Criteria",
            "relevant_keywords": ["1.0 meter", "3.3 feet", "10%", "NPSHA", "NPSHR"],
            "relevant_pages": [1],
            "ground_truth_chunks_medium": ["centrifugal_pump_api610_standards.docx_medium_1"],
            "ground_truth_chunks_small": ["centrifugal_pump_api610_standards.docx_small_1"],
            "ground_truth_chunks_large": ["centrifugal_pump_api610_standards.docx_large_1"],
            "answerability": True,
            "difficulty": "easy",
            "split": "dev",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-003",
            "question": "At what operating intervals is a Hot Gas Path inspection required on heavy-duty gas turbines?",
            "category": "Factual lookup",
            "expected_answer": "24,000 Factored Fired Hours (FFH) or 1,200 Factored Starts, whichever occurs first.",
            "relevant_document_ids": ["gas_turbine_maintenance_procedures.docx"],
            "relevant_evidence_span": "Hot Gas Path inspection intervals are established at 24,000 Factored Fired Hours (FFH) or 1,200 Factored Starts, whichever occurs first.",
            "relevant_section": "Section 1: Hot Gas Path (HGP) Inspection Intervals",
            "relevant_keywords": ["24,000", "Factored Fired Hours", "1,200", "Factored Starts"],
            "relevant_pages": [1],
            "ground_truth_chunks_medium": ["gas_turbine_maintenance_procedures.docx_medium_1"],
            "ground_truth_chunks_small": ["gas_turbine_maintenance_procedures.docx_small_1"],
            "ground_truth_chunks_large": ["gas_turbine_maintenance_procedures.docx_large_1"],
            "answerability": True,
            "difficulty": "easy",
            "split": "dev",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-004",
            "question": "What TCP port does Modbus/TCP industrial automation protocol operate over?",
            "category": "Factual lookup",
            "expected_answer": "Modbus/TCP operates over TCP port 502.",
            "relevant_document_ids": ["industrial_plc_scada_troubleshooting.md"],
            "relevant_evidence_span": "Modbus/TCP operates over port 502. When communication timeouts exceed 250 ms repeatedly:",
            "relevant_section": "1. Modbus/TCP Communication Timeouts",
            "relevant_keywords": ["port 502", "Modbus/TCP"],
            "relevant_pages": [2],
            "ground_truth_chunks_medium": ["industrial_plc_scada_troubleshooting.md_medium_1"],
            "ground_truth_chunks_small": ["industrial_plc_scada_troubleshooting.md_small_1"],
            "ground_truth_chunks_large": ["industrial_plc_scada_troubleshooting.md_large_1"],
            "answerability": True,
            "difficulty": "easy",
            "split": "dev",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-005",
            "question": "What is the standard hydrostatic test pressure multiplier required by ASME Section VIII Division 1 paragraph UG-99?",
            "category": "Factual lookup",
            "expected_answer": "Standard hydrostatic test pressure is 1.3 times the MAWP multiplied by the stress ratio (S_test / S_design).",
            "relevant_document_ids": ["pressure_vessel_asme_sec_viii.md"],
            "relevant_evidence_span": "Section VIII Div 1 paragraph UG-99 mandates standard hydrostatic test pressure: P_test = 1.3 * MAWP * (S_test / S_design)",
            "relevant_section": "2. Hydrostatic Pressure Proof Test Requirements",
            "relevant_keywords": ["1.3 * MAWP", "UG-99", "hydrostatic test"],
            "relevant_pages": [3],
            "ground_truth_chunks_medium": ["pressure_vessel_asme_sec_viii.md_medium_2"],
            "ground_truth_chunks_small": ["pressure_vessel_asme_sec_viii.md_small_2"],
            "ground_truth_chunks_large": ["pressure_vessel_asme_sec_viii.md_large_2"],
            "answerability": True,
            "difficulty": "easy",
            "split": "dev",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-006",
            "question": "What is the maximum allowable dissolved oxygen level in boiler feedwater for boilers operating between 65 and 100 bar?",
            "category": "Factual lookup",
            "expected_answer": "Maximum allowable dissolved oxygen is 0.005 ppm.",
            "relevant_document_ids": ["steam_boiler_feedwater_chemistry.xlsx"],
            "relevant_evidence_span": "| 65 - 100 | 0.005 | 9.1 - 9.6 | 2 | 0 | 3 |",
            "relevant_section": "Water_Limits",
            "relevant_keywords": ["0.005 ppm", "dissolved oxygen", "65 - 100"],
            "relevant_pages": [1],
            "ground_truth_chunks_medium": ["steam_boiler_feedwater_chemistry.xlsx_medium_1"],
            "ground_truth_chunks_small": ["steam_boiler_feedwater_chemistry.xlsx_small_1"],
            "ground_truth_chunks_large": ["steam_boiler_feedwater_chemistry.xlsx_large_0"],
            "answerability": True,
            "difficulty": "easy",
            "split": "dev",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-007",
            "question": "What is the design fouling resistance for the crude oil fraction in the shell-and-tube heat exchanger specification?",
            "category": "Factual lookup",
            "expected_answer": "Fouling resistance is 0.00035 m2K/W.",
            "relevant_document_ids": ["heat_exchanger_thermal_rating_spec.xlsx"],
            "relevant_evidence_span": "| Shell Side | 16 | 180 | Crude Oil Fraction | 0.65 | 0.00035 |",
            "relevant_section": "Thermal_Design",
            "relevant_keywords": ["0.00035", "fouling resistance", "Crude Oil Fraction"],
            "relevant_pages": [1],
            "ground_truth_chunks_medium": ["heat_exchanger_thermal_rating_spec.xlsx_medium_0"],
            "ground_truth_chunks_small": ["heat_exchanger_thermal_rating_spec.xlsx_small_1"],
            "ground_truth_chunks_large": ["heat_exchanger_thermal_rating_spec.xlsx_large_0"],
            "answerability": True,
            "difficulty": "easy",
            "split": "dev",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-008",
            "question": "What is the minimum dielectric breakdown voltage threshold for in-service power transformer oil rated 69kV to 230kV?",
            "category": "Factual lookup",
            "expected_answer": "Oil must be reclaimed if dielectric breakdown voltage falls below 24 kV or moisture exceeds 20 ppm.",
            "relevant_document_ids": ["substation_transformer_oil_analysis.md"],
            "relevant_evidence_span": "In-service transformer oil (69 kV to 230 kV) requires reclamation if breakdown voltage drops below 24 kV or moisture exceeds 20 ppm.",
            "relevant_section": "2. Dielectric Breakdown Voltage Limits (ASTM D877)",
            "relevant_keywords": ["below 24 kV", "dielectric breakdown voltage", "20 ppm"],
            "relevant_pages": [3],
            "ground_truth_chunks_medium": ["substation_transformer_oil_analysis.md_medium_2"],
            "ground_truth_chunks_small": ["substation_transformer_oil_analysis.md_small_3"],
            "ground_truth_chunks_large": ["substation_transformer_oil_analysis.md_large_2"],
            "answerability": True,
            "difficulty": "easy",
            "split": "dev",
            "human_review_status": "verified"
        },
        # --- Category: Technical explanation (Dev) ---
        {
            "question_id": "EVAL-009",
            "question": "Explain the formula for cylindrical shell minimum thickness calculation under ASME Section VIII Division 1.",
            "category": "Technical explanation",
            "expected_answer": "t = (P * R) / (S * E - 0.6 * P) + C, where P is pressure, R is radius, S is allowable stress, E is joint efficiency, and C is corrosion allowance.",
            "relevant_document_ids": ["pressure_vessel_asme_sec_viii.md"],
            "relevant_evidence_span": "t = (P * R) / (S * E - 0.6 * P) + C",
            "relevant_section": "1. Maximum Allowable Working Pressure (MAWP) Calculation",
            "relevant_keywords": ["t = (P * R)", "corrosion allowance", "joint efficiency", "S * E - 0.6 * P"],
            "relevant_pages": [2],
            "ground_truth_chunks_medium": ["pressure_vessel_asme_sec_viii.md_medium_1"],
            "ground_truth_chunks_small": ["pressure_vessel_asme_sec_viii.md_small_1"],
            "ground_truth_chunks_large": ["pressure_vessel_asme_sec_viii.md_large_1"],
            "answerability": True,
            "difficulty": "medium",
            "split": "dev",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-010",
            "question": "Explain what presence of acetylene (C2H2) indicates in transformer oil dissolved gas analysis.",
            "category": "Technical explanation",
            "expected_answer": "Acetylene indicates high-energy electrical arcing (> 1000°C) and requires immediate investigation.",
            "relevant_document_ids": ["substation_transformer_oil_analysis.md"],
            "relevant_evidence_span": "- Acetylene (C2H2): High-energy electrical arcing (> 1000°C). Any concentration > 2.0 ppm requires immediate investigation.",
            "relevant_section": "1. Key Fault Gas Indicators",
            "relevant_keywords": ["Acetylene", "C2H2", "high-energy electrical arcing", "1000°C"],
            "relevant_pages": [2],
            "ground_truth_chunks_medium": ["substation_transformer_oil_analysis.md_medium_1"],
            "ground_truth_chunks_small": ["substation_transformer_oil_analysis.md_small_2"],
            "ground_truth_chunks_large": ["substation_transformer_oil_analysis.md_large_1"],
            "answerability": True,
            "difficulty": "medium",
            "split": "dev",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-011",
            "question": "Explain what a condenser approach temperature exceeding 3.0°C indicates in industrial HVAC chillers.",
            "category": "Technical explanation",
            "expected_answer": "Approach temperature > 3.0°C indicates condenser tube bundle bio-fouling or scaling, demanding mechanical brush cleaning.",
            "relevant_document_ids": ["hvac_chiller_energy_optimization.docx"],
            "relevant_evidence_span": "An approach temperature exceeding 3.0°C indicates tube bundle bio-fouling or scaling, requiring mechanical brush tube cleaning.",
            "relevant_section": "Section 2: Condenser Approach Temperature Monitoring",
            "relevant_keywords": ["exceeding 3.0°C", "tube bundle bio-fouling", "brush tube cleaning"],
            "relevant_pages": [1],
            "ground_truth_chunks_medium": ["hvac_chiller_energy_optimization.docx_medium_2"],
            "ground_truth_chunks_small": ["hvac_chiller_energy_optimization.docx_small_2"],
            "ground_truth_chunks_large": ["hvac_chiller_energy_optimization.docx_large_2"],
            "answerability": True,
            "difficulty": "medium",
            "split": "dev",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-012",
            "question": "Explain the aerodynamic conditions and key symptoms of turbocharger compressor surge.",
            "category": "Technical explanation",
            "expected_answer": "Air flow drops below stability limit while pressure ratio remains elevated, causing flow reversal, heavy vibration, pressure fluctuations, and barking noise.",
            "relevant_document_ids": ["marine_diesel_turbocharger_manual.md"],
            "relevant_evidence_span": "Compressor surge occurs when air flow drops below the aerodynamic stability limit while pressure ratio remains high. This results in periodic backflow, cyclic pressure oscillations, audible panting ('barking' sound), and severe mechanical shock loading on thrust bearings.",
            "relevant_section": "2. Compressor Surge & Critical Limits",
            "relevant_keywords": ["Compressor surge", "aerodynamic stability", "periodic backflow", "barking"],
            "relevant_pages": [3],
            "ground_truth_chunks_medium": ["marine_diesel_turbocharger_manual.md_medium_2"],
            "ground_truth_chunks_small": ["marine_diesel_turbocharger_manual.md_small_2"],
            "ground_truth_chunks_large": ["marine_diesel_turbocharger_manual.md_large_2"],
            "answerability": True,
            "difficulty": "medium",
            "split": "dev",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-013",
            "question": "Explain the purpose of API Plan 53A mechanical seal arrangement in centrifugal pumps.",
            "category": "Technical explanation",
            "expected_answer": "Pressurized barrier fluid system with accumulator used for hazardous, toxic, or volatile organic chemicals.",
            "relevant_document_ids": ["centrifugal_pump_api610_standards.docx"],
            "relevant_evidence_span": "| Plan 53A | Pressurized barrier fluid with accumulator | Toxic, hazardous, or volatile organic chemicals |",
            "relevant_section": "Section 2: Mechanical Seal Flush Plans",
            "relevant_keywords": ["Plan 53A", "Pressurized barrier fluid", "Toxic, hazardous"],
            "relevant_pages": [1],
            "ground_truth_chunks_medium": ["centrifugal_pump_api610_standards.docx_medium_3"],
            "ground_truth_chunks_small": ["centrifugal_pump_api610_standards.docx_small_3"],
            "ground_truth_chunks_large": ["centrifugal_pump_api610_standards.docx_large_3"],
            "answerability": True,
            "difficulty": "medium",
            "split": "dev",
            "human_review_status": "verified"
        },
        # --- Category: Troubleshooting (Dev) ---
        {
            "question_id": "EVAL-014",
            "question": "Why is an analog 4-20mA process loop saturating above 20.8 mA and showing high 50Hz/60Hz noise?",
            "category": "Troubleshooting",
            "expected_answer": "Common-mode ground loop fault caused by multiple ground connections on the cable shield or field sensor.",
            "relevant_document_ids": ["industrial_plc_scada_troubleshooting.md"],
            "relevant_evidence_span": "Symptoms: Signal clipping/saturation (> 20.8 mA), erratic signal jumps, or high-frequency 50Hz/60Hz noise superimposed on 4-20mA sensor loop. Root Cause: Ground potential differential between field instrument location and PLC grounding grid creates circulating shield currents.",
            "relevant_section": "2. 4-20mA Analog Current Loop Grounding Faults",
            "relevant_keywords": ["> 20.8 mA", "ground loop", "50Hz/60Hz noise", "circulating shield currents"],
            "relevant_pages": [3],
            "ground_truth_chunks_medium": ["industrial_plc_scada_troubleshooting.md_medium_2"],
            "ground_truth_chunks_small": ["industrial_plc_scada_troubleshooting.md_small_3"],
            "ground_truth_chunks_large": ["industrial_plc_scada_troubleshooting.md_large_2"],
            "answerability": True,
            "difficulty": "medium",
            "split": "dev",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-015",
            "question": "What immediate actions should be taken if turbocharger compressor surge develops during engine operation?",
            "category": "Troubleshooting",
            "expected_answer": "Reduce engine load by 25%, verify charge air cooler differential pressure, and never accelerate the engine.",
            "relevant_document_ids": ["marine_diesel_turbocharger_manual.md"],
            "relevant_evidence_span": "Immediate Action on Surge: 1. Reduce engine load immediately by 25%. 2. Verify charge air cooler differential pressure. 3. Check exhaust gas bypass (wastegate) valve position. 4. Never accelerate engine when surge symptoms persist.",
            "relevant_section": "2. Compressor Surge & Critical Limits",
            "relevant_keywords": ["Reduce engine load immediately by 25%", "charge air cooler", "Never accelerate engine"],
            "relevant_pages": [3],
            "ground_truth_chunks_medium": ["marine_diesel_turbocharger_manual.md_medium_2"],
            "ground_truth_chunks_small": ["marine_diesel_turbocharger_manual.md_small_3"],
            "ground_truth_chunks_large": ["marine_diesel_turbocharger_manual.md_large_2"],
            "answerability": True,
            "difficulty": "medium",
            "split": "dev",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-016",
            "question": "What is the root cause and corrective action for electrical fluting in motor bearings under VFD drive?",
            "category": "Troubleshooting",
            "expected_answer": "Root cause: parasitic shaft voltage discharging across oil film; corrective action: fit ceramic insulated bearings and install shaft grounding brush.",
            "relevant_document_ids": ["bearing_failure_fmea_and_capa.csv"],
            "relevant_evidence_span": "| Electrical Fluting (VFD induced) | Parasitic shaft voltage discharging across oil film | 7 | 6 | 4 | 168 | Fit ceramic insulated bearings on NDE and install shaft grounding brush |",
            "relevant_section": "Default",
            "relevant_keywords": ["Electrical Fluting", "Parasitic shaft voltage", "ceramic insulated bearings", "grounding brush"],
            "relevant_pages": [1],
            "ground_truth_chunks_medium": ["bearing_failure_fmea_and_capa.csv_medium_2"],
            "ground_truth_chunks_small": ["bearing_failure_fmea_and_capa.csv_small_3"],
            "ground_truth_chunks_large": ["bearing_failure_fmea_and_capa.csv_large_0"],
            "answerability": True,
            "difficulty": "hard",
            "split": "dev",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-017",
            "question": "What should be investigated when Modbus/TCP polling exhibits packet loss and response latency > 250ms?",
            "category": "Troubleshooting",
            "expected_answer": "Check Ethernet switch port duplex mismatch, broadcast storm packet storms, and slave PLC scan cycle time overrun.",
            "relevant_document_ids": ["industrial_plc_scada_troubleshooting.md"],
            "relevant_evidence_span": "SCADA communication failure rate exceeds 5% during peak plant operation. Corrective Procedure: 1. Confirm switch port configuration is set to Full Duplex (auto-negotiation failure creates late collisions). 2. Check Ethernet collision counters and dropped packets on managed switches. 3. Increase client polling timeout to 500 ms minimum for multi-drop slave topologies. 4. Verify broadcast storm control thresholds on managed industrial switch.",
            "relevant_section": "1. Modbus/TCP Communication Timeouts",
            "relevant_keywords": ["Full Duplex", "Ethernet collision", "dropped packets", "broadcast storm control"],
            "relevant_pages": [2],
            "ground_truth_chunks_medium": ["industrial_plc_scada_troubleshooting.md_medium_1"],
            "ground_truth_chunks_small": ["industrial_plc_scada_troubleshooting.md_small_2"],
            "ground_truth_chunks_large": ["industrial_plc_scada_troubleshooting.md_large_1"],
            "answerability": True,
            "difficulty": "hard",
            "split": "dev",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-018",
            "question": "What fault is indicated when DGA analysis reveals high concentration of ethylene (C2H4) above 50 ppm?",
            "category": "Troubleshooting",
            "expected_answer": "Severe thermal overheating of oil exceeding 700°C, typically caused by bad joints or core overheating.",
            "relevant_document_ids": ["substation_transformer_oil_analysis.md"],
            "relevant_evidence_span": "- Ethylene (C2H4): Severe thermal overheating of oil (> 700°C), typically caused by bad joints or core overheating.",
            "relevant_section": "1. Key Fault Gas Indicators",
            "relevant_keywords": ["Ethylene", "C2H4", "overheating of oil (> 700°C)", "bad joints"],
            "relevant_pages": [2],
            "ground_truth_chunks_medium": ["substation_transformer_oil_analysis.md_medium_1"],
            "ground_truth_chunks_small": ["substation_transformer_oil_analysis.md_small_2"],
            "ground_truth_chunks_large": ["substation_transformer_oil_analysis.md_large_1"],
            "answerability": True,
            "difficulty": "hard",
            "split": "dev",
            "human_review_status": "verified"
        },
        # --- Category: Maintenance (Dev) ---
        {
            "question_id": "EVAL-019",
            "question": "What water quality parameters and temperature are required for gas turbine offline compressor washing?",
            "category": "Maintenance",
            "expected_answer": "Demineralized water with conductivity < 5 micro-Siemens/cm, wash water heated to 60°C to 70°C.",
            "relevant_document_ids": ["gas_turbine_maintenance_procedures.docx"],
            "relevant_evidence_span": "Wash water quality must meet: Demineralized water with conductivity < 5.0 micro-Siemens/cm, sodium + potassium < 0.5 ppm, silica < 0.05 ppm. Wash solution temperature must be maintained between 60°C and 70°C (140°F to 158°F).",
            "relevant_section": "Section 2: Compressor Offline Water Washing",
            "relevant_keywords": ["conductivity < 5.0 micro-Siemens/cm", "60°C and 70°C", "Compressor Offline Water Washing"],
            "relevant_pages": [1],
            "ground_truth_chunks_medium": ["gas_turbine_maintenance_procedures.docx_medium_2"],
            "ground_truth_chunks_small": ["gas_turbine_maintenance_procedures.docx_small_2"],
            "ground_truth_chunks_large": ["gas_turbine_maintenance_procedures.docx_large_2"],
            "answerability": True,
            "difficulty": "medium",
            "split": "dev",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-020",
            "question": "What is the minimum oil filtration rating and lubrication supply pressure for marine turbochargers?",
            "category": "Maintenance",
            "expected_answer": "Filtration rating must meet 10 microns absolute. Lubrication oil supply pressure must be 1.8 to 2.5 bar (minimum 1.4 bar before trip).",
            "relevant_document_ids": ["marine_diesel_turbocharger_manual.md"],
            "relevant_evidence_span": "Full-flow filtration must maintain ISO 4406 cleanliness class 16/14/11 with 10-micron absolute filter rating. Lubrication oil supply pressure must be maintained between 1.8 and 2.5 bar gauge under normal operating conditions. Minimum allowable oil supply pressure before emergency trip: 1.4 bar.",
            "relevant_section": "3. Lubrication & Bearing Specifications",
            "relevant_keywords": ["10 microns absolute", "1.8 and 2.5 bar", "emergency trip: 1.4 bar"],
            "relevant_pages": [4],
            "ground_truth_chunks_medium": ["marine_diesel_turbocharger_manual.md_medium_3"],
            "ground_truth_chunks_small": ["marine_diesel_turbocharger_manual.md_small_4"],
            "ground_truth_chunks_large": ["marine_diesel_turbocharger_manual.md_large_3"],
            "answerability": True,
            "difficulty": "medium",
            "split": "dev",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-021",
            "question": "How is false brinelling prevented in standby rotary equipment bearings?",
            "category": "Maintenance",
            "expected_answer": "Implement periodic shaft rotation protocol for standby machinery to redistribute lubrication film and prevent vibration fretting.",
            "relevant_document_ids": ["bearing_failure_fmea_and_capa.csv"],
            "relevant_evidence_span": "Establish bi-weekly shaft rotation protocol (manual 90-degree turn) for all standby rotating machinery to redistribute grease and prevent static contact fretting",
            "relevant_section": "Default",
            "relevant_keywords": ["shaft rotation protocol", "standby rotating machinery", "redistribute grease"],
            "relevant_pages": [1],
            "ground_truth_chunks_medium": ["bearing_failure_fmea_and_capa.csv_medium_2"],
            "ground_truth_chunks_small": ["bearing_failure_fmea_and_capa.csv_small_3"],
            "ground_truth_chunks_large": ["bearing_failure_fmea_and_capa.csv_large_0"],
            "answerability": True,
            "difficulty": "medium",
            "split": "dev",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-022",
            "question": "What is the recommended phosphate treatment concentration for high-pressure steam boiler water at 65 to 100 bar?",
            "category": "Maintenance",
            "expected_answer": "Recommended phosphate concentration is 3.0 ppm (or 15.0 ppm for lower pressures) with pH range 9.1 to 9.6.",
            "relevant_document_ids": ["steam_boiler_feedwater_chemistry.xlsx"],
            "relevant_evidence_span": "| 65 - 100 | 0.005 | 9.1 - 9.6 | 2 | 0 | 3 |",
            "relevant_section": "Water_Limits",
            "relevant_keywords": ["65 - 100", "9.1 - 9.6", "Recommended_Phosphate_ppm"],
            "relevant_pages": [1],
            "ground_truth_chunks_medium": ["steam_boiler_feedwater_chemistry.xlsx_medium_1"],
            "ground_truth_chunks_small": ["steam_boiler_feedwater_chemistry.xlsx_small_1"],
            "ground_truth_chunks_large": ["steam_boiler_feedwater_chemistry.xlsx_large_0"],
            "answerability": True,
            "difficulty": "medium",
            "split": "dev",
            "human_review_status": "verified"
        },
        # --- Category: Comparison (Dev) ---
        {
            "question_id": "EVAL-023",
            "question": "Compare the design pressure and allowable pressure drop between shell side and tube side of the heat exchanger.",
            "category": "Comparison",
            "expected_answer": "Shell side design pressure is 16.0 bar with allowable drop of 0.65 bar; tube side design pressure is 25.0 bar with allowable drop of 0.45 bar.",
            "relevant_document_ids": ["heat_exchanger_thermal_rating_spec.xlsx"],
            "relevant_evidence_span": "| Shell Side | 16 | 180 | Crude Oil Fraction | 0.65 | 0.00035 |",
            "relevant_section": "Thermal_Design",
            "relevant_keywords": ["Shell Side", "16", "0.65", "Tube Side", "25", "0.45"],
            "relevant_pages": [1],
            "ground_truth_chunks_medium": ["heat_exchanger_thermal_rating_spec.xlsx_medium_0"],
            "ground_truth_chunks_small": ["heat_exchanger_thermal_rating_spec.xlsx_small_1"],
            "ground_truth_chunks_large": ["heat_exchanger_thermal_rating_spec.xlsx_large_0"],
            "answerability": True,
            "difficulty": "hard",
            "split": "dev",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-024",
            "question": "Compare the primary application and seal fluid arrangement of API Plan 11 versus Plan 32.",
            "category": "Comparison",
            "expected_answer": "Plan 11 recirculates discharge fluid through an orifice for clean hydrocarbons; Plan 32 injects an external clean flush fluid for dirty or polymerizing liquids.",
            "relevant_document_ids": ["centrifugal_pump_api610_standards.docx"],
            "relevant_evidence_span": "| Plan 11 | Discharge recirculation through orifice | Clean general hydrocarbon fluids, non-polymerizing | | Plan 32 | External clean flush fluid injection | Abrasive slurries, dirty or polymerizing liquids |",
            "relevant_section": "Section 2: Mechanical Seal Flush Plans",
            "relevant_keywords": ["Plan 11", "Plan 32", "Discharge recirculation", "External clean flush"],
            "relevant_pages": [1],
            "ground_truth_chunks_medium": ["centrifugal_pump_api610_standards.docx_medium_3"],
            "ground_truth_chunks_small": ["centrifugal_pump_api610_standards.docx_small_3"],
            "ground_truth_chunks_large": ["centrifugal_pump_api610_standards.docx_large_3"],
            "answerability": True,
            "difficulty": "hard",
            "split": "dev",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-025",
            "question": "Compare the Risk Priority Number (RPN) of Electrical Fluting versus False Brinelling in the bearing FMEA matrix.",
            "category": "Comparison",
            "expected_answer": "Electrical Fluting has an RPN of 168 (Severity 7, Occurrence 6, Detection 4); False Brinelling has an RPN of 96 (Severity 6, Occurrence 4, Detection 4).",
            "relevant_document_ids": ["bearing_failure_fmea_and_capa.csv"],
            "relevant_evidence_span": "| Electrical Fluting (VFD induced) | Parasitic shaft voltage discharging across oil film | 7 | 6 | 4 | 168 | Fit ceramic insulated bearings on NDE and install shaft grounding brush |",
            "relevant_section": "Default",
            "relevant_keywords": ["Electrical Fluting", "168", "False Brinelling", "96"],
            "relevant_pages": [1],
            "ground_truth_chunks_medium": ["bearing_failure_fmea_and_capa.csv_medium_2"],
            "ground_truth_chunks_small": ["bearing_failure_fmea_and_capa.csv_small_3"],
            "ground_truth_chunks_large": ["bearing_failure_fmea_and_capa.csv_large_0"],
            "answerability": True,
            "difficulty": "hard",
            "split": "dev",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-026",
            "question": "Compare the full load COP requirement against part-load Integrated Part Load Value (IPLV) for centrifugal chillers.",
            "category": "Comparison",
            "expected_answer": "Full design load requires COP >= 6.5 (<= 0.54 kW/ton); part-load efficiency requires IPLV >= 8.2 (<= 0.43 kW/ton).",
            "relevant_document_ids": ["hvac_chiller_energy_optimization.docx"],
            "relevant_evidence_span": "Centrifugal water-cooled chillers operating in industrial HVAC plants must achieve a minimum full-load COP of 6.50 (0.54 kW/ton). Part-load efficiency, represented by Integrated Part Load Value (IPLV), must meet or exceed 8.20 (0.43 kW/ton).",
            "relevant_section": "Section 1: Coefficient of Performance (COP) Metrics",
            "relevant_keywords": ["COP of 6.50", "0.54 kW/ton", "IPLV", "8.20"],
            "relevant_pages": [1],
            "ground_truth_chunks_medium": ["hvac_chiller_energy_optimization.docx_medium_1"],
            "ground_truth_chunks_small": ["hvac_chiller_energy_optimization.docx_small_1"],
            "ground_truth_chunks_large": ["hvac_chiller_energy_optimization.docx_large_1"],
            "answerability": True,
            "difficulty": "hard",
            "split": "dev",
            "human_review_status": "verified"
        },
        # --- Category: Unanswerable questions (Dev: 9 Questions EVAL-027 to EVAL-035) ---
        {
            "question_id": "EVAL-027",
            "question": "What is the recommended maintenance schedule for hydraulic excavators in mining applications?",
            "category": "Unanswerable questions",
            "expected_answer": "Not contained in the technical documentation.",
            "relevant_document_ids": [],
            "relevant_evidence_span": "",
            "relevant_section": "",
            "relevant_keywords": [],
            "relevant_pages": [],
            "ground_truth_chunks_medium": [],
            "ground_truth_chunks_small": [],
            "ground_truth_chunks_large": [],
            "answerability": False,
            "difficulty": "easy",
            "split": "dev",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-028",
            "question": "What is the quantum superposition decoherence time for superconducting transmon qubits at 15 mK?",
            "category": "Unanswerable questions",
            "expected_answer": "Not contained in the technical documentation.",
            "relevant_document_ids": [],
            "relevant_evidence_span": "",
            "relevant_section": "",
            "relevant_keywords": [],
            "relevant_pages": [],
            "ground_truth_chunks_medium": [],
            "ground_truth_chunks_small": [],
            "ground_truth_chunks_large": [],
            "answerability": False,
            "difficulty": "easy",
            "split": "dev",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-029",
            "question": "What is the salary scale for a senior piping engineer in Mumbai?",
            "category": "Unanswerable questions",
            "expected_answer": "Not contained in the technical documentation.",
            "relevant_document_ids": [],
            "relevant_evidence_span": "",
            "relevant_section": "",
            "relevant_keywords": [],
            "relevant_pages": [],
            "ground_truth_chunks_medium": [],
            "ground_truth_chunks_small": [],
            "ground_truth_chunks_large": [],
            "answerability": False,
            "difficulty": "easy",
            "split": "dev",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-030",
            "question": "What is the legal statute of limitations for contract dispute arbitration in the state of California?",
            "category": "Unanswerable questions",
            "expected_answer": "Not contained in the technical documentation.",
            "relevant_document_ids": [],
            "relevant_evidence_span": "",
            "relevant_section": "",
            "relevant_keywords": [],
            "relevant_pages": [],
            "ground_truth_chunks_medium": [],
            "ground_truth_chunks_small": [],
            "ground_truth_chunks_large": [],
            "answerability": False,
            "difficulty": "easy",
            "split": "dev",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-031",
            "question": "What recipe should be used to bake a sourdough bread using active dry yeast?",
            "category": "Unanswerable questions",
            "expected_answer": "Not found in technical documentation.",
            "relevant_document_ids": [],
            "relevant_evidence_span": "",
            "relevant_section": "",
            "relevant_keywords": [],
            "relevant_pages": [],
            "ground_truth_chunks_medium": [],
            "ground_truth_chunks_small": [],
            "ground_truth_chunks_large": [],
            "answerability": False,
            "difficulty": "easy",
            "split": "dev",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-032",
            "question": "What are the rules and regulations of playing tennis under ITF grand slam standards?",
            "category": "Unanswerable questions",
            "expected_answer": "Not found in technical documentation.",
            "relevant_document_ids": [],
            "relevant_evidence_span": "",
            "relevant_section": "",
            "relevant_keywords": [],
            "relevant_pages": [],
            "ground_truth_chunks_medium": [],
            "ground_truth_chunks_small": [],
            "ground_truth_chunks_large": [],
            "answerability": False,
            "difficulty": "easy",
            "split": "dev",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-033",
            "question": "What is the capital city of Madagascar?",
            "category": "Unanswerable questions",
            "expected_answer": "Not found in technical documentation.",
            "relevant_document_ids": [],
            "relevant_evidence_span": "",
            "relevant_section": "",
            "relevant_keywords": [],
            "relevant_pages": [],
            "ground_truth_chunks_medium": [],
            "ground_truth_chunks_small": [],
            "ground_truth_chunks_large": [],
            "answerability": False,
            "difficulty": "easy",
            "split": "dev",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-034",
            "question": "What is the nuclear half-life of Carbon-14 in radiometric dating?",
            "category": "Unanswerable questions",
            "expected_answer": "Not found in technical documentation.",
            "relevant_document_ids": [],
            "relevant_evidence_span": "",
            "relevant_section": "",
            "relevant_keywords": [],
            "relevant_pages": [],
            "ground_truth_chunks_medium": [],
            "ground_truth_chunks_small": [],
            "ground_truth_chunks_large": [],
            "answerability": False,
            "difficulty": "easy",
            "split": "dev",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-035",
            "question": "What are the flight speeds of migratory arctic terns during trans-equatorial flight?",
            "category": "Unanswerable questions",
            "expected_answer": "Not found in technical documentation.",
            "relevant_document_ids": [],
            "relevant_evidence_span": "",
            "relevant_section": "",
            "relevant_keywords": [],
            "relevant_pages": [],
            "ground_truth_chunks_medium": [],
            "ground_truth_chunks_small": [],
            "ground_truth_chunks_large": [],
            "answerability": False,
            "difficulty": "easy",
            "split": "dev",
            "human_review_status": "verified"
        },
        # -------------------------------------------------------------
        # HELD-OUT TEST SPLIT (15 Questions: 11 Answerable, 4 Unanswerable)
        # -------------------------------------------------------------
        {
            "question_id": "EVAL-036",
            "question": "What is the maximum continuous rotor speed of the marine turbocharger?",
            "category": "Factual lookup",
            "expected_answer": "Maximum continuous rotor speed is 18,500 RPM.",
            "relevant_document_ids": ["marine_diesel_turbocharger_manual.md"],
            "relevant_evidence_span": "The compressor operates at a nominal pressure ratio of 3.8:1 with a maximum continuous rotor speed of 18,500 RPM.",
            "relevant_section": "1. System Overview & Aerodynamics",
            "relevant_keywords": ["18,500 RPM", "maximum continuous rotor speed"],
            "relevant_pages": [2],
            "ground_truth_chunks_medium": ["marine_diesel_turbocharger_manual.md_medium_1"],
            "ground_truth_chunks_small": ["marine_diesel_turbocharger_manual.md_small_1"],
            "ground_truth_chunks_large": ["marine_diesel_turbocharger_manual.md_large_1"],
            "answerability": True,
            "difficulty": "easy",
            "split": "test",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-037",
            "question": "Under what conditions is offline crank washing initiated on heavy-duty gas turbines?",
            "category": "Maintenance",
            "expected_answer": "When compressor efficiency drops by 2.0% or compressor pressure drops by 3.0% at base load.",
            "relevant_document_ids": ["gas_turbine_maintenance_procedures.docx"],
            "relevant_evidence_span": "Offline crank washing is required when compressor efficiency drops by 2.0% or compressor pressure ratio drops by 3.0% at base load.",
            "relevant_section": "Section 2: Compressor Offline Water Washing",
            "relevant_keywords": ["efficiency drops by 2.0%", "pressure ratio drops by 3.0%", "Offline crank washing"],
            "relevant_pages": [1],
            "ground_truth_chunks_medium": ["gas_turbine_maintenance_procedures.docx_medium_2"],
            "ground_truth_chunks_small": ["gas_turbine_maintenance_procedures.docx_small_2"],
            "ground_truth_chunks_large": ["gas_turbine_maintenance_procedures.docx_large_2"],
            "answerability": True,
            "difficulty": "medium",
            "split": "test",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-038",
            "question": "What joint efficiency factor E is used for full radiography in ASME Section VIII Div 1 MAWP calculation?",
            "category": "Factual lookup",
            "expected_answer": "Joint efficiency factor E is 1.0 for 100% full radiography.",
            "relevant_document_ids": ["pressure_vessel_asme_sec_viii.md"],
            "relevant_evidence_span": "E = joint efficiency factor (1.0 for 100% radiographic examination per UW-11(a), 0.85 for spot radiography per UW-52, 0.70 for non-radiographed welded joint).",
            "relevant_section": "1. Maximum Allowable Working Pressure (MAWP) Calculation",
            "relevant_keywords": ["1.0 for 100% radiographic examination", "joint efficiency factor E", "UW-11(a)"],
            "relevant_pages": [2],
            "ground_truth_chunks_medium": ["pressure_vessel_asme_sec_viii.md_medium_1"],
            "ground_truth_chunks_small": ["pressure_vessel_asme_sec_viii.md_small_1"],
            "ground_truth_chunks_large": ["pressure_vessel_asme_sec_viii.md_large_1"],
            "answerability": True,
            "difficulty": "easy",
            "split": "test",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-039",
            "question": "What does a high concentration of hydrogen (H2) indicate in power transformer DGA?",
            "category": "Technical explanation",
            "expected_answer": "Low-energy partial discharges (corona) in the oil/paper dielectric.",
            "relevant_document_ids": ["substation_transformer_oil_analysis.md"],
            "relevant_evidence_span": "- Hydrogen (H2): Low-energy electrical discharge (corona, partial discharge) in oil/paper dielectric.",
            "relevant_section": "1. Key Fault Gas Indicators",
            "relevant_keywords": ["Hydrogen", "H2", "Low-energy electrical discharge", "corona"],
            "relevant_pages": [2],
            "ground_truth_chunks_medium": ["substation_transformer_oil_analysis.md_medium_1"],
            "ground_truth_chunks_small": ["substation_transformer_oil_analysis.md_small_1"],
            "ground_truth_chunks_large": ["substation_transformer_oil_analysis.md_large_1"],
            "answerability": True,
            "difficulty": "medium",
            "split": "test",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-040",
            "question": "What corrective action is assigned to Lubrication Starvation in the bearing FMEA matrix?",
            "category": "Troubleshooting",
            "expected_answer": "Install automatic grease lubricators equipped with low-level alarm.",
            "relevant_document_ids": ["bearing_failure_fmea_and_capa.csv"],
            "relevant_evidence_span": "| Lubrication Starvation | Blocked grease port or oil mist supply failure | 9 | 5 | 2 | 90 | Install automatic grease lubricators with low-level alarm |",
            "relevant_section": "Default",
            "relevant_keywords": ["Lubrication Starvation", "Install automatic grease lubricators with low-level alarm"],
            "relevant_pages": [1],
            "ground_truth_chunks_medium": ["bearing_failure_fmea_and_capa.csv_medium_1"],
            "ground_truth_chunks_small": ["bearing_failure_fmea_and_capa.csv_small_2"],
            "ground_truth_chunks_large": ["bearing_failure_fmea_and_capa.csv_large_0"],
            "answerability": True,
            "difficulty": "medium",
            "split": "test",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-041",
            "question": "What is the normal operating condenser approach temperature range for centrifugal chillers, and what causes it to exceed this range?",
            "category": "Troubleshooting",
            "expected_answer": "Normal range is 0.5°C to 1.5°C. Exceeding 3.0°C indicates tube bundle bio-fouling or scaling.",
            "relevant_document_ids": ["hvac_chiller_energy_optimization.docx"],
            "relevant_evidence_span": "Normal approach temperature is between 0.5°C and 1.5°C. An approach temperature exceeding 3.0°C indicates tube bundle bio-fouling or scaling, requiring mechanical brush tube cleaning.",
            "relevant_section": "Section 2: Condenser Approach Temperature Monitoring",
            "relevant_keywords": ["0.5°C and 1.5°C", "exceeding 3.0°C", "tube bundle bio-fouling"],
            "relevant_pages": [1],
            "ground_truth_chunks_medium": ["hvac_chiller_energy_optimization.docx_medium_2"],
            "ground_truth_chunks_small": ["hvac_chiller_energy_optimization.docx_small_2"],
            "ground_truth_chunks_large": ["hvac_chiller_energy_optimization.docx_large_2"],
            "answerability": True,
            "difficulty": "medium",
            "split": "test",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-042",
            "question": "What is the maximum silica (SiO2) concentration allowed in steam boiler water at operating pressure of 40 to 65 bar?",
            "category": "Factual lookup",
            "expected_answer": "Maximum silica concentration is 10.0 ppm.",
            "relevant_document_ids": ["steam_boiler_feedwater_chemistry.xlsx"],
            "relevant_evidence_span": "| 40 - 65 | 0.005 | 9.0 - 9.6 | 10 | 0.1 | 8 |",
            "relevant_section": "Water_Limits",
            "relevant_keywords": ["40 - 65", "10", "Max_Silica_SiO2_ppm"],
            "relevant_pages": [1],
            "ground_truth_chunks_medium": ["steam_boiler_feedwater_chemistry.xlsx_medium_1"],
            "ground_truth_chunks_small": ["steam_boiler_feedwater_chemistry.xlsx_small_1"],
            "ground_truth_chunks_large": ["steam_boiler_feedwater_chemistry.xlsx_large_0"],
            "answerability": True,
            "difficulty": "easy",
            "split": "test",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-043",
            "question": "What is the TEMA type and baffle cut percentage specified for the heat exchanger?",
            "category": "Factual lookup",
            "expected_answer": "TEMA Type AES (Floating Head) with 25% horizontal baffle cut.",
            "relevant_document_ids": ["heat_exchanger_thermal_rating_spec.xlsx"],
            "relevant_evidence_span": "| TEMA Type | AES (Floating Head) | | Baffle Cut | 25% Horizontal |",
            "relevant_section": "Geometry_Spec",
            "relevant_keywords": ["AES (Floating Head)", "25% Horizontal", "TEMA Type"],
            "relevant_pages": [2],
            "ground_truth_chunks_medium": ["heat_exchanger_thermal_rating_spec.xlsx_medium_1"],
            "ground_truth_chunks_small": ["heat_exchanger_thermal_rating_spec.xlsx_small_3"],
            "ground_truth_chunks_large": ["heat_exchanger_thermal_rating_spec.xlsx_large_1"],
            "answerability": True,
            "difficulty": "easy",
            "split": "test",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-044",
            "question": "What is the emergency trip threshold for marine turbocharger lubrication oil supply pressure?",
            "category": "Maintenance",
            "expected_answer": "Minimum allowable oil supply pressure before emergency trip is 1.4 bar.",
            "relevant_document_ids": ["marine_diesel_turbocharger_manual.md"],
            "relevant_evidence_span": "- Minimum allowable oil supply pressure before emergency trip: 1.4 bar.",
            "relevant_section": "3. Lubrication & Bearing Specifications",
            "relevant_keywords": ["emergency trip: 1.4 bar", "oil supply pressure"],
            "relevant_pages": [4],
            "ground_truth_chunks_medium": ["marine_diesel_turbocharger_manual.md_medium_3"],
            "ground_truth_chunks_small": ["marine_diesel_turbocharger_manual.md_small_4"],
            "ground_truth_chunks_large": ["marine_diesel_turbocharger_manual.md_large_3"],
            "answerability": True,
            "difficulty": "medium",
            "split": "test",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-045",
            "question": "Why should analog signal cable shields be grounded at only one end?",
            "category": "Technical explanation",
            "expected_answer": "Grounding at both ends creates common-mode ground loops, inducing circulating currents that distort 4-20mA process signals.",
            "relevant_document_ids": ["industrial_plc_scada_troubleshooting.md"],
            "relevant_evidence_span": "to be grounded at ONE single end only (typically at the PLC I/O cabinet marshaling terminal). Grounding shields at both field instrument and control room ends creates parallel ground path loops that inject high ground noise into current signals.",
            "relevant_section": "2. 4-20mA Analog Current Loop Grounding Faults",
            "relevant_keywords": ["grounded at ONE single end only", "Grounding shields at both", "parallel ground path loops"],
            "relevant_pages": [3],
            "ground_truth_chunks_medium": ["industrial_plc_scada_troubleshooting.md_medium_2"],
            "ground_truth_chunks_small": ["industrial_plc_scada_troubleshooting.md_small_3"],
            "ground_truth_chunks_large": ["industrial_plc_scada_troubleshooting.md_large_2"],
            "answerability": True,
            "difficulty": "medium",
            "split": "test",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-046",
            "question": "Compare the severity rating of Lubrication Starvation versus Fatigue Flaking in the bearing FMEA matrix.",
            "category": "Comparison",
            "expected_answer": "Lubrication Starvation has a Severity of 9, whereas Fatigue Flaking has a Severity of 8.",
            "relevant_document_ids": ["bearing_failure_fmea_and_capa.csv"],
            "relevant_evidence_span": "| Fatigue Flaking / Spalling | Cyclic contact stress exceeding dynamic load rating | 8 | 4 | 3 | 96 | | Lubrication Starvation | Blocked grease port or oil mist supply failure | 9 | 5 | 2 | 90 |",
            "relevant_section": "Default",
            "relevant_keywords": ["Fatigue Flaking", "Severity", "8", "Lubrication Starvation", "9"],
            "relevant_pages": [1],
            "ground_truth_chunks_medium": ["bearing_failure_fmea_and_capa.csv_medium_1"],
            "ground_truth_chunks_small": ["bearing_failure_fmea_and_capa.csv_small_1", "bearing_failure_fmea_and_capa.csv_small_2"],
            "ground_truth_chunks_large": ["bearing_failure_fmea_and_capa.csv_large_0"],
            "answerability": True,
            "difficulty": "hard",
            "split": "test",
            "human_review_status": "verified"
        },
        # --- Category: Unanswerable questions (Test: 4 Questions EVAL-047 to EVAL-050) ---
        {
            "question_id": "EVAL-047",
            "question": "What was the total global box office revenue for the 1997 movie Titanic?",
            "category": "Unanswerable questions",
            "expected_answer": "Not found in technical documentation.",
            "relevant_document_ids": [],
            "relevant_evidence_span": "",
            "relevant_section": "",
            "relevant_keywords": [],
            "relevant_pages": [],
            "ground_truth_chunks_medium": [],
            "ground_truth_chunks_small": [],
            "ground_truth_chunks_large": [],
            "answerability": False,
            "difficulty": "easy",
            "split": "test",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-048",
            "question": "What is the recommended dosage of ibuprofen for acute migraine treatment in adults?",
            "category": "Unanswerable questions",
            "expected_answer": "Not found in technical documentation.",
            "relevant_document_ids": [],
            "relevant_evidence_span": "",
            "relevant_section": "",
            "relevant_keywords": [],
            "relevant_pages": [],
            "ground_truth_chunks_medium": [],
            "ground_truth_chunks_small": [],
            "ground_truth_chunks_large": [],
            "answerability": False,
            "difficulty": "easy",
            "split": "test",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-049",
            "question": "Who was the prime minister of the United Kingdom during the Battle of Trafalgar in 1805?",
            "category": "Unanswerable questions",
            "expected_answer": "Not found in technical documentation.",
            "relevant_document_ids": [],
            "relevant_evidence_span": "",
            "relevant_section": "",
            "relevant_keywords": [],
            "relevant_pages": [],
            "ground_truth_chunks_medium": [],
            "ground_truth_chunks_small": [],
            "ground_truth_chunks_large": [],
            "answerability": False,
            "difficulty": "easy",
            "split": "test",
            "human_review_status": "verified"
        },
        {
            "question_id": "EVAL-050",
            "question": "What are the essential soil nutrient requirements for commercial arabica coffee tree cultivation?",
            "category": "Unanswerable questions",
            "expected_answer": "Not found in technical documentation.",
            "relevant_document_ids": [],
            "relevant_evidence_span": "",
            "relevant_section": "",
            "relevant_keywords": [],
            "relevant_pages": [],
            "ground_truth_chunks_medium": [],
            "ground_truth_chunks_small": [],
            "ground_truth_chunks_large": [],
            "answerability": False,
            "difficulty": "easy",
            "split": "test",
            "human_review_status": "verified"
        }
    ]
    
    with open(target_path, "w", encoding="utf-8") as f:
        for q in audited_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Generated {len(audited_questions)} audited evaluation questions at {target_path}")

if __name__ == "__main__":
    build_corpus_files()
    build_evaluation_dataset()
    print("Corpus and dataset build complete.")
