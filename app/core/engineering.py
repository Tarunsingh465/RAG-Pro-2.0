"""
Engineering and Industrial Intelligence Classifier.
Provides sub-millisecond discipline detection, document categorization,
and query enhancement for multi-discipline engineering problems.
"""

import re
from typing import TypedDict


class EngineeringClassification(TypedDict):
    discipline: str
    discipline_name: str
    doc_type: str
    is_engineering: bool
    confidence: float
    key_terms: list[str]


# Keyword weights for domain detection
DISCIPLINE_KEYWORDS: dict[str, dict[str, list[str]]] = {
    "cs_it": {
        "name": "Computer Science & Software Engineering",
        "terms": [
            "python", "java", "c++", "golang", "rust", "javascript", "typescript", "c#", "php",
            "algorithm", "data structure", "big-o", "time complexity", "space complexity",
            "recursion", "dynamic programming", "binary search", "graph", "tree", "hash table",
            "database", "sql", "nosql", "postgres", "mongodb", "redis", "indexing", "b-tree",
            "acid", "transaction", "deadlock", "concurrency", "multithreading", "asyncio",
            "cache", "caching", "cache stampede", "distributed system", "lock", "locking",
            "fastapi", "django", "flask", "docker", "kubernetes", "microservices", "api",
            "rest", "graphql", "grpc", "git", "ci/cd", "devops", "cloud", "aws", "azure",
            "machine learning", "deep learning", "neural network", "transformer", "llm", "rag",
            "vector database", "embedding", "nlp", "computer vision", "pytorch", "tensorflow",
            "operating system", "linux", "kernel", "memory management", "garbage collection",
            "tcp/ip", "http", "dns", "socket", "network protocol", "cybersecurity", "encryption",
            "system design", "design pattern", "refactoring", "frontend", "backend", "full stack"
        ]
    },
    "mechanical": {
        "name": "Mechanical Engineering & Thermal Systems",
        "terms": [
            "thermodynamics", "heat transfer", "fluid mechanics", "fluid dynamics", "cfd",
            "conduction", "convection", "radiation", "fourier", "nusselt", "prandtl", "reynolds",
            "carnot", "rankine", "otto cycle", "diesel cycle", "brayton", "entropy", "enthalpy",
            "stress", "strain", "shear stress", "normal stress", "bending moment", "torsion",
            "mohr's circle", "von mises", "tresca", "finite element", "fea", "ansys",
            "machine design", "bearing", "gear", "shaft", "coupling", "clutch", "spring",
            "fatigue", "s-n curve", "goodman", "factor of safety", "tolerance", "fits",
            "pump", "centrifugal pump", "cavitation", "npsh", "impeller", "compressor", "turbine",
            "internal combustion", "ic engine", "machining", "cnc", "lathe", "milling", "welding",
            "casting", "metallurgy", "heat treatment", "vibration", "natural frequency", "damping",
            "cad", "solidworks", "creo", "kinematics", "dynamics", "mechanism"
        ]
    },
    "civil": {
        "name": "Civil & Structural Engineering",
        "terms": [
            "concrete", "reinforced concrete", "rcc", "pre-stressed concrete", "cement", "mortar",
            "rebar", "tensile steel", "structural analysis", "truss", "cantilever", "simply supported",
            "deflection", "bending moment diagram", "shear force diagram", "bmd", "sfd",
            "slab", "one-way slab", "two-way slab", "column", "beam", "lintel", "footing", "retaining wall",
            "is 456", "is 800", "is 1893", "eurocode", "aci 318", "astm", "limit state", "working stress",
            "soil mechanics", "geotechnical", "bearing capacity", "terzaghi", "permeability", "compaction",
            "consolidation", "shear strength", "triaxial", "foundation", "shallow foundation", "pile foundation",
            "hydrology", "open channel", "mannings", "weir", "culvert", "dam", "spillway",
            "surveying", "theodolite", "total station", "asphalt", "pavement design", "traffic engineering",
            "seismic", "earthquake resistant", "wind load", "dead load", "live load"
        ]
    },
    "chemical": {
        "name": "Chemical & Process Engineering",
        "terms": [
            "reaction kinetics", "chemical reaction", "reaction rate", "arrhenius", "activation energy",
            "cstr", "pfr", "plug flow", "batch reactor", "catalyst", "catalytic", "stoichiometry",
            "distillation", "distillation column", "reflux ratio", "mccabe-thiele", "fenske", "trays", "packing",
            "mass transfer", "fick's law", "absorption", "stripping", "extraction", "leaching",
            "heat exchanger", "lmtd", "shell and tube", "overall heat transfer coefficient",
            "p&id", "process flow diagram", "pfd", "hazop", "process safety", "runaway reaction",
            "relief valve", "prv", "chemical equilibrium", "raoult's law", "henry's law", "vle", "lle",
            "fluidization", "drying", "crystallization", "filtration", "scale-up", "polymer", "petrochemical"
        ]
    },
    "electrical": {
        "name": "Electrical, Electronics & Control Engineering",
        "terms": [
            "voltage", "current", "resistance", "impedance", "reactance", "ohm's law", "kcl", "kvl",
            "kirchhoff", "thevenin", "norton", "ac circuit", "dc circuit", "power factor", "three phase",
            "transformer", "step up", "step down", "induction motor", "synchronous motor", "dc motor",
            "generator", "transmission line", "substation", "switchgear", "circuit breaker", "inverter",
            "semiconductor", "diode", "bjt", "mosfet", "igbt", "op-amp", "operational amplifier",
            "filter", "rectifier", "adc", "dac", "microcontroller", "arduino", "stm32", "esp32", "fpga",
            "control system", "transfer function", "bode plot", "nyquist", "root locus", "pid controller",
            "plc", "scada", "sensors", "transducers", "signal processing", "fourier transform", "dsp"
        ]
    },
    "industrial_maintenance": {
        "name": "Industrial Reliability, QA & Maintenance",
        "terms": [
            "root cause analysis", "rca", "5 whys", "fishbone", "ishikawa", "fmea", "fmeca",
            "reliability", "mtbf", "mean time between failures", "mttr", "mean time to repair",
            "availability", "overall equipment effectiveness", "oee", "preventive maintenance",
            "predictive maintenance", "condition monitoring", "vibration analysis", "thermography",
            "six sigma", "dmaic", "spc", "control charts", "cpk", "ppk", "quality control",
            "standard operating procedure", "sop", "troubleshooting", "failure analysis", "iso 9001", "iso 14001"
        ]
    }
}

DOCUMENT_TYPE_PATTERNS = {
    "standard_specification": [
        r"\b(iso|asme|astm|ieee|is\s*\d+|din|bs|en|aci|api)\s*[-:]?\s*\d+",
        r"\bstandard specification\b",
        r"\bcode of practice\b",
        r"\bcompliance\b",
        r"\bregulatory requirement\b",
    ],
    "troubleshooting_guide": [
        r"\btroubleshooting\b",
        r"\bfault finding\b",
        r"\berror code\b",
        r"\broot cause\b",
        r"\bremedy\b",
        r"\bcorrective action\b",
        r"\bpreventive maintenance\b",
        r"\bfailure analysis\b",
    ],
    "engineering_datasheet": [
        r"\bdatasheet\b",
        r"\btechnical specifications?\b",
        r"\boperating parameters\b",
        r"\brated capacity\b",
        r"\btolerance\b",
        r"\bmaterial properties\b",
        r"\bdimensions and ratings\b",
    ],
    "candidate_resume": [
        r"\bcurriculum vitae\b",
        r"\bresume\b",
        r"\beducation\b",
        r"\bwork experience\b",
        r"\bcgpa\b",
        r"\btechnical skills\b",
        r"\bprojects\b",
        r"\bcertifications\b",
    ],
    "textbook_handbook": [
        r"\bchapter\s*\d+\b",
        r"\bexercise\b",
        r"\bhandbook\b",
        r"\bprinciples of\b",
        r"\bintroduction to\b",
        r"\bderivation\b",
    ],
}


def classify_engineering_domain(text: str, filename: str = "") -> EngineeringClassification:
    """
    Classify content into an engineering discipline and document type in < 1 millisecond.
    """
    sample_text = f"{filename} {text[:8000]}".lower()

    scores: dict[str, int] = {}
    matched_terms: dict[str, list[str]] = {}

    for discipline, config in DISCIPLINE_KEYWORDS.items():
        score = 0
        matches = []
        for term in config["terms"]:
            if len(term) <= 3:
                # Use word boundaries for short terms like "rag", "sql", "rcc", "p&id"
                pattern = rf"\b{re.escape(term)}\b"
                if re.search(pattern, sample_text):
                    score += 2
                    matches.append(term)
            else:
                if term in sample_text:
                    score += 1
                    matches.append(term)
        scores[discipline] = score
        matched_terms[discipline] = matches

    # Determine highest scoring discipline
    top_discipline = max(scores, key=lambda k: scores[k])
    top_score = scores[top_discipline]

    # Threshold for engineering domain recognition
    if top_score < 3:
        discipline = "general"
        discipline_name = "General Knowledge / Documentation"
        is_engineering = False
        confidence = 0.50
        key_terms = []
    else:
        discipline = top_discipline
        discipline_name = DISCIPLINE_KEYWORDS[top_discipline]["name"]
        is_engineering = True
        confidence = min(0.99, 0.60 + (top_score * 0.04))
        key_terms = matched_terms[top_discipline][:6]

    # Detect document type
    detected_doc_type = "general_document"
    for doc_type, patterns in DOCUMENT_TYPE_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, sample_text, re.IGNORECASE):
                detected_doc_type = doc_type
                break
        if detected_doc_type != "general_document":
            break

    return {
        "discipline": discipline,
        "discipline_name": discipline_name,
        "doc_type": detected_doc_type,
        "is_engineering": is_engineering,
        "confidence": confidence,
        "key_terms": key_terms,
    }


def is_engineering_query(query: str) -> tuple[bool, str, str]:
    """
    Check if a query asks an engineering/industrial question.
    Returns (is_eng, discipline, guidance_tag).
    """
    result = classify_engineering_domain(query)
    q_lower = query.lower()

    # Look for engineering action intents
    action_intents = [
        "calculate", "derive", "formula", "equation", "design", "solve",
        "troubleshoot", "why is", "failure", "tolerance", "specification",
        "standard", "code", "deflection", "stress", "efficiency", "sizing",
        "implement", "architecture", "algorithm", "optimize", "how does"
    ]
    has_action = any(act in q_lower for act in action_intents)

    if result["is_engineering"] or (has_action and result["confidence"] > 0.4):
        return True, result["discipline"], result["discipline_name"]

    return False, "general", "General"
