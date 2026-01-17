"""
Test Dataset for RAG Evaluation

This module contains ground truth test cases for evaluating
the medical chatbot's RAG performance.
"""

# Medical Q&A Test Cases
MEDICAL_TEST_CASES = [
    {
        "query": "What are the symptoms of diabetes?",
        "relevant_docs": ["diabetes_overview.pdf", "diabetes_symptoms.pdf"],
        "reference_answer": "Common symptoms of diabetes include increased thirst, frequent urination, extreme hunger, unexplained weight loss, fatigue, blurred vision, slow-healing sores, and frequent infections.",
        "category": "general",
        "role": "patient"
    },
    {
        "query": "What is the recommended HbA1c target for diabetic patients?",
        "relevant_docs": ["diabetes_management.pdf", "clinical_guidelines.pdf"],
        "reference_answer": "The American Diabetes Association recommends an HbA1c target of less than 7% for most adults with diabetes. However, individual targets may vary based on patient factors.",
        "category": "clinical",
        "role": "doctor"
    },
    {
        "query": "How should I monitor blood pressure in hypertensive patients?",
        "relevant_docs": ["hypertension_guidelines.pdf", "nursing_protocols.pdf"],
        "reference_answer": "Blood pressure should be monitored regularly, typically at each clinical visit. For hypertensive patients, home monitoring is also recommended with readings taken twice daily. Normal BP is below 120/80 mmHg.",
        "category": "clinical",
        "role": "nurse"
    },
    {
        "query": "What are the contraindications for MRI scans?",
        "relevant_docs": ["mri_safety.pdf", "imaging_guidelines.pdf"],
        "reference_answer": "Absolute contraindications include cardiac pacemakers, cochlear implants, certain metallic implants, and intraocular metallic foreign bodies. Pregnancy in the first trimester is a relative contraindication.",
        "category": "clinical",
        "role": "doctor"
    },
    {
        "query": "What is a normal resting heart rate?",
        "relevant_docs": ["vital_signs.pdf", "health_basics.pdf"],
        "reference_answer": "A normal resting heart rate for adults ranges from 60 to 100 beats per minute. Athletes and highly fit individuals may have lower resting heart rates.",
        "category": "general",
        "role": "patient"
    },
    {
        "query": "What are the first-line medications for Type 2 diabetes?",
        "relevant_docs": ["diabetes_medication.pdf", "treatment_guidelines.pdf"],
        "reference_answer": "Metformin is the first-line medication for Type 2 diabetes unless contraindicated. It helps control blood sugar by decreasing glucose production in the liver and improving insulin sensitivity.",
        "category": "clinical",
        "role": "doctor"
    },
    {
        "query": "How do I prepare for a colonoscopy?",
        "relevant_docs": ["colonoscopy_prep.pdf", "patient_instructions.pdf"],
        "reference_answer": "Preparation typically involves a clear liquid diet for 24 hours before the procedure, taking prescribed laxatives to cleanse the bowel, and fasting for 8 hours before the exam. Follow your doctor's specific instructions.",
        "category": "general",
        "role": "patient"
    },
    {
        "query": "What are the warning signs of a stroke?",
        "relevant_docs": ["stroke_recognition.pdf", "emergency_guidelines.pdf"],
        "reference_answer": "Remember FAST: Face drooping, Arm weakness, Speech difficulty, Time to call emergency services. Other signs include sudden confusion, trouble seeing, severe headache, and loss of balance or coordination.",
        "category": "general",
        "role": "patient"
    },
    {
        "query": "What is the proper technique for administering insulin injections?",
        "relevant_docs": ["insulin_administration.pdf", "nursing_procedures.pdf"],
        "reference_answer": "Clean the injection site with alcohol, pinch skin if needed, insert needle at 90-degree angle (45 degrees for thin patients), inject slowly, hold for 5-10 seconds, then withdraw. Rotate injection sites to prevent lipodystrophy.",
        "category": "clinical",
        "role": "nurse"
    },
    {
        "query": "What vaccinations are recommended for adults over 65?",
        "relevant_docs": ["vaccination_schedule.pdf", "geriatric_care.pdf"],
        "reference_answer": "Recommended vaccinations include annual flu vaccine, pneumococcal vaccine (PPSV23 and PCV13), shingles vaccine (Shingrix), Tdap booster every 10 years, and COVID-19 vaccine with boosters as recommended.",
        "category": "general",
        "role": "patient"
    }
]


def get_test_cases_by_role(role: str):
    """Filter test cases by user role"""
    return [tc for tc in MEDICAL_TEST_CASES if tc["role"] == role or tc["category"] == "general"]


def get_test_cases_by_category(category: str):
    """Filter test cases by category"""
    return [tc for tc in MEDICAL_TEST_CASES if tc["category"] == category]
