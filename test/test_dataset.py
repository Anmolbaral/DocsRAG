from datasets import Dataset
testDataset = [
    {
        "question": "What is Anmol's work experience?",
        "contexts": [
            "Anmol worked as an intern at Karkhana, where he was selected from a competitive pool of 65 applicants. During his time at Karkhana, he demonstrated a strong curiosity in robotics and took on the role of project lead in various robotics events.",
            "At Apple, Anmol worked as an Engineering Intern from May 2024 to August 2024. During his internship, he developed and optimized system software for AirPods across multiple development platforms, applying Test-Driven Development (TDD) principles."
        ],
        "answer": "Anmol's work experience includes being an intern at Karkhana, where he was selected from a competitive pool of 65 applicants and took on project lead roles in robotics events. He also worked as an Engineering Intern at Apple from May 2024 to August 2024, developing and optimizing system software for AirPods.",
        "ground_truth": "Anmol's work experience includes being an intern at Karkhana, where he was selected from a competitive pool of 65 applicants and took on project lead roles in robotics events. He also worked as an Engineering Intern at Apple from May 2024 to August 2024, developing and optimizing system software for AirPods."
    },
    {
        "question": "What did he do at Apple?",
        "contexts": [
            "At Apple, Anmol worked as an Engineering Intern from May 2024 to August 2024. During his internship, he developed and optimized system software for AirPods across multiple development platforms, applying Test-Driven Development (TDD) principles.",
            "Additionally, he identified and resolved a critical CI/CD pipeline bug that bottlenecked over 200 engineers, which reduced build times by 5% and improved team efficiency. He also coordinated with over five cross-functional teams to align on feature updates and ensure timely delivery of software releases."
        ],
        "answer": "At Apple, Anmol worked as an Engineering Intern from May 2024 to August 2024. During his internship, he developed and optimized system software for AirPods across multiple development platforms, applying Test-Driven Development (TDD) principles. Additionally, he identified and resolved a critical CI/CD pipeline bug that bottlenecked over 200 engineers, which reduced build times by 5% and improved team efficiency.",
        "ground_truth": "At Apple, Anmol worked as an Engineering Intern from May 2024 to August 2024. During his internship, he developed and optimized system software for AirPods across multiple development platforms, applying Test-Driven Development (TDD) principles. Additionally, he identified and resolved a critical CI/CD pipeline bug that bottlenecked over 200 engineers, which reduced build times by 5% and improved team efficiency."
    },
    {
        "question": "How long was he at Apple?",
        "contexts": [
            "At Apple, Anmol worked as an Engineering Intern from May 2024 to August 2024.",
            "Anmol was at Apple for a duration of three months, from May 2024 to August 2024."
        ],
        "answer": "Anmol was at Apple for a duration of three months, from May 2024 to August 2024.",
        "ground_truth": "Anmol was at Apple for a duration of three months, from May 2024 to August 2024."
    }
]


"""----Creating dataset from the test dataset---"""
dataset = Dataset.from_list(testDataset)