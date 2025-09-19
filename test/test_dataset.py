from modules.rag import RAGPipeline

from datasets import Dataset
testDataset = [
    {
        "question": "What is Anmol's work experience?",
        "contexts": [
            "Anmol worked at Apple as an Engineering Intern, focusing on iOS development and machine learning projects. He was there for 3 months during the summer.",
            "During his time at Apple, he worked on user experience improvements and backend optimization."
        ],
        "answer": "Anmol worked at Apple as an Engineering Intern, focusing on airpod's features development for 3 months during the summer.",
        "ground_truth": "Anmol worked at Apple as an Engineering Intern, focusing on airpod's features development for 3 months during the summer."
    },
    {
        "question": "What did he do at Apple?",
        "contexts": [
            "At Apple, he worked on iOS development and machine learning projects, focusing on user experience improvements and backend optimization.",
            "He contributed to several features and CI/CD pipelines."
        ],
        "answer": "At Apple, he worked on iOS development and machine learning projects, focusing on user experience improvements and backend optimization.",
        "ground_truth": "At Apple, he worked on iOS development and machine learning projects, focusing on user experience improvements and backend optimization."
    },
    {
        "question": "How long was he at Apple?",
        "contexts": [
            "Anmol worked at Apple as an Engineering Intern for 3 months during the summer.",
            "His internship lasted from May to August, totaling 3 months."
        ],
        "answer": "He was at Apple for 3 months during the summer.",
        "ground_truth": "He was at Apple for 3 months during the summer."
    }
]


"""----Creating dataset from the test dataset---"""
dataset = Dataset.from_list(testDataset)