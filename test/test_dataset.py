from datasets import Dataset

testDataset = [
    {
        "question": "What is Anmol's work experience?",
        "contexts": [
            "Apple (San Diego, CA) | Engineering Intern May 2024 - August 2024: Developed and optimized system software for AirPods across multiple development platforms, applying TDD principles. Identified and resolved a critical CI/CD pipeline bug that bottlenecked over 200 engineers, reducing build times by 5% and improving team efficiency. Coordinated with 5+ cross-functional teams to align on feature updates and ensure the timely delivery of software releases.",
            "Maroon (Nashville, TN) | Software Developer Intern June 2025 - September 2025: Architected and implemented a real-time notification process for an Android/iOS dating platform leveraging Python (FastAPI), Java, AWS services (SNS, Lambda, DynamoDB). Managed the development lifecycle, encompassing API engineering and automated testing, accelerating product validation by 20%. Developed a robust automation suite in Python, integrating seamlessly with the CI/CD workflow to ensure system reliability.",
            "Vanderbilt University (Nashville, TN) | Software Engineering Intern Jan 2025 - April 2025: Automated IT incident management by integrating TDX ticketing, Statuspage, and MS Teams APIs, reducing manual effort by 30%. Designed and developed a RESTful API-driven deduplication mechanism, refining the resolution process. Engineered dynamic component mapping within an Agile framework, boosting real-time incident transparency by 50%.",
            "Vanderbilt University (Nashville, TN) | Web Development Intern Feb 2024 - April 2024: Revamped Vanderbilt University's IT website, enhancing accessibility and user experience through updated design and structure. Improved the site's user interface to align with VU's latest branding, resulting in a modern, intuitive, and visually appealing design. Developed enhanced self-service features, including password updates and resets, leveraging HTML and JavaScript to improve user engagement and efficiency by 45%. Collaborated with the design team using Git and agile workflows to integrate key features into the website, ensuring cohesive functionality and maintainability.",
        ],
        "ground_truth": "Anmol has extensive work experience across multiple organizations. He worked as an Engineering Intern at Apple (San Diego, CA) from May 2024 to August 2024, where he developed and optimized system software for AirPods and resolved critical CI/CD pipeline issues. He also has an upcoming Software Developer Intern position at Maroon (Nashville, TN) from June 2025 to September 2025, focusing on real-time notification systems for dating platforms. Additionally, he worked at Vanderbilt University in two roles: as a Software Engineering Intern from Jan 2025 to April 2025, automating IT incident management, and as a Web Development Intern from Feb 2024 to April 2024, revamping the university's IT website.",
    },
    {
        "question": "What did he do at Apple?",
        "contexts": [
            "Apple (San Diego, CA) | Engineering Intern May 2024 - August 2024: Developed and optimized system software for AirPods across multiple development platforms, applying Test-Driven Development (TDD) principles.",
            "At Apple, he identified and resolved a critical CI/CD pipeline bug that bottlenecked over 200 engineers, which reduced build times by 5% and improved team efficiency.",
            "He also coordinated with 5+ cross-functional teams to align on feature updates and ensure timely delivery of software releases during his Apple internship.",
        ],
        "ground_truth": "At Apple, Anmol worked as an Engineering Intern from May 2024 to August 2024. During his internship, he developed and optimized system software for AirPods across multiple development platforms, applying Test-Driven Development (TDD) principles. He identified and resolved a critical CI/CD pipeline bug that bottlenecked over 200 engineers, reducing build times by 5% and improving team efficiency. Additionally, he coordinated with 5+ cross-functional teams to align on feature updates and ensure timely delivery of software releases.",
    },
    {
        "question": "How long was he at Apple?",
        "contexts": [
            "Apple (San Diego, CA) | Engineering Intern May 2024 - August 2024",
            "Anmol worked at Apple as an Engineering Intern for a duration of four months, from May 2024 to August 2024.",
        ],
        "ground_truth": "Anmol was at Apple for four months, from May 2024 to August 2024.",
    },
    {
        "question": "What technologies and tools did Anmol work with?",
        "contexts": [
            "At Apple, Anmol applied Test-Driven Development (TDD) principles and worked on CI/CD pipeline optimization.",
            "At Maroon, he used Python (FastAPI), Java, and AWS services including SNS, Lambda, and DynamoDB for developing real-time notification systems.",
            "At Vanderbilt University as Software Engineering Intern, he worked with TDX ticketing, Statuspage, and MS Teams APIs, and developed RESTful APIs.",
            "As Web Development Intern at Vanderbilt, he used HTML, JavaScript, and Git with agile workflows for website development.",
        ],
        "ground_truth": "Anmol has worked with a diverse range of technologies including Python (FastAPI), Java, AWS services (SNS, Lambda, DynamoDB), HTML, JavaScript, Git, TDD principles, CI/CD pipelines, RESTful APIs, TDX ticketing, Statuspage, and MS Teams APIs. He has experience with agile workflows and various development platforms.",
    },
    {
        "question": "What achievements or impact did Anmol have in his roles?",
        "contexts": [
            "At Apple, he reduced build times by 5% and improved team efficiency by resolving a critical CI/CD pipeline bug that bottlenecked over 200 engineers.",
            "At Maroon, he accelerated product validation by 20% through API engineering and automated testing.",
            "At Vanderbilt as Software Engineering Intern, he reduced manual effort by 30% and boosted real-time incident transparency by 50%.",
            "As Web Development Intern, he improved user engagement and efficiency by 45% through enhanced self-service features.",
        ],
        "ground_truth": "Anmol achieved significant measurable impacts across his roles: At Apple, he reduced build times by 5% and improved efficiency for 200+ engineers. At Maroon, he accelerated product validation by 20%. At Vanderbilt University, he reduced manual effort by 30% and increased incident transparency by 50% in his Software Engineering role, and improved user engagement by 45% in his Web Development role.",
    },
]

# Creating dataset from the test dataset
dataset = Dataset.from_list(testDataset)
