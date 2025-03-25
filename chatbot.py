import json
import os
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import torch
import pickle
from fuzzywuzzy import fuzz, process
import re
import numpy as np

# Cache file paths
MODEL_CACHE_FILE = "model_cache.pkl"
EMBEDDINGS_CACHE_FILE = "embeddings_cache.pkl"


class ChatBot:
    def __init__(self):
        # Initialize the chatbot with the necessary data and models
        self.model = self.load_model()
        self.data = self.load_data()
        self.qa_pairs = []
        self.experience_qa_pairs = []
        self.achievements = []
        self.extract_qa(self.data)
        self.answer_embeddings, self.experience_embeddings = self.load_embeddings()

    def load_model(self):
        # Load or create model (with caching)
        if os.path.exists(MODEL_CACHE_FILE):
            with open(MODEL_CACHE_FILE, "rb") as f:
                model = pickle.load(f)
            print("Model loaded from cache.")
        else:
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            with open(MODEL_CACHE_FILE, "wb") as f:
                pickle.dump(model, f)
            print("Model loaded and cached.")
        return SentenceTransformer("paraphrase-MiniLM-L6-v2")

    def load_data(self):
        # Load data from JSON file
        with open("Sohail_Data.json", "r") as file:
            return json.load(file)

    def extract_qa(self, data):
        # Extract QA pairs from data
        for section, items in data.items():
            if isinstance(items, dict):  # Handle dictionaries (bio, skills)
                if section == "bio":
                    summary = items.get("summary")
                    questions = items.get("questions", [])
                    for question in questions:
                        self.qa_pairs.append((question, summary, section))  # bio data
                elif section == "skills":
                    for skill_category, skill_data in items.items():
                        tools = skill_data.get("tools", [])
                        questions = skill_data.get("questions", [])
                        answer = ", ".join(tools)
                        for question in questions:
                            self.qa_pairs.append(
                                (question, answer, "tools_" + skill_category)
                            )
            elif isinstance(items, list):
                if section == "achievements":
                    self.achievements = items  # Handle list sections (like experiences)
                for item in items:
                    question = item.get("question")
                    answer = item.get("answer")
                    these_paraphrases = item.get("paraphrases", [])
                    if question and answer:
                        if section == "experiences":  # Add to experience_qa_pairs
                            self.experience_qa_pairs.append((question, answer, section))
                            for paraphrase in these_paraphrases:
                                self.experience_qa_pairs.append(
                                    (paraphrase, answer, section)
                                )
                        else:  # Add to general qa_pairs
                            self.qa_pairs.append((question, answer, section))
                            for paraphrase in these_paraphrases:
                                self.qa_pairs.append((paraphrase, answer, section))

    def load_embeddings(self):
        # Load or calculate and cache embeddings
        if os.path.exists(EMBEDDINGS_CACHE_FILE):
            with open(EMBEDDINGS_CACHE_FILE, "rb") as f:
                cached_data = pickle.load(f)
                answer_embeddings = cached_data["answer_embeddings"]
                experience_embeddings = cached_data["experience_embeddings"]
            print("Embeddings loaded from cache.")
        else:
            answers = [qa[1] for qa in self.qa_pairs]
            answer_embeddings = self.model.encode(
                answers
            )  # Use the loaded/cached model

            experience_answers = [qa[1] for qa in self.experience_qa_pairs]
            experience_embeddings = self.model.encode(
                experience_answers
            )  # Use the loaded/cached model

            cached_data = {
                "answer_embeddings": answer_embeddings,
                "experience_embeddings": experience_embeddings,
            }
            with open(EMBEDDINGS_CACHE_FILE, "wb") as f:
                pickle.dump(cached_data, f)
            print("Embeddings calculated and cached.")

        answers = [qa[1] for qa in self.qa_pairs]
        answer_embeddings = self.model.encode(answers)

        experience_answers = [qa[1] for qa in self.experience_qa_pairs]
        experience_embeddings = self.model.encode(
            experience_answers
        )  # Calculate experience_embeddings HERE

        return answer_embeddings, experience_embeddings

    def find_best_answer(self, query, top_n=3, threshold=0.4):
        # 3. Skills Queries (Prioritized Keyword and Direct Retrieval)
        skill_categories = [
            "graphic_design",
            "video_editing",
            "3d_modeling",
            "programming",
            "ui_ux_design",
        ]

        # 1. Direct Tool Retrieval (CHECKED FIRST, OUTSIDE THE CATEGORY LOOP)
        for category in skill_categories:  # Iterate through categories to access tools
            tools = self.data["skills"][category]["tools"]
            for tool in tools:
                if tool.lower() in query.lower():
                    return f"Yes, Sohail is proficient in {tool}"

        # 4. Category-Specific Queries (After Tool Retrieval)
        for category in skill_categories:
            if category in query.lower() or any(
                keyword in query.lower() for keyword in category.split("_")
            ):
                tools = self.data["skills"][category]["tools"]

                # 2.a Check for Experience/Background in Category
                experience_keywords = [
                    "experienced",
                    "background",
                    "expertise",
                    "strong in",
                    "familiar with",
                    "knowledge of",
                    "skilled in",
                ]
                if any(keyword in query.lower() for keyword in experience_keywords):
                    return f"Yes, Sohail is experienced in {category.replace('_','')}."

                # 2.b Check for Tools with Category
                tool_keywords = ["tools", "software", "programs", "applications"]
                if any(keyword in query.lower() for keyword in tool_keywords):
                    return (
                        f"Sohail's skills in {category.replace('_','')} include: "
                        + ", ".join(tools)
                        + "."
                    )

                # 2.c Category Listing (Only if NO other more specific match is found)
                return f"Sohail is skilled in {category.replace('_','')}."  # Simplified

        # 3. What does Sohail do? (Moved DOWN, AFTER skills)
        if any(
            keyword in query.lower()
            for keyword in ["do", "work", "job", "profession", "roles", "known"]
        ):
            roles = self.data["bio"].get("roles")
            if roles:
                return "He works as a " + ", ".join(roles) + "."
            else:
                return "Information about Sohail's roles is not available."

        # Check for experience queries
        company_names = [
            exp.get("company")
            for exp in self.data["achievements"]
            if exp.get("type") == "work"
        ]
        year = None
        company = None

        # Extract year from query
        year_match = re.search(r"\b\d{4}\b", query)
        if year_match:
            year = year_match.group()

        if any(
            keyword in query.lower()
            for keyword in [
                "experience",
                "work",
                "job",
                "career",
                "employment",
                "internship",
                "position",
                "role",
                "background",
            ]
        ):
            # Use fuzzy matching to find the best match
            best_match = process.extractOne("utkarsh", company_names)

            if best_match and best_match[1] > 60:  # threshold for fuzzy matching
                company = best_match[0]

            filtered_achievements = []
            for exp in self.data["achievements"]:
                if exp.get("type") == "work" and (
                    (
                        year
                        and (year in exp.get("date", "") or year in exp.get("date", ""))
                    )
                    or (company and company.lower() in exp.get("company", "").lower())
                ):
                    filtered_achievements.append(exp)

            if filtered_achievements:
                achievement_summary = []
                for exp in filtered_achievements:
                    achievement_summary.append(
                        f"{exp.get('title', '')} at {exp.get('company', '')} ({exp.get('date', '')}): {exp.get('description', '')}"
                    )

                if year:
                    return (
                        f"Sohail's work experience in {year} includes:\n"
                        + "\n".join(achievement_summary)
                    )
                elif company:
                    return (
                        f"Sohail's work experience at {company} includes:\n"
                        + "\n".join(achievement_summary)
                    )
                else:
                    return "Sohail's work experience includes:\n" + "\n".join(
                        achievement_summary
                    )

            else:
                if year:
                    return f"No work experience found for the year {year}."
                elif company:
                    return f"No work experience found at {company}."
                else:
                    return "Sohail's work experience information is not available."

        # 4. General Skills Overview (Corrected and Moved Down)
        general_skill_keywords = [
            "skills",
            "expertise",
            "proficient",
            "abilities",
            "areas of expertise",
            "what can sohail do",
            "skillset",
            "capabilities",
            "what are his skills",
            "what skills does he have",
        ]  # More keywords
        if any(
            keyword in query.lower() for keyword in general_skill_keywords
        ) and not any(
            keyword in query.lower()
            for keyword in [
                "photoshop",
                "premiere pro",
                "after effects",
                "davinci resolve",
                "filmora",
                "3ds max",
                "blender",
                "maya",
                "substance 3d painter",
                "zbrush",
                "mudbox",
                "java",
                "html",
                "css",
                "javascript",
                "python",
                "google colab",
                "vs code",
                "adobe xd",
                "figma",
            ]
        ):  # Added exception for all the tools
            skill_categories_formatted = [
                cat.replace("_", " ") for cat in skill_categories
            ]
            return "Sohail has skills in " + ", ".join(skill_categories_formatted) + "."

        # 5. Prioritized Keyword Checks (Portfolio and Contact)
        if any(
            keyword in query.lower()
            for keyword in ["work", "projects", "portfolio", "github"]
        ):
            github_link = self.data["bio"]["social_links"].get("github")
            if github_link:
                return f"You can find more of Sohail's projects on his GitHub profile: {github_link}"
            else:
                return "Sohail's portfolio information is not available at the moment."

        # Tokenize the query
        tokens = nltk.word_tokenize(query)
        pos_tags = nltk.pos_tag(tokens)

        # Identify the intent behind the query
        education_keywords = [
            "education",
            "school",
            "college",
            "studied",
            "pursued",
            "degree",
            "graduation",
            "university",
        ]
        bio_keywords = ["who", "name", "sohail"]
        intent = None
        for token, tag in pos_tags:
            if token.lower() in education_keywords:
                intent = "education"
                break
            elif token.lower() in bio_keywords:
                intent = "bio"
                break

        # If no keywords are found, set the intent to "unknown"
        if intent is None:
            intent = "unknown"

        # Generate an answer to the query
        if intent == "education":
            # Define a list of possible answers
            answers = []
            for achievement in self.achievements:
                if (
                    "12th" in achievement["degree"].lower()
                    or "hsc" in achievement["degree"].lower()
                ):
                    answers.append(
                        f"He completed his 12th grade from {achievement['school']}."
                    )
                elif (
                    "10th" in achievement["degree"].lower()
                    or "ssc" in achievement["degree"].lower()
                ):
                    answers.append(
                        f"He completed his 10th grade from {achievement['school']}."
                    )
                else:
                    answers.append(
                        f"He is currently pursuing {achievement['degree']} from {achievement['school']}."
                    )

            # Check if the query is asking about the degree
            if "degree" in query.lower() or "pursued" in query.lower():
                for achievement in self.achievements:
                    if "Bachelor" in achievement["degree"]:
                        return f"He is currently pursuing his {achievement['degree']} from {achievement['school']}."

            # Compute the embeddings of the query and the possible answers
            query_embedding = self.model.encode([query])
            answer_embeddings = self.model.encode(answers)

            # Compute the cosine similarity between the query and the possible answers
            cosine_scores = util.cos_sim(query_embedding, answer_embeddings)

            # Get the index of the answer with the highest similarity score
            answer_index = torch.argmax(cosine_scores)

            # Return the answer with the highest similarity score
            return answers[answer_index]

        if any(
            keyword in query.lower()
            for keyword in [
                "contact",
                "reach",
                "touch",
                "connect",
                "social media",
                "links",
                "linkedin",
                "x",
                "insta",
                "facebook",
            ]
        ):
            linkedin_link = self.data["bio"]["social_links"].get("linkedin")
            x_link = self.data["bio"]["social_links"].get("x")
            insta_link = self.data["bio"]["social_links"].get("insta")
            facebook_link = self.data["bio"]["social_links"].get("facebook")

            contact_info = []
            if linkedin_link:
                contact_info.append(f"LinkedIn: {linkedin_link}")
            if x_link:
                contact_info.append(f"X (Twitter): {x_link}")
            if insta_link:
                contact_info.append(f"Instagram: {insta_link}")
            if facebook_link:
                contact_info.append(f"Facebook: {facebook_link}")

            if contact_info:
                return (
                    "You can contact him through his "
                    + " and ".join(contact_info)
                    + "."
                )
            else:
                return "Contact information for Sohail is not available at the moment."

        # Bio Handling
        best_match = None
        best_similarity = 0
        for q, a, s in self.qa_pairs:
            if s == "bio":
                similarity = fuzz.ratio(query.lower(), q.lower())
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (q, a, s)

        if best_match and best_similarity > 60:  # adjust the threshold as needed
            return best_match[1]

        # Semantic Search for Skills
        skill_candidate_indices = []
        for i, (question, answer, section) in enumerate(self.qa_pairs):
            if section.startswith("tools_"):
                similarity = fuzz.partial_ratio(query.lower(), question.lower())
                if similarity > 80:  # adjust the threshold as needed
                    skill_candidate_indices.append(i)

                if not skill_candidate_indices:
                    skill_candidate_indices = [
                        i
                        for i, (_, _, section) in enumerate(self.qa_pairs)
                        if section.startswith("tools_")
                    ]

        # Get the embeddings for the skill candidates
        skill_candidate_indices = [
            i
            for i, (_, _, section) in enumerate(self.qa_pairs)
            if section.startswith("tools_")
        ]
        if not skill_candidate_indices:
            skill_candidate_indices = [
                i
                for i, (_, _, section) in enumerate(self.qa_pairs)
                if section.startswith("experience_")
            ]

        if not skill_candidate_indices:
            return "Sorry, I couldn't find a good match for your query."

        skill_candidate_embeddings = []
        for i in skill_candidate_indices:
            if isinstance(self.answer_embeddings, dict) and i in self.answer_embeddings:
                skill_candidate_embeddings.append(self.answer_embeddings[i])
            elif isinstance(self.answer_embeddings, list) and i < len(
                self.answer_embeddings
            ):
                skill_candidate_embeddings.append(self.answer_embeddings[i])

        if not skill_candidate_embeddings:
            return "Sorry, I couldn't find a good match for your query."

        # Calculate the similarities
        query_embedding = self.model.encode([query])
        skill_similarities = cosine_similarity(
            query_embedding, skill_candidate_embeddings
        )

        # Get the top N indices
        skill_top_n_indices = np.argsort(-skill_similarities[0])[
            : min(top_n, len(skill_candidate_embeddings))
        ]

        # Get the best matches
        skill_best_matches = []
        for i in skill_top_n_indices:
            similarity = skill_similarities[0][i]
            question, answer, section = self.qa_pairs[skill_candidate_indices[i]]
            skill_best_matches.append((question, answer, section, similarity))

        # Check if the best match is the exact question
        for question, answer, section, similarity in skill_best_matches:
            if question.lower() == query.lower():
                return answer

        # Return the best match if its similarity is above the threshold
        if skill_best_matches and skill_best_matches[0][3] >= threshold:
            return skill_best_matches[0][1]

        # 10. General Semantic Search (Fallback)
        experience_keywords = [
            "experience",
            "work",
            "job",
            "career",
            "employment",
            "internship",
            "position",
            "role",
            "background",
        ]
        if not any(
            keyword in query.lower() for keyword in experience_keywords
        ):  # Only if no experience keyword is present
            candidate_indices = []
            for i, (question, answer, section) in enumerate(self.qa_pairs):
                keywords_in_query = any(
                    keyword in query.lower() for keyword in question.lower().split()
                )
                if keywords_in_query:
                    candidate_indices.append(i)

            if not candidate_indices:
                candidate_indices = list(range(len(self.qa_pairs)))

            if not candidate_indices:
                return "Sorry, I couldn't find a good match for your query."

            candidate_embeddings = self.answer_embeddings[candidate_indices]
            query_embedding = self.model.encode([query])
            cosine_similarities = cosine_similarity(
                query_embedding, candidate_embeddings
            )[0]

            top_n_indices = cosine_similarities.argsort()[::-1][:top_n]

            best_matches = []
            for i in top_n_indices:
                similarity = cosine_similarities[i]
                question, answer, section = self.qa_pairs[candidate_indices[i]]
                best_matches.append((question, answer, section, similarity))

            for question, answer, section, similarity in best_matches:
                if question.lower() == query.lower():
                    return answer

            if best_matches and best_matches[0][3] >= threshold:
                return best_matches[0][1]

        return "Sorry, I couldn't find a good match for your query."


# queries = [
#     "Tell me about Sohail.",
#     "Who is Sohail Shriyan?",
#     "What can you tell me about Sohail's background?",
#     "What are Sohail's roles?",
#     "What kind of work does Sohail do?",
#     "What are Sohail's key skills and interests?",
#     "How would you describe Sohail as an individual?",
#     "What are Sohail's aspirations?",
#     "What motivates Sohail?",
#     "What are Sohail's strengths and weaknesses?",
#     "What graphic design software is Sohail skilled in?",
#     "Is Sohail proficient in Python?",
#     "What are Sohail's video editing skills?",
#     "What are Sohail's 3D modeling skills?",
#     "What are Sohail's programming skills?",
#     "What UI/UX design tools does Sohail use?",
#     "Is Sohail experienced with Figma?",
#     "What are Sohail's areas of expertise within graphic design?",
#     "What programming languages is Sohail proficient with?",
#     "What video editing software does Sohail use?",
#     "What is Sohail's work experience?",
#     "What are Sohail's achievements in graphic design?",
#     "Tell me about Sohail's experience as a video editor.",
#     "What are Sohail's 3D modeling projects?",
#     "What are Sohail's programming achievements?",
#     "What are Sohail's UI/UX design projects?",
#     "What is Sohail's education background?",
#     "What are Sohail's skills in Adobe Photoshop?",
#     "What are Sohail's skills in Adobe Premiere Pro?",
#     "What are Sohail's skills in Autodesk Maya?",
#     "What did Sohail do at MAAC #24fps?",
#     "What was Sohail's role at Prathamesh Creations?",
#     "What is Sohail's experience with Adobe After Effects?",
#     "What are Sohail's skills in Canva?",
#     "What is Sohail's experience with 3D modeling and texturing?",
#     "What are Sohail's skills in Autodesk 3Ds Max?",
#     "What is Sohail's experience with video editing software?",
#     "What are Sohail's skills in Adobe Substance 3D Painter?",
#     "What is Sohail's experience with UI/UX design tools?",
#     "What are Sohail's skills in Blender?",
#     "What did Sohail do in 2023?",
#     "What were Sohail's achievements in 2024?",
#     "What is Sohail's work experience in 2023?",
#     "What are Sohail's skills in 2024?",
#     "What were Sohail's projects in 2023?",
#     "What did Sohail do at Freelance?",
#     "What were Sohail's achievements at MAAC #24fps?",
#     "What is Sohail's experience at Prathamesh Creations?",
#     "What were Sohail's projects at Gakarot Creations?",
#     "What is Sohail's work experience at MAAC (Utkarsh)?",
# ]
# for query in queries:
#     answer = chatbot.find_best_answer(query)
#     print(f"Query: {query}\nAI: {answer}\n")
