"""
AI Resume Analyzer - Complete Single File Application
Upload PDF resumes for comprehensive AI-powered analysis
Python 3.11 Compatible with google.generativeai
"""

import streamlit as st
import pdfplumber
import re
import json
from typing import List, Dict, Any
from google import genai

# ============================================================================
# RESUME PARSER CLASS
# ============================================================================

from google import genai


class ResumeParser:
    def __init__(self, gemini_api_key: str):
        # NEW, supported client
        self.client = genai.Client(api_key=gemini_api_key)

    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF file"""
        text = ""
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting PDF: {str(e)}")

    def extract_contact_info(self, text):
        """Extract contact information using regex"""
        contact = {}

        # Email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        contact['email'] = email_match.group(0) if email_match else None

        # Phone
        phone_pattern = r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phone_match = re.search(phone_pattern, text)
        contact['phone'] = phone_match.group(0) if phone_match else None

        # LinkedIn
        linkedin_pattern = r'(?:linkedin\.com/in/|linkedin\.com/pub/)([a-zA-Z0-9-]+)'
        linkedin_match = re.search(linkedin_pattern, text, re.IGNORECASE)
        contact['linkedin'] = linkedin_match.group(0) if linkedin_match else None

        # GitHub
        github_pattern = r'(?:github\.com/)([a-zA-Z0-9-]+)'
        github_match = re.search(github_pattern, text, re.IGNORECASE)
        contact['github'] = github_match.group(0) if github_match else None

        # Name (first line or first capitalized words)
        lines = text.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if len(line) > 3 and len(line) < 50 and not any(char.isdigit() for char in line):
                if line.replace(' ', '').isalpha():
                    contact['name'] = line
                    break

        return contact

    def structure_resume_with_ai(self, text):
        """Use Gemini to structure resume data"""
        prompt = f"""
        Analyze this resume text and extract structured information in JSON format.

        Resume Text:
        {text[:4000]}

        Return ONLY a valid JSON object with these keys:
        {{
            "personal_info": {{"name": "", "email": "", "phone": "", "linkedin": "", "github": ""}},
            "summary": "professional summary text",
            "education": [
                {{"degree": "", "institution": "", "year": "", "details": ""}}
            ],
            "skills": {{"technical": [], "soft": [], "tools": []}},
            "experience": [
                {{"title": "", "company": "", "duration": "", "responsibilities": []}}
            ],
            "projects": [
                {{"name": "", "description": "", "technologies": [], "highlights": []}}
            ]
        }}

        Rules:
        - Extract all available information
        - If section not found, return empty array/string
        - Ensure valid JSON format
        - Skills should be categorized
        - Projects section is critical - extract all details
        - Return ONLY the JSON, no markdown formatting
        """

        try:
            # Configure generation settings
            generation_config = {
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }

            response = self.client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt
            )

            json_text = response.text

            # Clean markdown formatting if present
            if json_text.startswith('```json'):
                json_text = json_text.replace('```json', '').replace('```', '').strip()
            elif json_text.startswith('```'):
                json_text = json_text.replace('```', '').strip()

            # Remove any text before the first {
            if '{' in json_text:
                json_text = json_text[json_text.index('{'):]

            # Remove any text after the last }
            if '}' in json_text:
                json_text = json_text[:json_text.rindex('}') + 1]

            structured_data = json.loads(json_text)
            return structured_data
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse AI response as JSON: {str(e)}. Response was: {json_text[:200]}")
        except Exception as e:
            raise Exception(f"AI structuring error: {str(e)}")

    def parse_resume(self, pdf_file):
        """Main parsing function"""
        # Extract text
        text = self.extract_text_from_pdf(pdf_file)

        if not text or len(text) < 100:
            raise Exception("Resume appears to be empty or too short")

        # Extract contact info with regex
        contact_info = self.extract_contact_info(text)

        # Structure with AI
        structured_data = self.structure_resume_with_ai(text)

        # Merge contact info (prioritize regex extraction)
        if 'personal_info' not in structured_data:
            structured_data['personal_info'] = {}

        for key, value in contact_info.items():
            if value:
                structured_data['personal_info'][key] = value

        # Add raw text for further analysis
        structured_data['raw_text'] = text

        return structured_data


# ============================================================================
# PROFILE MATCHER CLASS
# ============================================================================

class ProfileMatcher:
    def __init__(self):
        self.job_profiles = {
            "Full Stack Developer": {
                "technical_skills": ["JavaScript", "Python", "React", "Node.js", "SQL", "MongoDB", "REST API", "Git",
                                     "HTML", "CSS"],
                "soft_skills": ["Problem Solving", "Team Collaboration", "Communication"],
                "tools": ["Docker", "AWS", "Jenkins", "Postman"]
            },
            "Data Scientist": {
                "technical_skills": ["Python", "R", "Machine Learning", "Deep Learning", "SQL", "Statistics", "Pandas",
                                     "NumPy", "Scikit-learn", "TensorFlow"],
                "soft_skills": ["Analytical Thinking", "Problem Solving", "Communication"],
                "tools": ["Jupyter", "Tableau", "Power BI", "Git"]
            },
            "DevOps Engineer": {
                "technical_skills": ["Linux", "Docker", "Kubernetes", "CI/CD", "Python", "Bash", "Terraform", "Ansible",
                                     "AWS", "Azure"],
                "soft_skills": ["Problem Solving", "Automation Mindset", "Team Collaboration"],
                "tools": ["Jenkins", "Git", "Prometheus", "Grafana"]
            },
            "Frontend Developer": {
                "technical_skills": ["JavaScript", "React", "Angular", "Vue.js", "HTML", "CSS", "TypeScript",
                                     "Responsive Design", "REST API"],
                "soft_skills": ["Creativity", "Attention to Detail", "User Focus"],
                "tools": ["Webpack", "Git", "Figma", "npm"]
            },
            "Backend Developer": {
                "technical_skills": ["Python", "Java", "Node.js", "SQL", "MongoDB", "REST API", "GraphQL",
                                     "Microservices", "Redis"],
                "soft_skills": ["Problem Solving", "System Design", "Team Collaboration"],
                "tools": ["Docker", "Postman", "Git", "RabbitMQ"]
            },
            "Mobile Developer": {
                "technical_skills": ["React Native", "Flutter", "Swift", "Kotlin", "Java", "iOS", "Android", "REST API",
                                     "Firebase"],
                "soft_skills": ["User Experience", "Problem Solving", "Adaptability"],
                "tools": ["Xcode", "Android Studio", "Git", "Postman"]
            },
            "ML Engineer": {
                "technical_skills": ["Python", "Machine Learning", "Deep Learning", "PyTorch", "TensorFlow", "MLOps",
                                     "Feature Engineering", "Model Deployment"],
                "soft_skills": ["Research Mindset", "Problem Solving", "Experimentation"],
                "tools": ["Docker", "Kubernetes", "MLflow", "Git"]
            },
            "Cloud Architect": {
                "technical_skills": ["AWS", "Azure", "GCP", "Kubernetes", "Terraform", "Networking", "Security",
                                     "Serverless", "Microservices"],
                "soft_skills": ["Architecture Design", "Problem Solving", "Strategic Thinking"],
                "tools": ["Docker", "Jenkins", "Ansible", "Git"]
            }
        }

    def normalize_skill(self, skill):
        """Normalize skill for comparison"""
        return skill.lower().strip()

    def get_all_resume_skills(self, resume_data):
        """Extract all skills from resume"""
        skills_obj = resume_data.get('skills', {})
        all_skills = []

        if isinstance(skills_obj, dict):
            all_skills.extend(skills_obj.get('technical', []))
            all_skills.extend(skills_obj.get('soft', []))
            all_skills.extend(skills_obj.get('tools', []))
        elif isinstance(skills_obj, list):
            all_skills.extend(skills_obj)

        # Also extract from experience and projects
        for exp in resume_data.get('experience', []):
            if isinstance(exp, dict):
                resp = exp.get('responsibilities', [])
                if isinstance(resp, list):
                    all_skills.extend([r for r in resp if isinstance(r, str)])

        for proj in resume_data.get('projects', []):
            if isinstance(proj, dict):
                techs = proj.get('technologies', [])
                if isinstance(techs, list):
                    all_skills.extend(techs)

        return [self.normalize_skill(s) for s in all_skills if s]

    def calculate_skill_match(self, resume_skills, required_skills):
        """Calculate percentage match for skills"""
        if not required_skills:
            return 0, [], []

        resume_skills_norm = set(resume_skills)
        required_skills_norm = set([self.normalize_skill(s) for s in required_skills])

        matched = []
        missing = []

        for req_skill in required_skills:
            req_norm = self.normalize_skill(req_skill)
            found = False
            for res_skill in resume_skills_norm:
                if req_norm in res_skill or res_skill in req_norm:
                    matched.append(req_skill)
                    found = True
                    break
            if not found:
                missing.append(req_skill)

        match_percentage = (len(matched) / len(required_skills)) * 100
        return match_percentage, matched, missing

    def analyze_profile_match(self, resume_data):
        """Analyze resume against all job profiles"""
        resume_skills = self.get_all_resume_skills(resume_data)
        results = []

        for profile_name, profile_reqs in self.job_profiles.items():
            tech_match, tech_matched, tech_missing = self.calculate_skill_match(
                resume_skills, profile_reqs['technical_skills']
            )

            soft_match, soft_matched, soft_missing = self.calculate_skill_match(
                resume_skills, profile_reqs['soft_skills']
            )

            tools_match, tools_matched, tools_missing = self.calculate_skill_match(
                resume_skills, profile_reqs['tools']
            )

            overall_match = (tech_match * 0.6) + (soft_match * 0.2) + (tools_match * 0.2)

            strengths = tech_matched[:5] if tech_matched else []
            all_missing = tech_missing + soft_missing + tools_missing

            results.append({
                "profile": profile_name,
                "overall_match": round(overall_match, 1),
                "technical_match": round(tech_match, 1),
                "soft_skills_match": round(soft_match, 1),
                "tools_match": round(tools_match, 1),
                "strengths": strengths,
                "missing_skills": all_missing[:8]
            })

        results.sort(key=lambda x: x['overall_match'], reverse=True)
        return results[:5]


# ============================================================================
# ATS ANALYZER CLASS
# ============================================================================

class ATSAnalyzer:
    def __init__(self):
        self.keywords = {
            "action_verbs": ["developed", "designed", "implemented", "managed", "led", "created",
                             "built", "improved", "optimized", "achieved", "delivered", "analyzed"],
            "technical_terms": ["api", "database", "cloud", "agile", "scrum", "ci/cd", "testing",
                                "deployment", "architecture", "framework", "algorithm", "optimization"],
            "soft_skills": ["leadership", "communication", "teamwork", "problem-solving",
                            "collaboration", "analytical", "critical thinking"],
            "certifications": ["certified", "certification", "aws", "azure", "google cloud",
                               "pmp", "scrum master"],
            "education": ["bachelor", "master", "phd", "degree", "university", "college", "gpa"],
            "metrics": [r'\d+%', r'\d+x', r'\$\d+', r'\d+\+']
        }

    def calculate_keyword_score(self, resume_data):
        """Calculate keyword matching score"""
        text = resume_data.get('raw_text', '').lower()

        total_keywords = 0
        found_keywords = 0
        keyword_details = {}

        for category, keywords in self.keywords.items():
            if category == "metrics":
                patterns_found = 0
                for pattern in keywords:
                    if re.search(pattern, text):
                        patterns_found += 1
                keyword_details[category] = {"found": patterns_found, "total": len(keywords)}
                total_keywords += len(keywords)
                found_keywords += patterns_found
            else:
                found = [kw for kw in keywords if kw in text]
                keyword_details[category] = {"found": len(found), "total": len(keywords), "list": found}
                total_keywords += len(keywords)
                found_keywords += len(found)

        score = (found_keywords / total_keywords) * 100
        return round(score, 1), keyword_details

    def calculate_completeness_score(self, resume_data):
        """Calculate resume completeness score"""
        sections = {
            "personal_info": 15,
            "summary": 10,
            "education": 15,
            "skills": 20,
            "experience": 25,
            "projects": 15
        }

        total_points = 0
        max_points = sum(sections.values())
        section_status = {}

        for section, points in sections.items():
            data = resume_data.get(section, None)

            if section == "personal_info":
                if isinstance(data, dict):
                    essential = ['name', 'email', 'phone']
                    filled = sum([1 for field in essential if data.get(field)])
                    earned = (filled / len(essential)) * points
                    total_points += earned
                    section_status[section] = f"{filled}/{len(essential)} fields"

            elif section == "summary":
                if data and len(str(data)) > 50:
                    total_points += points
                    section_status[section] = "Present"
                else:
                    section_status[section] = "Missing/Too Short"

            elif section in ["education", "experience", "projects"]:
                if isinstance(data, list) and len(data) > 0:
                    entries = min(len(data), 3)
                    earned = (entries / 3) * points
                    total_points += earned
                    section_status[section] = f"{len(data)} entries"
                else:
                    section_status[section] = "Missing"

            elif section == "skills":
                if isinstance(data, dict):
                    skill_count = sum([len(v) for v in data.values() if isinstance(v, list)])
                    if skill_count > 5:
                        total_points += points
                        section_status[section] = f"{skill_count} skills"
                    else:
                        total_points += (skill_count / 5) * points
                        section_status[section] = f"{skill_count} skills (needs more)"
                elif isinstance(data, list) and len(data) > 5:
                    total_points += points
                    section_status[section] = f"{len(data)} skills"
                else:
                    section_status[section] = "Missing"

        score = (total_points / max_points) * 100
        return round(score, 1), section_status

    def calculate_skill_relevance(self, resume_data):
        """Calculate skill relevance score"""
        hot_skills = {
            "programming": ["python", "javascript", "java", "typescript", "go", "rust"],
            "frameworks": ["react", "angular", "vue", "node.js", "django", "flask", "spring"],
            "data": ["sql", "mongodb", "postgresql", "redis", "elasticsearch"],
            "cloud": ["aws", "azure", "gcp", "kubernetes", "docker"],
            "ai_ml": ["machine learning", "deep learning", "tensorflow", "pytorch", "nlp"],
            "devops": ["ci/cd", "jenkins", "github actions", "terraform", "ansible"],
            "mobile": ["react native", "flutter", "swift", "kotlin"]
        }

        text = resume_data.get('raw_text', '').lower()
        skills_obj = resume_data.get('skills', {})

        all_skills = []
        if isinstance(skills_obj, dict):
            for skill_list in skills_obj.values():
                if isinstance(skill_list, list):
                    all_skills.extend([s.lower() for s in skill_list])

        total_hot_skills = sum([len(skills) for skills in hot_skills.values()])
        found_hot_skills = 0
        category_breakdown = {}

        for category, skills in hot_skills.items():
            found = [skill for skill in skills if skill in text or skill in ' '.join(all_skills)]
            found_hot_skills += len(found)
            category_breakdown[category] = {
                "found": len(found),
                "total": len(skills),
                "skills": found
            }

        score = (found_hot_skills / total_hot_skills) * 100
        return round(score, 1), category_breakdown

    def analyze_ats_compatibility(self, resume_data):
        """Perform complete ATS analysis"""
        keyword_score, keyword_details = self.calculate_keyword_score(resume_data)
        completeness_score, section_status = self.calculate_completeness_score(resume_data)
        relevance_score, skill_breakdown = self.calculate_skill_relevance(resume_data)

        overall_score = (keyword_score * 0.4) + (completeness_score * 0.35) + (relevance_score * 0.25)

        recommendations = []
        if keyword_score < 70:
            recommendations.append("Add more action verbs (developed, implemented, managed)")
            recommendations.append("Include quantifiable metrics and achievements")

        if completeness_score < 80:
            missing_sections = [sec for sec, status in section_status.items() if "Missing" in status]
            if missing_sections:
                recommendations.append(f"Complete missing sections: {', '.join(missing_sections)}")

        if relevance_score < 60:
            recommendations.append("Add more modern, in-demand technical skills")
            recommendations.append("Consider adding cloud technologies (AWS/Azure/GCP)")

        return {
            "overall_score": round(overall_score, 1),
            "keyword_score": keyword_score,
            "completeness_score": completeness_score,
            "relevance_score": relevance_score,
            "section_status": section_status,
            "skill_breakdown": skill_breakdown,
            "recommendations": recommendations,
            "grade": self._get_grade(overall_score)
        }

    def _get_grade(self, score):
        """Convert score to letter grade"""
        if score >= 90:
            return "A (Excellent)"
        elif score >= 80:
            return "B (Good)"
        elif score >= 70:
            return "C (Average)"
        elif score >= 60:
            return "D (Below Average)"
        else:
            return "F (Needs Improvement)"


# ============================================================================
# QUESTION GENERATOR CLASS
# ============================================================================

class QuestionGenerator:
    def __init__(self, gemini_api_key: str):
        self.client = genai.Client(api_key=gemini_api_key)

    def generate_interview_questions(self, resume_data, top_profile):
        skills = resume_data.get('skills', {})
        projects = resume_data.get('projects', [])
        experience = resume_data.get('experience', [])

        top_skills = []
        if isinstance(skills, dict):
            for skill_list in skills.values():
                if isinstance(skill_list, list):
                    top_skills.extend(skill_list[:3])
        top_skills = top_skills[:5]

        prompt = f"""
        Generate exactly 8 interview questions for {top_profile['profile']}.

        Rules:
        - 3 technical
        - 3 project-based
        - 2 behavioral

        Return ONLY valid JSON array:
        [
          {{"question": "...", "type": "technical", "focus_area": "..."}}
        ]
        """

        response = self.client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )

        json_text = response.text.strip()

        if json_text.startswith("```"):
            json_text = json_text.replace("```json", "").replace("```", "").strip()

        json_text = json_text[json_text.find("["): json_text.rfind("]") + 1]

        return json.loads(json_text)

    def generate_interview_questions(self, resume_data, top_profile):
        """Generate personalized interview questions"""

        skills = resume_data.get('skills', {})
        projects = resume_data.get('projects', [])
        experience = resume_data.get('experience', [])

        top_skills = []
        if isinstance(skills, dict):
            for skill_list in skills.values():
                if isinstance(skill_list, list):
                    top_skills.extend(skill_list[:3])
        top_skills = top_skills[:5]

        project_details = []
        for proj in projects[:3]:
            if isinstance(proj, dict):
                project_details.append({
                    "name": proj.get('name', 'N/A'),
                    "technologies": proj.get('technologies', []),
                    "description": proj.get('description', '')
                })

        prompt = f"""
        You are an expert technical interviewer. Generate 8 interview questions for a candidate applying for: {top_profile['profile']}

        Candidate Profile:
        - Top Skills: {', '.join(top_skills) if top_skills else 'General skills'}
        - Projects: {len(projects)} projects
        - Experience: {len(experience)} positions

        Generate exactly 8 questions:
        - 3 Technical questions
        - 3 Project-based questions
        - 2 Behavioral questions

        Return ONLY a valid JSON array:
        [
            {{"question": "Your question?", "type": "technical", "focus_area": "topic"}},
            ...
        ]

        No markdown formatting, just the JSON array.
        """

        try:
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            }

            response = self.client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt
            )

            json_text = response.text.strip()

            if json_text.startswith('```json'):
                json_text = json_text.replace('```json', '').replace('```', '').strip()
            elif json_text.startswith('```'):
                json_text = json_text.replace('```', '').strip()

            if '[' in json_text:
                json_text = json_text[json_text.index('['):]
            if ']' in json_text:
                json_text = json_text[:json_text.rindex(']') + 1]

            questions = json.loads(json_text)

            if not isinstance(questions, list):
                raise ValueError("Response is not a list")

            validated_questions = []
            for q in questions:
                if isinstance(q, dict) and 'question' in q:
                    validated_questions.append({
                        "question": q.get('question', ''),
                        "type": q.get('type', 'general'),
                        "focus_area": q.get('focus_area', 'general')
                    })

            return validated_questions[:8]

        except:
            return self._generate_fallback_questions(top_profile['profile'])

    def _generate_fallback_questions(self, profile):
        """Generate fallback questions if AI fails"""
        return [
            {"question": "Can you walk me through a challenging technical problem you've solved?", "type": "technical",
             "focus_area": "problem-solving"},
            {"question": "Describe your most complex project and the technologies you used.", "type": "project",
             "focus_area": "project experience"},
            {"question": "How do you approach debugging a difficult issue in production?", "type": "technical",
             "focus_area": "debugging"},
            {"question": "Tell me about a time you had to learn a new technology quickly.", "type": "behavioral",
             "focus_area": "learning agility"},
            {"question": f"What interests you most about working as a {profile}?", "type": "behavioral",
             "focus_area": "motivation"},
            {"question": "How do you ensure code quality and maintainability?", "type": "technical",
             "focus_area": "code quality"},
            {"question": "Describe a situation where you collaborated with a difficult team member.",
             "type": "behavioral", "focus_area": "teamwork"},
            {"question": "What was the biggest technical challenge in your recent project?", "type": "project",
             "focus_area": "challenges"}
        ]


# ============================================================================
# STREAMLIT UI FUNCTIONS
# ============================================================================

def get_score_class(score):
    if score >= 80:
        return "score-excellent"
    elif score >= 60:
        return "score-good"
    elif score >= 40:
        return "score-average"
    else:
        return "score-poor"


def display_personal_info(personal_info):
    st.markdown('<p class="section-header">📋 Personal Information</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Name:** {personal_info.get('name', 'N/A')}")
        st.write(f"**Email:** {personal_info.get('email', 'N/A')}")
    with col2:
        st.write(f"**Phone:** {personal_info.get('phone', 'N/A')}")
        if personal_info.get('linkedin'):
            st.write(f"**LinkedIn:** {personal_info.get('linkedin')}")
        if personal_info.get('github'):
            st.write(f"**GitHub:** {personal_info.get('github')}")


def display_summary(summary):
    st.markdown('<p class="section-header">💼 Professional Summary</p>', unsafe_allow_html=True)
    if summary and len(summary) > 10:
        st.write(summary)
    else:
        st.info("No professional summary found in resume")


def display_education(education):
    st.markdown('<p class="section-header">🎓 Education</p>', unsafe_allow_html=True)

    if education and len(education) > 0:
        for edu in education:
            with st.container():
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.write(f"**{edu.get('degree', 'Degree')}** - {edu.get('institution', 'Institution')}")
                st.write(f"Year: {edu.get('year', 'N/A')}")
                if edu.get('details'):
                    st.write(f"Details: {edu.get('details')}")
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No education information found")


def display_skills(skills):
    st.markdown('<p class="section-header">🛠️ Skills</p>', unsafe_allow_html=True)

    if isinstance(skills, dict):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Technical Skills:**")
            tech_skills = skills.get('technical', [])
            if tech_skills:
                for skill in tech_skills:
                    st.write(f"• {skill}")
            else:
                st.write("None listed")

        with col2:
            st.write("**Soft Skills:**")
            soft_skills = skills.get('soft', [])
            if soft_skills:
                for skill in soft_skills:
                    st.write(f"• {skill}")
            else:
                st.write("None listed")

        with col3:
            st.write("**Tools & Technologies:**")
            tools = skills.get('tools', [])
            if tools:
                for tool in tools:
                    st.write(f"• {tool}")
            else:
                st.write("None listed")
    elif isinstance(skills, list):
        for skill in skills:
            st.write(f"• {skill}")
    else:
        st.info("No skills information found")


def display_experience(experience):
    st.markdown('<p class="section-header">💼 Work Experience</p>', unsafe_allow_html=True)

    if experience and len(experience) > 0:
        for exp in experience:
            with st.expander(f"{exp.get('title', 'Position')} at {exp.get('company', 'Company')}"):
                st.write(f"**Duration:** {exp.get('duration', 'N/A')}")
                st.write("**Responsibilities:**")
                responsibilities = exp.get('responsibilities', [])
                if responsibilities:
                    for resp in responsibilities:
                        st.write(f"• {resp}")
    else:
        st.info("No work experience found")


def display_projects(projects):
    st.markdown('<p class="section-header">🚀 Projects (Key Section)</p>', unsafe_allow_html=True)

    if projects and len(projects) > 0:
        for idx, proj in enumerate(projects, 1):
            with st.expander(f"Project {idx}: {proj.get('name', 'Unnamed Project')}", expanded=True):
                st.write(f"**Description:** {proj.get('description', 'No description')}")

                technologies = proj.get('technologies', [])
                if technologies:
                    st.write("**Technologies Used:**")
                    st.write(", ".join(technologies))

                highlights = proj.get('highlights', [])
                if highlights:
                    st.write("**Key Highlights:**")
                    for highlight in highlights:
                        st.write(f"• {highlight}")
    else:
        st.warning("⚠️ No projects found - This is a critical section!")


def display_profile_matches(matches):
    st.markdown('<p class="section-header">🎯 Job Profile Matching</p>', unsafe_allow_html=True)

    for match in matches:
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader(match['profile'])
            with col2:
                score_class = get_score_class(match['overall_match'])
                st.markdown(f'<p class="{score_class}">{match["overall_match"]}%</p>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Technical", f"{match['technical_match']}%")
            with col2:
                st.metric("Soft Skills", f"{match['soft_skills_match']}%")
            with col3:
                st.metric("Tools", f"{match['tools_match']}%")

            if match['strengths']:
                st.write("**✅ Strengths:**", ", ".join(match['strengths']))

            if match['missing_skills']:
                with st.expander("❌ Missing Skills"):
                    st.write(", ".join(match['missing_skills']))

            st.markdown('</div>', unsafe_allow_html=True)


def display_ats_analysis(ats_results):
    st.markdown('<p class="section-header">🤖 ATS Compatibility Analysis</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        score_class = get_score_class(ats_results['overall_score'])
        st.markdown(
            f'<div class="metric-card"><p>Overall ATS</p><p class="{score_class}">{ats_results["overall_score"]}%</p><p>{ats_results["grade"]}</p></div>',
            unsafe_allow_html=True)
    with col2:
        st.markdown(
            f'<div class="metric-card"><p>Keywords</p><p class="score-good">{ats_results["keyword_score"]}%</p></div>',
            unsafe_allow_html=True)
    with col3:
        st.markdown(
            f'<div class="metric-card"><p>Complete</p><p class="score-good">{ats_results["completeness_score"]}%</p></div>',
            unsafe_allow_html=True)
    with col4:
        st.markdown(
            f'<div class="metric-card"><p>Relevance</p><p class="score-good">{ats_results["relevance_score"]}%</p></div>',
            unsafe_allow_html=True)

    st.write("### 📊 Section Analysis")
    col1, col2 = st.columns(2)

    section_items = list(ats_results['section_status'].items())
    mid = len(section_items) // 2

    with col1:
        for section, status in section_items[:mid]:
            st.write(f"**{section.replace('_', ' ').title()}:** {status}")
    with col2:
        for section, status in section_items[mid:]:
            st.write(f"**{section.replace('_', ' ').title()}:** {status}")

    if ats_results['recommendations']:
        st.write("### 💡 Recommendations")
        for rec in ats_results['recommendations']:
            st.info(rec)


def display_interview_questions(questions):
    st.markdown('<p class="section-header">❓ Personalized Interview Questions</p>', unsafe_allow_html=True)

    technical = [q for q in questions if q['type'] == 'technical']
    project = [q for q in questions if q['type'] == 'project']
    behavioral = [q for q in questions if q['type'] == 'behavioral']

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("### 🔧 Technical")
        for idx, q in enumerate(technical, 1):
            with st.expander(f"Q{idx}: {q['focus_area']}"):
                st.write(q['question'])

    with col2:
        st.write("### 🚀 Project")
        for idx, q in enumerate(project, 1):
            with st.expander(f"Q{idx}: {q['focus_area']}"):
                st.write(q['question'])

    with col3:
        st.write("### 🧠 Behavioral")
        for idx, q in enumerate(behavioral, 1):
            with st.expander(f"Q{idx}: {q['focus_area']}"):
                st.write(q['question'])


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.set_page_config(
        page_title="AI Resume Analyzer",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: #1f77b4;
            margin-bottom: 2rem;
        }
        .section-header {
            font-size: 1.8rem;
            font-weight: bold;
            color: #2c3e50;
            margin-top: 2rem;
            margin-bottom: 1rem;
            border-bottom: 3px solid #1f77b4;
            padding-bottom: 0.5rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
            margin: 1rem 0;
        }
        .score-excellent {
            color: #28a745;
            font-weight: bold;
            font-size: 2rem;
        }
        .score-good {
            color: #17a2b8;
            font-weight: bold;
            font-size: 2rem;
        }
        .score-average {
            color: #ffc107;
            font-weight: bold;
            font-size: 2rem;
        }
        .score-poor {
            color: #dc3545;
            font-weight: bold;
            font-size: 2rem;
        }
        .info-box {
            background-color: #e7f3ff;
            padding: 1rem;
            border-radius: 5px;
            margin: 0.5rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="main-header">📄 AI Resume Analyzer</p>', unsafe_allow_html=True)
    st.write("Upload your resume in PDF format for comprehensive AI-powered analysis")

    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("Gemini API Key", type="password", help="Get your key from Google AI Studio")

        st.write("---")
        st.write("### About")
        st.info("AI-powered resume analysis:\n\n"
                "• Data extraction\n"
                "• Job profile matching\n"
                "• ATS scoring\n"
                "• Interview questions")

        st.write("---")
        st.markdown("[Get API Key](https://makersuite.google.com/app/apikey)")

    uploaded_file = st.file_uploader("Upload Resume (PDF only)", type=['pdf'])

    if uploaded_file and api_key:
        if st.button("🔍 Analyze Resume", type="primary"):
            with st.spinner("Analyzing... This may take 30-60 seconds"):
                try:
                    parser = ResumeParser(api_key)
                    matcher = ProfileMatcher()
                    ats_analyzer = ATSAnalyzer()
                    question_gen = QuestionGenerator(api_key)

                    with st.status("Parsing resume..."):
                        resume_data = parser.parse_resume(uploaded_file)

                    st.success("✅ Resume parsed successfully!")

                    display_personal_info(resume_data.get('personal_info', {}))
                    display_summary(resume_data.get('summary', ''))
                    display_education(resume_data.get('education', []))
                    display_skills(resume_data.get('skills', {}))
                    display_experience(resume_data.get('experience', []))
                    display_projects(resume_data.get('projects', []))

                    with st.status("Matching profiles..."):
                        matches = matcher.analyze_profile_match(resume_data)
                    display_profile_matches(matches)

                    with st.status("ATS analysis..."):
                        ats_results = ats_analyzer.analyze_ats_compatibility(resume_data)
                    display_ats_analysis(ats_results)

                    with st.status("Generating questions..."):
                        questions = question_gen.generate_interview_questions(resume_data, matches[0])
                    display_interview_questions(questions)

                    st.success("🎉 Analysis complete!")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.write("Check your API key and PDF file.")

    elif uploaded_file and not api_key:
        st.warning("⚠️ Please enter your Gemini API key")
    else:
        st.info("👆 Upload a PDF resume to start")


if __name__ == "__main__":
    main()