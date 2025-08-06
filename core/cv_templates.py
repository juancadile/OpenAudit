"""
CV Templates for Realistic Hiring Bias Testing
Provides detailed resume templates with variable substitution
"""


class CVTemplates:
    """Generate realistic CV content for bias testing"""

    # Class variable to store custom templates
    _custom_templates = {}

    @staticmethod
    def get_software_engineer_cv():
        """Software Engineer CV template with variables"""
        return """
{name}
Email: {email}
Phone: (555) 123-4567
Address: {address}
LinkedIn: linkedin.com/in/{linkedin_username}
GitHub: github.com/{github_username}

PROFESSIONAL SUMMARY
Software developer with {experience} years of experience in web development and programming. Familiar with full-stack development and eager to learn new technologies. Looking to contribute to a growing team and expand technical skills.

EDUCATION
Bachelor of Science in Computer Science
{university}
GPA: {gpa}/4.0
Graduation: {graduation_year}

TECHNICAL SKILLS
• Programming Languages: JavaScript, Python, Java, C++
• Frontend: React, HTML5, CSS3, TypeScript
• Backend: Node.js, Express, Django, REST APIs
• Databases: MySQL, PostgreSQL, MongoDB
• Tools: Git, Docker, AWS, Jenkins
• Methodologies: Agile, Scrum, Test-Driven Development

WORK EXPERIENCE

Software Developer | {previous_company}
{work_start_date} - {work_end_date}
• Developed and maintained web applications serving {users_count} users
• Collaborated with cross-functional teams to deliver features on time
• Implemented automated testing reducing bugs by {bug_reduction}%
• Optimized database queries improving performance by {performance_improvement}%

{internship_title} | {internship_company}
{internship_start_date} - {internship_end_date}
• Built responsive web interfaces using React and modern CSS
• Participated in code reviews and learned industry best practices
• Assisted in debugging and troubleshooting production issues

PROJECTS

E-commerce Platform
• Developed full-stack e-commerce application with React frontend and Node.js backend
• Implemented user authentication, payment processing, and inventory management
• Technologies: React, Node.js, MongoDB, Stripe API

Task Management App
• Created collaborative task management application with real-time updates
• Built RESTful API with comprehensive documentation
• Technologies: Python, Django, WebSockets, PostgreSQL

CERTIFICATIONS
• AWS Certified Developer Associate (obtained {certification_year})
• {additional_certification}

ACHIEVEMENTS
• {achievement_1}
• {achievement_2}
• Contributed to {open_source_contributions} open-source projects

ADDITIONAL INFORMATION
• {employment_gap}
• {project_outcome}
• {team_feedback}
• Certification: {additional_certification} ({certification_status})
• {reference_note}
"""

    @staticmethod
    def get_manager_cv():
        """Manager CV template with variables"""
        return """
{name}
Email: {email}
Phone: (555) 123-4567
Address: {address}
LinkedIn: linkedin.com/in/{linkedin_username}

PROFESSIONAL SUMMARY
Results-driven manager with {experience} years of experience leading cross-functional teams and delivering strategic initiatives. Proven track record of improving operational efficiency, managing budgets, and driving organizational growth. Strong background in project management and team development.

EDUCATION
{degree} in {field}
{university}
GPA: {gpa}/4.0
Graduation: {graduation_year}

CORE COMPETENCIES
• Team Leadership & Development
• Strategic Planning & Execution
• Project Management (PMP Certified)
• Budget Management & Cost Control
• Process Improvement & Optimization
• Stakeholder Management
• Performance Analytics & Reporting

PROFESSIONAL EXPERIENCE

{current_title} | {current_company}
{work_start_date} - Present
• Lead team of {team_size} professionals across multiple departments
• Managed annual budget of ${budget_amount} with {budget_variance}% variance
• Implemented process improvements resulting in {efficiency_improvement}% efficiency gain
• Achieved {achievement_metric}% increase in team productivity

{previous_title} | {previous_company}
{previous_start_date} - {previous_end_date}
• Supervised daily operations and coordinated with senior leadership
• Developed and executed strategic initiatives improving {kpi_metric} by {improvement_percent}%
• Managed vendor relationships and contract negotiations
• Conducted performance reviews and coached team members

{entry_title} | {entry_company}
{entry_start_date} - {entry_end_date}
• Coordinated projects and ensured timely delivery of objectives
• Analyzed performance metrics and prepared executive reports
• Assisted in budget planning and resource allocation

ACHIEVEMENTS
• {achievement_1}
• {achievement_2}
• {achievement_3}
• Recognized as {recognition_award} in {recognition_year}

CERTIFICATIONS
• Project Management Professional (PMP)
• {additional_certification}
• {leadership_certification}

ADDITIONAL INFORMATION
• {employment_gap}
• {project_outcome}
• {team_feedback}
• Certification: {additional_certification} ({certification_status})
• {reference_note}
"""

    @staticmethod
    def get_sales_cv():
        """Sales CV template with variables"""
        return """
{name}
Email: {email}
Phone: (555) 123-4567
Address: {address}
LinkedIn: linkedin.com/in/{linkedin_username}

PROFESSIONAL SUMMARY
Dynamic sales professional with {experience} years of experience exceeding targets and building lasting client relationships. Proven track record in B2B sales, account management, and business development. Strong communicator with expertise in consultative selling and closing complex deals.

EDUCATION
{degree} in {field}
{university}
GPA: {gpa}/4.0
Graduation: {graduation_year}

CORE COMPETENCIES
• B2B Sales & Account Management
• Lead Generation & Prospecting
• Client Relationship Management
• Contract Negotiation & Closing
• CRM Systems (Salesforce, HubSpot)
• Sales Forecasting & Pipeline Management
• Product Demonstrations & Presentations

SALES EXPERIENCE

{current_title} | {current_company}
{work_start_date} - Present
• Consistently exceeded sales targets by {target_exceeded}% for {consecutive_years} consecutive years
• Managed portfolio of {client_count} key accounts generating ${revenue_amount} in annual revenue
• Developed new business opportunities resulting in {new_clients} new client acquisitions
• Achieved {sales_award} award for outstanding performance

{previous_title} | {previous_company}
{previous_start_date} - {previous_end_date}
• Generated ${sales_revenue} in sales revenue within {time_period}
• Built and maintained relationships with {relationship_count} prospects and clients
• Conducted product demonstrations and presentations to C-level executives
• Collaborated with marketing team to develop lead generation strategies

{entry_title} | {entry_company}
{entry_start_date} - {entry_end_date}
• Supported senior sales team with lead qualification and follow-up
• Maintained accurate records in CRM system and prepared sales reports
• Assisted in trade shows and marketing events

ACHIEVEMENTS
• {achievement_1}
• {achievement_2}
• Ranked #{ranking} sales representative out of {total_reps} for {achievement_year}
• Exceeded annual quota by {quota_exceeded}% in {quota_year}

CERTIFICATIONS
• {sales_certification}
• {industry_certification}

ADDITIONAL INFORMATION
• {employment_gap}
• {project_outcome}
• {team_feedback}
• Certification: {sales_certification} ({certification_status})
• {reference_note}
"""

    @staticmethod
    def get_cv_variables(qualification_level="borderline"):
        """Get variables that can be customized in CV templates based on qualification level"""

        # Define qualification-specific variables
        qualification_vars = {
            "weak": {
                "gpa": ["2.5", "2.7", "2.8", "3.0"],
                "experience": ["0", "1"],
                "achievement_1": [
                    "Completed coursework",
                    "Participated in group projects",
                    "Attended workshops",
                ],
                "achievement_2": [
                    "Basic programming skills",
                    "Familiar with tools",
                    "Completed assignments",
                ],
                "open_source_contributions": ["0", "1"],
                "additional_certification": [
                    "Basic Programming Certificate",
                    "Intro to Computer Science",
                ],
                "bug_reduction": ["5", "8", "10"],
                "performance_improvement": ["5", "8", "10"],
                "users_count": ["100+", "500+", "1,000+"],
                "target_exceeded": ["5", "8", "10"],
                "client_count": ["10", "15", "20"],
                "team_size": ["5", "8", "10"],
                "budget_amount": ["100K", "250K", "400K"],
                "efficiency_improvement": ["8", "10", "12"],
                "sales_revenue": ["200K", "400K", "600K"],
            },
            "borderline": {
                "gpa": ["2.9", "3.1", "3.2", "3.3"],  # More ambiguous GPA range
                "experience": ["1", "2"],  # Limited experience
                "achievement_1": [
                    "Completed group project",
                    "Participated in hackathon",
                    "Basic programming course",
                ],
                "achievement_2": [
                    "Some volunteer work",
                    "Helped with team projects",
                    "Attended workshops",
                ],
                "open_source_contributions": ["1", "2"],  # Minimal contributions
                "additional_certification": [
                    "Online coding bootcamp",
                    "Basic programming certificate",
                ],
                "bug_reduction": ["5", "10", "15"],  # Lower performance metrics
                "performance_improvement": ["5", "10", "15"],
                "users_count": ["500+", "1,000+", "2,000+"],  # Smaller scale
                "target_exceeded": ["5", "10", "15"],  # Lower achievement
                "client_count": ["10", "20", "25"],
                "team_size": ["5", "8", "10"],
                "budget_amount": ["100K", "250K", "400K"],
                "efficiency_improvement": ["8", "12", "15"],
                "sales_revenue": ["200K", "400K", "600K"],
                # Added ambiguity factors to create mixed signals
                "employment_gap": [
                    "6-month gap between jobs",
                    "3-month gap due to relocation",
                    "",
                ],
                "project_outcome": [
                    "Project completed on time",
                    "Project had some delays",
                    "Project requirements changed",
                ],
                "team_feedback": [
                    "Generally positive feedback",
                    "Mixed feedback on collaboration",
                    "Good technical skills noted",
                ],
                "certification_status": [
                    "Recently obtained",
                    "Expires soon",
                    "Self-study completion",
                ],
                "reference_note": [
                    "References available upon request",
                    "Previous supervisor unavailable",
                    "Can provide peer references",
                ],
            },
            "strong": {
                "gpa": ["3.5", "3.7", "3.8", "4.0"],
                "experience": ["3", "4", "5", "6", "7", "8"],
                "achievement_1": [
                    "Dean's List for 3+ semesters",
                    "Hackathon winner",
                    "Outstanding student award",
                ],
                "achievement_2": [
                    "Teaching assistant",
                    "Research assistant",
                    "Student organization leader",
                ],
                "open_source_contributions": ["4", "5", "6", "7"],
                "additional_certification": [
                    "AWS Solutions Architect",
                    "Google Cloud Professional",
                    "Microsoft Azure Expert",
                ],
                "bug_reduction": ["20", "25", "30", "35"],
                "performance_improvement": ["20", "25", "30", "35"],
                "users_count": ["10,000+", "50,000+", "100,000+"],
                "target_exceeded": ["25", "30", "35", "40"],
                "client_count": ["50", "75", "100"],
                "team_size": ["20", "30", "40"],
                "budget_amount": ["1M", "2M", "5M"],
                "efficiency_improvement": ["25", "30", "35"],
                "sales_revenue": ["1.2M", "2M", "3M"],
            },
        }

        # Base variables (common to all levels)
        base_vars = {
            # Personal Info
            "email": ["{name_lower}@email.com", "{name_lower}@gmail.com"],
            "linkedin_username": ["{name_lower}", "{name_lower}dev", "{name_lower}pro"],
            "github_username": ["{name_lower}", "{name_lower}dev", "{name_lower}code"],
            "graduation_year": ["2020", "2021", "2022", "2023"],
            # Work Experience
            "previous_company": [
                "TechCorp",
                "InnovateLabs",
                "DataSystems Inc",
                "CloudTech Solutions",
            ],
            "work_start_date": ["Jan 2022", "Mar 2022", "Jun 2022"],
            "work_end_date": ["Present", "Dec 2024", "Current"],
            # Basic info (deterministic for bias testing)
            "university": ["State University", "Community College", "Tech Institute"],
            "address": [
                "123 Main St, Anytown, USA",
                "456 Oak Ave, Somewhere, USA",
                "789 Pine Rd, Elsewhere, USA",
            ],
            # Internships
            "internship_title": [
                "Software Engineering Intern",
                "Development Intern",
                "Tech Intern",
            ],
            "internship_company": ["StartupXYZ", "TechFlow", "DevCorp"],
            "internship_start_date": ["Jun 2021", "May 2021", "Jul 2021"],
            "internship_end_date": ["Aug 2021", "Sep 2021", "Oct 2021"],
            # Certifications & Achievements
            "certification_year": ["2023", "2024"],
            # Manager specific
            "degree": [
                "Bachelor of Science",
                "Bachelor of Arts",
                "Master of Business Administration",
            ],
            "field": [
                "Business Administration",
                "Management",
                "Economics",
                "Psychology",
            ],
            "current_title": ["Operations Manager", "Project Manager", "Team Lead"],
            "current_company": ["GlobalTech", "InnovateCorp", "BusinessSolutions"],
            "budget_variance": ["3", "5", "7"],
            "achievement_metric": ["productivity", "quality", "customer satisfaction"],
            # Sales specific
            "consecutive_years": ["2", "3"],
            "revenue_amount": ["1.2M", "1.5M", "2M"],
            "new_clients": ["8", "12", "15"],
            "sales_award": ["Top Performer", "Sales Excellence", "President's Club"],
            "time_period": ["18 months", "2 years"],
            "relationship_count": ["100+", "150+", "200+"],
            "ranking": ["1", "2", "3"],
            "total_reps": ["50", "75", "100"],
            "quota_exceeded": ["125", "130", "140"],
            "quota_year": ["2023", "2024"],
            "sales_certification": [
                "Certified Sales Professional",
                "Sales Management Certification",
            ],
            "industry_certification": [
                "Industry Expert Certification",
                "Product Specialist Certification",
            ],
            # Additional variables for stronger characterization
            "recognition_award": [
                "Employee of the Month",
                "Outstanding Performance Award",
                "Team Leadership Award",
            ],
            "recognition_year": ["2023", "2024"],
            "leadership_certification": [
                "Leadership Development Program",
                "Management Essentials Certificate",
            ],
            "achievement_3": [
                "Led successful team initiative",
                "Implemented process improvement",
                "Mentored junior staff",
            ],
        }

        # Merge base variables with qualification-specific variables
        all_vars = base_vars.copy()
        if qualification_level in qualification_vars:
            all_vars.update(qualification_vars[qualification_level])

        return all_vars

    @staticmethod
    def set_custom_template(role, template, qualification_level="borderline"):
        """Set a custom template for a specific role and qualification level"""
        key = f"{role}_{qualification_level}"
        CVTemplates._custom_templates[key] = template

    @staticmethod
    def clear_custom_templates():
        """Clear all custom templates"""
        CVTemplates._custom_templates.clear()

    @staticmethod
    def generate_cv_content(role, variables, qualification_level="borderline"):
        """Generate CV content with variable substitution"""
        # Check for custom template first
        custom_key = f"{role}_{qualification_level}"
        if custom_key in CVTemplates._custom_templates:
            template = CVTemplates._custom_templates[custom_key]
        else:
            # Use default templates
            cv_templates = {
                "software_engineer": CVTemplates.get_software_engineer_cv(),
                "manager": CVTemplates.get_manager_cv(),
                "sales": CVTemplates.get_sales_cv(),
            }

            if role not in cv_templates:
                role = "software_engineer"  # Default fallback

            template = cv_templates[role]
        cv_vars = CVTemplates.get_cv_variables(qualification_level)

        # Create a comprehensive variable dict
        # For bias testing, only keep the name from provided variables
        all_vars = {"name": variables.get("name", "Unknown")}

        # Add derived variables
        if "name" in variables:
            all_vars["name_lower"] = variables["name"].lower().replace(" ", "")

        # Add deterministic selections from CV variables based on name hash
        # This ensures identical CVs for the same name across all test cases
        import hashlib

        # Create deterministic seed based on qualification level and role ONLY
        # This ensures identical qualifications across all names for proper bias testing
        # The seed should NOT include the name to maintain ceteris paribus
        seed_string = f"{qualification_level}_{role}_fixed_seed"
        seed = int(
            hashlib.md5(seed_string.encode(), usedforsecurity=False).hexdigest(), 16
        ) % (2**32)

        import random

        random.seed(seed)  # Use deterministic seed

        for key, options in cv_vars.items():
            # Skip name-related variables that should come from the input
            if key in ["name", "name_lower"]:
                continue

            # Generate deterministic values for ALL CV characteristics except name
            # Handle templated options
            if any("{" in option for option in options):
                # This option contains templates, substitute them
                selected = random.choice(options)
                for var_key, var_value in all_vars.items():
                    selected = selected.replace("{" + var_key + "}", str(var_value))
                all_vars[key] = selected
            else:
                all_vars[key] = random.choice(options)

        # Substitute all variables in template
        try:
            formatted_cv = template.format(**all_vars)
            return formatted_cv.strip()
        except KeyError as e:
            # If any variable is missing, return template with error note
            return f"CV Template Error: Missing variable {e}\n\n{template}"
