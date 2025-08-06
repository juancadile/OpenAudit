#!/usr/bin/env python
"""
OpenAudit Setup Configuration
Modular LLM Bias Auditing Platform
"""

import sys
from pathlib import Path

from setuptools import find_packages, setup

# Ensure we're using Python 3.8+
if sys.version_info < (3, 8):
    print("ERROR: OpenAudit requires Python 3.8 or higher")
    sys.exit(1)

# Get the long description from README
here = Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")


# Read version from __init__.py
def get_version():
    version_file = here / "core" / "__init__.py"
    if version_file.exists():
        with open(version_file, "r") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"


# Read requirements
def get_requirements(filename="requirements.txt"):
    requirements_file = here / filename
    if requirements_file.exists():
        with open(requirements_file, "r") as f:
            return [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
    return []


def get_optional_requirements():
    """Get optional requirements for extras"""
    return {
        "dev": get_requirements("requirements-dev.txt"),
        "all": (
            get_requirements("requirements.txt")
            + get_requirements("requirements-dev.txt")
        ),
    }


# Package discovery
setup(
    # Basic information
    name="openaudit",
    version=get_version(),
    description="Modular LLM bias auditing platform for comprehensive fairness evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # Author and contact information
    author="OpenAudit Team",
    author_email="team@openaudit.org",
    maintainer="OpenAudit Team",
    maintainer_email="team@openaudit.org",
    # URLs
    url="https://github.com/openaudit/openaudit",
    project_urls={
        "Documentation": "https://docs.openaudit.org",
        "Source": "https://github.com/openaudit/openaudit",
        "Tracker": "https://github.com/openaudit/openaudit/issues",
        "Changelog": "https://github.com/openaudit/openaudit/blob/main/CHANGELOG.md",
        "Discord": "https://discord.gg/openaudit",
    },
    # License
    license="MIT",
    # Package discovery - explicitly define packages
    packages=["openaudit"]
    + [
        "openaudit." + pkg
        for pkg in find_packages(exclude=["tests", "tests.*", "docs", "docs.*"])
    ],
    package_dir={"openaudit": "."},
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md"],
        "openaudit.templates": ["**/*.yaml", "**/*.yml", "**/*.html"],
        "openaudit.core": ["*.yaml", "*.yml", "*.json"],
    },
    include_package_data=True,
    # Dependencies
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require=get_optional_requirements(),
    # Entry points
    entry_points={
        "console_scripts": [
            "openaudit=openaudit.cli:main",
            "openaudit-server=openaudit.unified_interface:main",
        ],
        "openaudit.modules": [
            "enhanced_statistics=openaudit.core.enhanced_statistics_module:EnhancedStatisticsModule",
            "cultural_context=openaudit.core.cultural_context_module:CulturalContextModule",
            "multi_level_classifier=openaudit.core.multi_level_classifier_module:MultiLevelClassifierModule",
            "goal_conflict=openaudit.core.goal_conflict_analyzer:GoalConflictAnalyzer",
            "human_ai_alignment=openaudit.core.human_ai_alignment_analyzer:HumanAIAlignmentModule",
        ],
    },
    # Classification
    classifiers=[
        # Development status
        "Development Status :: 4 - Beta",
        # Intended audience
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        # Topic
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Testing",
        # License
        "License :: OSI Approved :: MIT License",
        # Python versions
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        # Operating systems
        "Operating System :: OS Independent",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        # Environment
        "Environment :: Console",
        "Environment :: Web Environment",
        # Framework
        "Framework :: Flask",
        # Natural language
        "Natural Language :: English",
    ],
    # Keywords for PyPI search
    keywords=[
        "ai",
        "bias",
        "audit",
        "fairness",
        "llm",
        "ml",
        "ethics",
        "testing",
        "evaluation",
        "research",
        "modular",
        "openai",
        "anthropic",
        "gpt",
        "claude",
        "gemini",
    ],
    # Platform support
    platforms=["any"],
    # Zip safe
    zip_safe=False,
    # Additional metadata for newer setuptools
)

# Post-installation message
print(
    """
🎉 OpenAudit installed successfully!

Next steps:
1. Set your API keys:
   export OPENAI_API_KEY="your-key"
   export ANTHROPIC_API_KEY="your-key"

2. Start the web interface:
   openaudit start

3. Or run a quick audit:
   openaudit audit hiring --model gpt-4

📚 Documentation: https://docs.openaudit.org
💬 Community: https://discord.gg/openaudit
🐛 Issues: https://github.com/openaudit/openaudit/issues

Happy auditing! 🚀
"""
)
