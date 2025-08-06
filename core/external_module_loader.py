"""
External Module Loader for OpenAudit Plugin Architecture

Enables dynamic loading of external analysis modules from files, packages,
and remote sources with security validation and dependency management.
"""

import hashlib
import importlib
import importlib.util
import inspect
import json
import logging
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from .base_analyzer import BaseAnalysisModule
from .module_registry import AnalysisModuleRegistry, get_global_registry
from .validators import validate_module_interface

logger = logging.getLogger(__name__)


@dataclass
class ModuleManifest:
    """Manifest describing an external module"""

    name: str
    version: str
    description: str
    author: str
    module_class: str
    file_path: str
    dependencies: List[str]
    permissions: List[str]
    checksum: Optional[str] = None
    verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "module_class": self.module_class,
            "file_path": self.file_path,
            "dependencies": self.dependencies,
            "permissions": self.permissions,
            "checksum": self.checksum,
            "verified": self.verified,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModuleManifest":
        """Create from dictionary"""
        return cls(**data)

    @classmethod
    def from_file(cls, manifest_path: Union[str, Path]) -> "ModuleManifest":
        """Load manifest from JSON file"""
        with open(manifest_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


class ModuleSecurityValidator:
    """Security validation for external modules"""

    SAFE_IMPORTS = {
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "matplotlib",
        "seaborn",
        "typing",
        "dataclasses",
        "datetime",
        "json",
        "pathlib",
        "logging",
        "collections",
        "itertools",
        "functools",
        "statistics",
        "math",
    }

    RESTRICTED_IMPORTS = {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "socket",
        "urllib",
        "requests",
        "pickle",
        "eval",
        "exec",
        "compile",
        "__import__",
    }

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def validate_source_code(self, source_code: str) -> Dict[str, Any]:
        """Validate source code for security issues"""
        self.warnings.clear()
        self.errors.clear()

        # Check for restricted imports
        self._check_imports(source_code)

        # Check for dangerous functions
        self._check_dangerous_functions(source_code)

        # Check for file system operations
        self._check_file_operations(source_code)

        # Check for network operations
        self._check_network_operations(source_code)

        return {
            "valid": len(self.errors) == 0,
            "warnings": self.warnings.copy(),
            "errors": self.errors.copy(),
            "strict_mode": self.strict_mode,
        }

    def _check_imports(self, source_code: str):
        """Check for restricted imports"""
        lines = source_code.split("\n")
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line.startswith("import ") or line.startswith("from "):
                # Extract module name
                if line.startswith("import "):
                    module = line.split("import ")[1].split()[0].split(".")[0]
                else:  # from X import Y
                    module = line.split("from ")[1].split(" import")[0].split(".")[0]

                if module in self.RESTRICTED_IMPORTS:
                    if self.strict_mode:
                        self.errors.append(f"Line {i}: Restricted import '{module}'")
                    else:
                        self.warnings.append(
                            f"Line {i}: Potentially unsafe import '{module}'"
                        )
                elif module not in self.SAFE_IMPORTS and not module.startswith("core."):
                    self.warnings.append(
                        f"Line {i}: Unknown import '{module}' - verify safety"
                    )

    def _check_dangerous_functions(self, source_code: str):
        """Check for dangerous function calls"""
        dangerous_functions = [
            "eval(",
            "exec(",
            "compile(",
            "__import__(",
            "getattr(",
            "setattr(",
        ]

        lines = source_code.split("\n")
        for i, line in enumerate(lines, 1):
            for func in dangerous_functions:
                if func in line:
                    self.errors.append(
                        f"Line {i}: Dangerous function call '{func.rstrip('(')}'"
                    )

    def _check_file_operations(self, source_code: str):
        """Check for file system operations"""
        file_ops = ["open(", "file(", "remove(", "rmdir(", "mkdir(", "chmod("]

        lines = source_code.split("\n")
        for i, line in enumerate(lines, 1):
            for op in file_ops:
                if op in line:
                    if self.strict_mode:
                        self.errors.append(
                            f"Line {i}: File operation '{op.rstrip('(')}' not allowed"
                        )
                    else:
                        self.warnings.append(
                            f"Line {i}: File operation '{op.rstrip('(')}' detected"
                        )

    def _check_network_operations(self, source_code: str):
        """Check for network operations"""
        network_ops = ["socket.", "urllib.", "requests.", "http.", "ftp."]

        lines = source_code.split("\n")
        for i, line in enumerate(lines, 1):
            for op in network_ops:
                if op in line:
                    self.errors.append(
                        f"Line {i}: Network operation '{op.rstrip('.')}' not allowed"
                    )


class ExternalModuleLoader:
    """Loader for external analysis modules"""

    def __init__(
        self,
        modules_directory: Union[str, Path] = "external_modules",
        registry: Optional[AnalysisModuleRegistry] = None,
        strict_security: bool = True,
    ):
        self.modules_directory = Path(modules_directory)
        self.registry = registry or get_global_registry()
        self.security_validator = ModuleSecurityValidator(strict_mode=strict_security)
        self.loaded_modules: Dict[str, Dict[str, Any]] = {}

        # Ensure modules directory exists
        self.modules_directory.mkdir(exist_ok=True)

        # Create manifests directory
        self.manifests_directory = self.modules_directory / "manifests"
        self.manifests_directory.mkdir(exist_ok=True)

    def load_module_from_file(
        self,
        module_path: Union[str, Path],
        module_name: Optional[str] = None,
        class_name: Optional[str] = None,
        validate_security: bool = True,
        auto_register: bool = True,
    ) -> Dict[str, Any]:
        """
        Load a module from a Python file

        Args:
            module_path: Path to the Python file containing the module
            module_name: Name to register the module as (defaults to filename)
            class_name: Name of the class implementing BaseAnalysisModule
            validate_security: Whether to perform security validation
            auto_register: Whether to automatically register with the global registry

        Returns:
            Dictionary with loading results and module information
        """
        module_path = Path(module_path)

        if not module_path.exists():
            return {"success": False, "error": f"Module file not found: {module_path}"}

        if not module_path.suffix == ".py":
            return {
                "success": False,
                "error": "Module file must be a Python (.py) file",
            }

        # Default module name from filename
        if not module_name:
            module_name = module_path.stem

        try:
            # Read source code
            with open(module_path, "r", encoding="utf-8") as f:
                source_code = f.read()

            # Security validation
            if validate_security:
                security_result = self.security_validator.validate_source_code(
                    source_code
                )
                if not security_result["valid"]:
                    return {
                        "success": False,
                        "error": "Security validation failed",
                        "security_issues": security_result["errors"],
                        "security_warnings": security_result["warnings"],
                    }

            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if not spec or not spec.loader:
                return {"success": False, "error": "Could not create module spec"}

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find the analysis module class
            module_class = self._find_analysis_module_class(module, class_name)
            if not module_class:
                return {
                    "success": False,
                    "error": f"No valid analysis module class found. Class must inherit from BaseAnalysisModule.",
                }

            # Validate the module interface
            validation_result = validate_module_interface(module_class)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": "Module interface validation failed",
                    "validation_errors": validation_result["errors"],
                }

            # Create module manifest
            manifest = self._create_manifest(
                module_name, module_path, module_class.__name__, source_code
            )

            # Store module information
            self.loaded_modules[module_name] = {
                "manifest": manifest,
                "module_class": module_class,
                "source_path": str(module_path),
                "loaded_at": None,  # Will be set when registered
            }

            # Auto-register if requested
            if auto_register:
                success = self.registry.register_module(module_name, module_class)
                if success:
                    self.loaded_modules[module_name][
                        "loaded_at"
                    ] = module_class.__name__
                    # Save manifest
                    self._save_manifest(manifest)
                else:
                    return {
                        "success": False,
                        "error": f"Failed to register module '{module_name}' with registry",
                    }

            return {
                "success": True,
                "module_name": module_name,
                "class_name": module_class.__name__,
                "manifest": manifest.to_dict(),
                "registered": auto_register,
                "security_warnings": (
                    security_result.get("warnings", []) if validate_security else []
                ),
            }

        except Exception as e:
            logger.exception(f"Error loading module from {module_path}")
            return {"success": False, "error": f"Error loading module: {str(e)}"}

    def load_module_from_manifest(
        self, manifest_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """Load a module using a manifest file"""
        try:
            manifest = ModuleManifest.from_file(manifest_path)

            # Verify file exists
            module_path = Path(manifest.file_path)
            if not module_path.is_absolute():
                module_path = self.modules_directory / module_path

            return self.load_module_from_file(
                module_path=module_path,
                module_name=manifest.name,
                class_name=manifest.module_class,
                validate_security=not manifest.verified,
            )

        except Exception as e:
            return {"success": False, "error": f"Error loading manifest: {str(e)}"}

    def load_all_modules(
        self, directory: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Load all modules from a directory"""
        if directory is None:
            directory = self.modules_directory
        else:
            directory = Path(directory)

        results = {"successful": [], "failed": [], "warnings": []}

        # Load from manifest files first
        for manifest_file in directory.glob("**/*.manifest.json"):
            result = self.load_module_from_manifest(manifest_file)
            if result["success"]:
                results["successful"].append(result)
            else:
                results["failed"].append(
                    {"file": str(manifest_file), "error": result["error"]}
                )

        # Load Python files without manifests
        for py_file in directory.glob("**/*.py"):
            # Skip if already loaded via manifest
            module_name = py_file.stem
            if module_name in self.loaded_modules:
                continue

            # Skip __init__.py and other special files
            if py_file.name.startswith("__"):
                continue

            result = self.load_module_from_file(py_file)
            if result["success"]:
                results["successful"].append(result)
                if result.get("security_warnings"):
                    results["warnings"].extend(result["security_warnings"])
            else:
                results["failed"].append(
                    {"file": str(py_file), "error": result["error"]}
                )

        return results

    def create_module_template(
        self,
        module_name: str,
        author: str = "Unknown",
        description: str = "Custom analysis module",
    ) -> Dict[str, Any]:
        """Create a template for a new external module"""

        # Create module file
        module_code = f'''"""
{description}

Author: {author}
Created for OpenAudit Modular Analysis System
"""

from typing import Dict, List, Any
import logging

from core.base_analyzer import BaseAnalysisModule, ModuleInfo, ModuleCategory, ModuleRequirements

logger = logging.getLogger(__name__)


class {module_name.replace('_', ' ').title().replace(' ', '')}Module(BaseAnalysisModule):
    """
    {description}

    This module implements custom bias analysis logic.
    Customize the analyze() method to implement your specific analysis.
    """

    def _create_module_info(self) -> ModuleInfo:
        """Create module information"""
        return ModuleInfo(
            name="{module_name}",
            version="1.0.0",
            description="{description}",
            author="{author}",
            category=ModuleCategory.CUSTOM,
            tags=["custom", "bias_analysis", "external"],
            requirements=ModuleRequirements(
                min_samples=5,
                min_groups=2,
                data_types=["responses", "dataframe"],
                dependencies=["pandas"],
                optional_dependencies=["numpy", "scipy"]
            )
        )

    def validate_input(self, data: Any) -> Dict[str, Any]:
        """
        Validate input data

        Args:
            data: Input data to validate (typically pandas DataFrame)

        Returns:
            Validation result with 'valid' boolean and 'errors' list
        """
        # Call parent validation first
        validation_result = super().validate_input(data)

        # Add your custom validation logic here
        # Example:
        # import pandas as pd
        # if not isinstance(data, pd.DataFrame):
        #     validation_result["errors"].append("Input must be a pandas DataFrame")
        #     validation_result["valid"] = False
        #
        # required_columns = ['model', 'response', 'demographic']
        # missing_columns = [col for col in required_columns if col not in data.columns]
        # if missing_columns:
        #     validation_result["errors"].append(f"Missing required columns: {{missing_columns}}")
        #     validation_result["valid"] = False

        return validation_result

    def analyze(self, data: Any, **kwargs) -> Dict[str, Any]:
        """
        Perform custom bias analysis

        Args:
            data: Input data (typically pandas DataFrame with LLM responses)
            **kwargs: Additional parameters (alpha, effect_size_threshold, etc.)

        Returns:
            Analysis results following the standard format
        """
        logger.info(f"Running {{self.__class__.__name__}} analysis")

        # Extract common parameters
        alpha = kwargs.get('alpha', 0.05)
        effect_size_threshold = kwargs.get('effect_size_threshold', 0.1)

        # TODO: Implement your custom analysis logic here
        # This is where you would analyze the data for bias

        # Example placeholder analysis
        total_responses = len(data) if hasattr(data, '__len__') else 0

        # Return results in standard format
        return {{
            "summary": {{
                "analysis_successful": True,
                "bias_detected": False,  # Update based on your analysis
                "total_responses_analyzed": total_responses,
                "custom_metric": 0.5  # Add your custom metrics
            }},
            "detailed_results": {{
                "parameters_used": {{
                    "alpha": alpha,
                    "effect_size_threshold": effect_size_threshold
                }},
                "custom_analysis_details": {{
                    # Add detailed results from your analysis
                    "placeholder_result": "Implement your analysis logic"
                }}
            }},
            "key_findings": [
                "Custom analysis completed",
                "TODO: Add meaningful findings from your analysis"
            ],
            "confidence_score": 0.7,  # Confidence in the analysis results (0-1)
            "recommendations": [
                "Review custom analysis implementation",
                "TODO: Add actionable recommendations based on findings"
            ],
            "metadata": {{
                "module": "{module_name}",
                "version": "1.0.0",
                "analysis_timestamp": str(data.__class__.__name__ if hasattr(data, '__class__') else 'unknown')
            }}
        }}
'''

        # Create manifest
        manifest_data = {
            "name": module_name,
            "version": "1.0.0",
            "description": description,
            "author": author,
            "module_class": f"{module_name.replace('_', ' ').title().replace(' ', '')}Module",
            "file_path": f"{module_name}.py",
            "dependencies": [],
            "permissions": ["data_analysis"],
            "verified": False,
        }

        # Save files
        module_file = self.modules_directory / f"{module_name}.py"
        manifest_file = self.modules_directory / f"{module_name}.manifest.json"

        try:
            with open(module_file, "w", encoding="utf-8") as f:
                f.write(module_code)

            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)

            return {
                "success": True,
                "module_file": str(module_file),
                "manifest_file": str(manifest_file),
                "message": f"Module template created successfully. Edit {module_file} to implement your analysis logic.",
            }

        except Exception as e:
            return {"success": False, "error": f"Error creating template: {str(e)}"}

    def list_loaded_modules(self) -> Dict[str, Dict[str, Any]]:
        """List all loaded external modules"""
        return {
            name: {
                "manifest": info["manifest"].to_dict(),
                "source_path": info["source_path"],
                "registered": info["loaded_at"] is not None,
            }
            for name, info in self.loaded_modules.items()
        }

    def unload_module(self, module_name: str) -> Dict[str, Any]:
        """Unload a module from the registry and loader"""
        if module_name not in self.loaded_modules:
            return {"success": False, "error": f"Module '{module_name}' not loaded"}

        # Remove from registry if registered
        # Note: Current registry doesn't support unregistration, this is a placeholder
        # In a full implementation, you'd add unregister functionality to the registry

        # Remove from loaded modules
        del self.loaded_modules[module_name]

        return {"success": True, "message": f"Module '{module_name}' unloaded"}

    def get_module_info(self, module_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a loaded module"""
        if module_name not in self.loaded_modules:
            return None

        info = self.loaded_modules[module_name]
        return {
            "manifest": info["manifest"].to_dict(),
            "source_path": info["source_path"],
            "registered": info["loaded_at"] is not None,
            "class_name": info["module_class"].__name__,
        }

    def load_module_from_github(
        self,
        github_url: str,
        module_name: Optional[str] = None,
        branch: str = "main",
        module_path: Optional[str] = None,
        validate_security: bool = True,
        auto_register: bool = True,
    ) -> Dict[str, Any]:
        """
        Load a module from a GitHub repository

        Args:
            github_url: GitHub repository URL (e.g., "https://github.com/user/repo")
            module_name: Name to register the module as
            branch: Git branch to download from
            module_path: Path to the module file within the repo (e.g., "modules/my_module.py")
            validate_security: Whether to perform security validation
            auto_register: Whether to automatically register with the global registry

        Returns:
            Dictionary with loading results
        """
        try:
            # Parse GitHub URL
            github_info = self._parse_github_url(github_url)
            if not github_info:
                return {"success": False, "error": "Invalid GitHub URL format"}

            owner, repo = github_info["owner"], github_info["repo"]

            # Download repository
            download_result = self._download_github_repo(owner, repo, branch)
            if not download_result["success"]:
                return download_result

            repo_path = Path(download_result["repo_path"])

            # Find module file
            if module_path:
                module_file = repo_path / module_path
                if not module_file.exists():
                    return {
                        "success": False,
                        "error": f"Module file not found: {module_path}",
                    }
            else:
                # Auto-discover Python files that might be analysis modules
                py_files = list(repo_path.glob("**/*.py"))
                module_file = None

                for py_file in py_files:
                    if py_file.name.startswith("__"):
                        continue

                    # Try to find a module with BaseAnalysisModule
                    if self._file_contains_analysis_module(py_file):
                        module_file = py_file
                        break

                if not module_file:
                    return {
                        "success": False,
                        "error": "No analysis module found. Specify module_path or ensure module inherits from BaseAnalysisModule",
                    }

            # Default module name from filename or repo name
            if not module_name:
                module_name = f"{repo}_{module_file.stem}"

            # Load the module
            result = self.load_module_from_file(
                module_path=module_file,
                module_name=module_name,
                validate_security=validate_security,
                auto_register=auto_register,
            )

            # Add GitHub metadata to result
            if result["success"]:
                result["github_info"] = {
                    "url": github_url,
                    "owner": owner,
                    "repo": repo,
                    "branch": branch,
                    "module_path": str(module_file.relative_to(repo_path)),
                }

            return result

        except Exception as e:
            logger.exception(f"Error loading module from GitHub: {github_url}")
            return {"success": False, "error": f"Error loading from GitHub: {str(e)}"}

    def install_module_from_github(
        self, github_url: str, module_name: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Install a module from GitHub permanently in the modules directory

        Args:
            github_url: GitHub repository URL
            module_name: Name for the installed module
            **kwargs: Additional arguments for load_module_from_github

        Returns:
            Installation result
        """
        # Load module from GitHub first
        load_result = self.load_module_from_github(
            github_url=github_url,
            module_name=module_name,
            auto_register=False,  # Don't register yet
            **kwargs,
        )

        if not load_result["success"]:
            return load_result

        try:
            # Copy module file to permanent location
            original_path = Path(load_result["github_info"]["module_path"])
            installed_name = module_name or load_result["module_name"]
            installed_path = self.modules_directory / f"{installed_name}.py"

            # Read and copy the source
            with open(original_path, "r", encoding="utf-8") as src:
                source_code = src.read()

            with open(installed_path, "w", encoding="utf-8") as dst:
                dst.write(source_code)

            # Create manifest with GitHub info
            manifest_data = {
                "name": installed_name,
                "version": "1.0.0",
                "description": f"Module from {github_url}",
                "author": load_result["github_info"]["owner"],
                "module_class": load_result["class_name"],
                "file_path": f"{installed_name}.py",
                "dependencies": [],
                "permissions": ["data_analysis"],
                "verified": False,
                "github_info": load_result["github_info"],
            }

            manifest_path = self.modules_directory / f"{installed_name}.manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)

            # Now load from the installed location
            final_result = self.load_module_from_file(
                module_path=installed_path,
                module_name=installed_name,
                validate_security=kwargs.get("validate_security", True),
                auto_register=True,
            )

            if final_result["success"]:
                final_result["installed_path"] = str(installed_path)
                final_result["manifest_path"] = str(manifest_path)
                final_result["github_info"] = load_result["github_info"]

            return final_result

        except Exception as e:
            logger.exception(f"Error installing module from GitHub")
            return {"success": False, "error": f"Installation failed: {str(e)}"}

    def _parse_github_url(self, url: str) -> Optional[Dict[str, str]]:
        """Parse GitHub URL to extract owner and repo"""
        import re

        # Support various GitHub URL formats
        patterns = [
            r"https://github\.com/([^/]+)/([^/]+)/?",
            r"git@github\.com:([^/]+)/([^/]+)\.git",
            r"([^/]+)/([^/]+)",  # Simple owner/repo format
        ]

        for pattern in patterns:
            match = re.match(pattern, url.strip())
            if match:
                owner, repo = match.groups()
                # Remove .git suffix if present
                repo = repo.replace(".git", "")
                return {"owner": owner, "repo": repo}

        return None

    def _download_github_repo(
        self, owner: str, repo: str, branch: str = "main"
    ) -> Dict[str, Any]:
        """Download GitHub repository as ZIP"""
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix="openaudit_github_")

            # Download URL
            download_url = f"https://github.com/{owner}/{repo}/archive/{branch}.zip"

            # Download the ZIP file
            zip_path = Path(temp_dir) / f"{repo}.zip"

            try:
                urllib.request.urlretrieve(download_url, zip_path)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to download repository: {str(e)}",
                }

            # Extract ZIP
            extract_path = Path(temp_dir) / "extracted"
            extract_path.mkdir()

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)

            # Find the extracted repository directory
            extracted_items = list(extract_path.iterdir())
            if len(extracted_items) != 1:
                return {"success": False, "error": "Unexpected archive structure"}

            repo_path = extracted_items[0]

            return {"success": True, "repo_path": str(repo_path), "temp_dir": temp_dir}

        except Exception as e:
            logger.exception(f"Error downloading GitHub repo: {owner}/{repo}")
            return {"success": False, "error": f"Download failed: {str(e)}"}

    def _file_contains_analysis_module(self, file_path: Path) -> bool:
        """Check if a Python file contains a class that inherits from BaseAnalysisModule"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Simple heuristic: check for BaseAnalysisModule inheritance
            return (
                "BaseAnalysisModule" in content
                and ("class " in content)
                and ("def analyze(" in content or "def analyze(self" in content)
            )
        except Exception:
            return False

    def _find_analysis_module_class(
        self, module, class_name: Optional[str] = None
    ) -> Optional[Type]:
        """Find a class that inherits from BaseAnalysisModule"""
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Skip if specific class name requested and this isn't it
            if class_name and name != class_name:
                continue

            # Check if it inherits from BaseAnalysisModule
            if (
                issubclass(obj, BaseAnalysisModule)
                and obj != BaseAnalysisModule
                and obj.__module__ == module.__name__
            ):
                return obj

        return None

    def _create_manifest(
        self, module_name: str, module_path: Path, class_name: str, source_code: str
    ) -> ModuleManifest:
        """Create a manifest for a loaded module"""

        # Calculate checksum
        checksum = hashlib.sha256(source_code.encode()).hexdigest()

        return ModuleManifest(
            name=module_name,
            version="1.0.0",  # Could be extracted from module docstring
            description=f"External module: {module_name}",
            author="Unknown",  # Could be extracted from module docstring
            module_class=class_name,
            file_path=str(module_path),
            dependencies=[],  # Could be analyzed from imports
            permissions=["data_analysis"],
            checksum=checksum,
            verified=False,
        )

    def _save_manifest(self, manifest: ModuleManifest):
        """Save a manifest to the manifests directory"""
        manifest_file = self.manifests_directory / f"{manifest.name}.manifest.json"

        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)


# Global instance for easy access
_global_loader: Optional[ExternalModuleLoader] = None


def get_global_module_loader() -> ExternalModuleLoader:
    """Get the global external module loader instance"""
    global _global_loader
    if _global_loader is None:
        _global_loader = ExternalModuleLoader()
    return _global_loader


def load_external_module(
    module_path: Union[str, Path], module_name: Optional[str] = None, **kwargs
) -> Dict[str, Any]:
    """Convenience function to load an external module"""
    loader = get_global_module_loader()
    return loader.load_module_from_file(module_path, module_name, **kwargs)


def create_module_template(module_name: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to create a module template"""
    loader = get_global_module_loader()
    return loader.create_module_template(module_name, **kwargs)


def list_external_modules() -> Dict[str, Dict[str, Any]]:
    """Convenience function to list loaded external modules"""
    loader = get_global_module_loader()
    return loader.list_loaded_modules()
