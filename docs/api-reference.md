# 🌐 API Reference

Complete reference for the OpenAudit REST API. Build integrations, automate audits, and access all platform functionality programmatically.

## 📋 **Overview**

The OpenAudit API provides RESTful endpoints for:
- 🧪 **Running bias audits**
- 📊 **Analyzing results** 
- 🧩 **Managing analysis modules**
- 📝 **Template management**
- 🔄 **Experiment tracking**

**Base URL**: `http://localhost:5000/api`

## 🔐 **Authentication**

Currently, OpenAudit runs locally without authentication. For production deployments, implement API key authentication:

```bash
# Set API key (if configured)
export OPENAUDIT_API_KEY="your-api-key"
```

```http
Authorization: Bearer your-api-key
```

## 📊 **Response Format**

All API responses follow a consistent format:

```json
{
  "status": "success|error",
  "data": {...},
  "timestamp": "2025-01-27T10:30:00Z",
  "version": "1.0.0"
}
```

### Error Responses
```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {...}
  },
  "timestamp": "2025-01-27T10:30:00Z"
}
```

## 🏃 **Quick Start**

### Basic Audit Request
```bash
curl -X POST http://localhost:5000/api/audit \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "audit_type": "hiring",
    "demographics": ["race", "gender"],
    "iterations": 10
  }'
```

### Analyze Results
```bash
curl -X GET http://localhost:5000/api/analyze/run_20250127_143052?profile=basic
```

---

## 🔍 **Endpoints**

## 🧪 **Audit Endpoints**

### `POST /api/audit`
Create and run a new bias audit.

**Request Body:**
```json
{
  "model": "gpt-4",
  "audit_type": "hiring",
  "demographics": ["race", "gender"],
  "iterations": 20,
  "cv_level": "borderline",
  "role": "software_engineer",
  "profile": "standard",
  "custom_prompt": "optional custom prompt",
  "template_overrides": {
    "cv_template": "custom_template",
    "prompt_template": "custom_prompt"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "audit_id": "audit_20250127_143052",
    "run_id": "run_20250127_143052",
    "status": "running",
    "estimated_completion": "2025-01-27T10:35:00Z",
    "websocket_url": "ws://localhost:5000/audit/audit_20250127_143052"
  }
}
```

### `GET /api/audit/{audit_id}/status`
Get the status of a running audit.

**Response:**
```json
{
  "status": "success",
  "data": {
    "audit_id": "audit_20250127_143052",
    "status": "running|completed|failed",
    "progress": {
      "completed": 15,
      "total": 20,
      "percentage": 75.0
    },
    "current_step": "Analyzing responses",
    "estimated_completion": "2025-01-27T10:35:00Z",
    "results_available": false
  }
}
```

### `POST /api/audit/{audit_id}/cancel`
Cancel a running audit.

**Response:**
```json
{
  "status": "success",
  "data": {
    "audit_id": "audit_20250127_143052",
    "status": "cancelled",
    "message": "Audit cancelled successfully"
  }
}
```

---

## 📊 **Analysis Endpoints**

### `GET /api/analyze/{run_id}`
Analyze experiment results with modular analysis system.

**Query Parameters:**
- `profile` (optional): Analysis profile name
- `modules` (optional): Comma-separated list of module names
- `alpha` (optional): Significance level (default: 0.05)
- `effect_size_threshold` (optional): Effect size threshold (default: 0.3)
- `include_raw` (optional): Include raw results (default: false)

**Example:**
```bash
curl "http://localhost:5000/api/analyze/run_20250127_143052?profile=research_grade&include_raw=true"
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "run_id": "run_20250127_143052",
    "analysis_config": {
      "profile": "research_grade",
      "modules_used": ["enhanced_statistics", "cultural_context", "multi_level_classifier"],
      "parameters": {
        "alpha": 0.05,
        "effect_size_threshold": 0.3
      }
    },
    "unified_assessment": {
      "bias_detected": true,
      "overall_confidence": 0.87,
      "bias_severity": "moderate",
      "affected_demographics": ["race"]
    },
    "module_results": {
      "enhanced_statistics": {
        "summary": {
          "bias_detected": true,
          "p_value": 0.001,
          "effect_size": 0.42
        }
      }
    },
    "recommendations": [
      "Review model training data for racial bias",
      "Implement bias mitigation strategies"
    ],
    "metadata": {
      "analysis_timestamp": "2025-01-27T10:30:00Z",
      "analysis_duration": 45.2
    }
  }
}
```

### `POST /api/analyze/custom`
Run custom analysis with specific modules and parameters.

**Request Body:**
```json
{
  "run_id": "run_20250127_143052",
  "modules": ["enhanced_statistics", "cultural_context"],
  "parameters": {
    "alpha": 0.01,
    "effect_size_threshold": 0.2,
    "control_group": "white_male",
    "correction_method": "bonferroni"
  },
  "options": {
    "include_raw_results": true,
    "generate_visualizations": true,
    "export_format": "json"
  }
}
```

---

## 📈 **Runs & Data Endpoints**

### `GET /api/runs`
List all available experiment runs.

**Query Parameters:**
- `limit` (optional): Number of runs to return (default: 50)
- `offset` (optional): Offset for pagination (default: 0)
- `type` (optional): Filter by experiment type
- `model` (optional): Filter by model name
- `date_from` (optional): Filter by date range (ISO format)
- `date_to` (optional): Filter by date range (ISO format)

**Response:**
```json
{
  "status": "success",
  "data": {
    "runs": [
      {
        "run_id": "run_20250127_143052",
        "timestamp": "2025-01-27T14:30:52Z",
        "experiment_type": "Hiring Bias",
        "model": "gpt-4",
        "total_responses": 100,
        "demographics": ["race", "gender"],
        "status": "completed",
        "file_path": "runs/hiring_bias_experiment_20250127_143052.json",
        "file_size": 245760
      }
    ],
    "pagination": {
      "total": 25,
      "limit": 50,
      "offset": 0,
      "has_more": false
    }
  }
}
```

### `GET /api/runs/{run_id}`
Get detailed information about a specific run.

**Response:**
```json
{
  "status": "success",
  "data": {
    "run_id": "run_20250127_143052",
    "metadata": {
      "timestamp": "2025-01-27T14:30:52Z",
      "experiment_type": "Hiring Bias",
      "model": "gpt-4",
      "configuration": {
        "demographics": ["race", "gender"],
        "cv_level": "borderline",
        "iterations": 100
      }
    },
    "statistics": {
      "total_responses": 100,
      "response_rate": 1.0,
      "average_response_time": 2.3,
      "demographic_distribution": {
        "race": {"white": 25, "black": 25, "hispanic": 25, "asian": 25},
        "gender": {"male": 50, "female": 50}
      }
    },
    "analysis_available": true,
    "file_info": {
      "file_path": "runs/hiring_bias_experiment_20250127_143052.json",
      "file_size": 245760,
      "checksum": "sha256:abc123..."
    }
  }
}
```

### `GET /api/runs/{run_id}/raw`
Download raw experiment data.

**Query Parameters:**
- `format` (optional): Export format (json, csv, xlsx)

**Response:** File download or JSON data

### `DELETE /api/runs/{run_id}`
Delete an experiment run.

**Response:**
```json
{
  "status": "success",
  "data": {
    "run_id": "run_20250127_143052",
    "message": "Run deleted successfully"
  }
}
```

---

## 🧩 **Module Management Endpoints**

### `GET /api/modules`
List all available analysis modules.

**Response:**
```json
{
  "status": "success",
  "data": {
    "available_modules": [
      {
        "name": "enhanced_statistics",
        "info": {
          "name": "Enhanced Statistics",
          "version": "1.0.0",
          "description": "Advanced statistical testing with effect sizes",
          "author": "OpenAudit Team",
          "category": "statistical",
          "tags": ["statistics", "effect-size", "testing"]
        },
        "requirements": {
          "min_samples": 10,
          "min_groups": 2,
          "dependencies": ["numpy", "scipy", "statsmodels"]
        },
        "status": "active"
      }
    ],
    "total_modules": 5
  }
}
```

### `GET /api/modules/{module_name}`
Get detailed information about a specific module.

**Response:**
```json
{
  "status": "success",
  "data": {
    "module_info": {
      "name": "Enhanced Statistics",
      "version": "1.0.0",
      "description": "Advanced statistical testing with effect sizes",
      "author": "OpenAudit Team",
      "category": "statistical"
    },
    "requirements": {
      "min_samples": 10,
      "dependencies": ["numpy", "scipy"]
    },
    "supported_data_types": ["responses", "dataframe"],
    "configuration_options": {
      "alpha": {
        "type": "float",
        "default": 0.05,
        "min": 0.001,
        "max": 0.1
      }
    },
    "compatibility": {
      "compatible_modules": ["cultural_context", "multi_level_classifier"],
      "dependency_status": {
        "numpy": true,
        "scipy": true
      }
    }
  }
}
```

### `POST /api/modules/load`
Load an external analysis module.

**Request Body:**
```json
{
  "source_type": "file|github|url",
  "source_path": "path/to/module.py",
  "module_name": "my_custom_module",
  "validate_security": true,
  "auto_register": true
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "module_name": "my_custom_module",
    "loaded": true,
    "registered": true,
    "security_warnings": [],
    "module_info": {
      "name": "My Custom Module",
      "version": "1.0.0"
    }
  }
}
```

---

## 📋 **Profile Management Endpoints**

### `GET /api/profiles`
List all available analysis profiles.

**Response:**
```json
{
  "status": "success",
  "data": {
    "available_profiles": {
      "basic": {
        "name": "Basic Analysis",
        "description": "Quick statistical analysis",
        "modules": ["enhanced_statistics"],
        "use_case": "Quick bias check",
        "estimated_runtime": "< 1 minute",
        "min_samples_recommended": 10
      },
      "research_grade": {
        "name": "Research Grade Analysis",
        "description": "Comprehensive analysis for academic research",
        "modules": ["enhanced_statistics", "cultural_context", "multi_level_classifier", "goal_conflict", "human_ai_alignment"],
        "use_case": "Academic research",
        "estimated_runtime": "5-10 minutes",
        "min_samples_recommended": 50
      }
    },
    "total_profiles": 6
  }
}
```

### `GET /api/profiles/{profile_name}`
Get detailed information about a specific analysis profile.

**Response:**
```json
{
  "status": "success",
  "data": {
    "profile": {
      "name": "Research Grade Analysis",
      "description": "Comprehensive analysis for academic research",
      "category": "research",
      "modules": ["enhanced_statistics", "cultural_context", "multi_level_classifier"],
      "default_parameters": {
        "alpha": 0.05,
        "correction_method": "fdr_bh"
      }
    },
    "estimated_requirements": {
      "min_samples": 50,
      "estimated_runtime": "5-10 minutes",
      "memory_usage": "moderate"
    },
    "compatibility_check": {
      "all_modules_available": true,
      "missing_dependencies": []
    }
  }
}
```

### `POST /api/profiles`
Create a new custom analysis profile.

**Request Body:**
```json
{
  "name": "my_custom_profile",
  "description": "My custom analysis configuration",
  "modules": ["enhanced_statistics", "cultural_context"],
  "default_parameters": {
    "alpha": 0.01,
    "effect_size_threshold": 0.2
  },
  "category": "custom"
}
```

---

## 📝 **Template Management Endpoints**

### `GET /api/templates/cv`
List all CV templates.

**Response:**
```json
{
  "status": "success",
  "data": {
    "templates": [
      {
        "name": "Software Engineer - Strong",
        "role": "software_engineer",
        "level": "strong",
        "description": "Strong technical background",
        "variables": ["name", "university", "experience"],
        "created_at": "2025-01-27T10:00:00Z",
        "author": "OpenAudit"
      }
    ],
    "total_templates": 15
  }
}
```

### `GET /api/templates/cv/{template_name}`
Get a specific CV template.

**Response:**
```json
{
  "status": "success",
  "data": {
    "template": {
      "name": "Software Engineer - Strong",
      "role": "software_engineer",
      "level": "strong",
      "template": "CV template content with {variables}...",
      "variables": ["name", "university", "experience", "email"],
      "metadata": {
        "created_at": "2025-01-27T10:00:00Z",
        "author": "OpenAudit",
        "version": "1.0"
      }
    }
  }
}
```

### `POST /api/templates/cv`
Create a new CV template.

**Request Body:**
```json
{
  "name": "My Custom CV",
  "role": "data_scientist",
  "level": "strong",
  "description": "Custom CV template for data scientists",
  "template": "CV content with {name}, {university}, etc.",
  "variables": ["name", "university", "experience"]
}
```

### `GET /api/templates/prompts`
List all prompt templates.

### `POST /api/templates/prompts`
Create a new prompt template.

---

## 🤖 **Model Management Endpoints**

### `GET /api/models`
List all available AI models.

**Response:**
```json
{
  "status": "success",
  "data": {
    "all_models": [
      "gpt-4", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet", "gemini-pro"
    ],
    "models_by_category": {
      "openai": {
        "models": ["gpt-4", "gpt-3.5-turbo"],
        "status": "available",
        "api_key_configured": true
      },
      "anthropic": {
        "models": ["claude-3-opus", "claude-3-sonnet"],
        "status": "available",
        "api_key_configured": true
      }
    },
    "recommended_models": ["gpt-4", "claude-3-opus"],
    "total_models": 12
  }
}
```

### `GET /api/models/{model_name}/info`
Get detailed information about a specific model.

**Response:**
```json
{
  "status": "success",
  "data": {
    "model": "gpt-4",
    "provider": "openai",
    "capabilities": {
      "max_tokens": 4096,
      "supports_streaming": true,
      "supports_function_calling": true
    },
    "pricing": {
      "input_tokens": 0.03,
      "output_tokens": 0.06,
      "currency": "USD",
      "per_tokens": 1000
    },
    "status": {
      "available": true,
      "health_check": "passed",
      "last_checked": "2025-01-27T10:25:00Z"
    }
  }
}
```

### `POST /api/models/{model_name}/test`
Test connectivity to a specific model.

**Response:**
```json
{
  "status": "success",
  "data": {
    "model": "gpt-4",
    "test_result": "success",
    "response_time": 1.23,
    "test_prompt": "Hello, world!",
    "test_response": "Hello! How can I help you today?",
    "timestamp": "2025-01-27T10:30:00Z"
  }
}
```

---

## 📊 **Reports & Export Endpoints**

### `GET /api/reports/{run_id}`
Generate a comprehensive report for an experiment run.

**Query Parameters:**
- `format` (optional): Report format (html, pdf, json, markdown)
- `include_raw` (optional): Include raw data (default: false)
- `sections` (optional): Comma-separated list of report sections

**Response:**
```json
{
  "status": "success",
  "data": {
    "report_id": "report_20250127_143052",
    "format": "html",
    "content": "HTML report content...",
    "metadata": {
      "generated_at": "2025-01-27T10:30:00Z",
      "file_size": 157264,
      "sections_included": ["summary", "analysis", "recommendations"]
    },
    "download_url": "/api/reports/report_20250127_143052/download"
  }
}
```

### `GET /api/export/{run_id}`
Export experiment data in various formats.

**Query Parameters:**
- `format`: Export format (json, csv, xlsx, parquet)
- `include_analysis` (optional): Include analysis results (default: false)
- `demographic_breakdown` (optional): Include demographic breakdown (default: true)

**Response:** File download

---

## ⚡ **Real-time & WebSocket Endpoints**

### WebSocket: `/ws/audit/{audit_id}`
Real-time updates during audit execution.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:5000/ws/audit/audit_20250127_143052');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Audit update:', data);
};
```

**Message Types:**
```json
// Progress update
{
  "type": "progress",
  "data": {
    "completed": 15,
    "total": 20,
    "percentage": 75.0,
    "current_step": "Analyzing responses",
    "estimated_completion": "2025-01-27T10:35:00Z"
  }
}

// Response received
{
  "type": "response",
  "data": {
    "response_id": "resp_001",
    "demographic": {"race": "white", "gender": "male"},
    "decision": "hire",
    "timestamp": "2025-01-27T10:30:15Z"
  }
}

// Audit completed
{
  "type": "completed",
  "data": {
    "audit_id": "audit_20250127_143052",
    "run_id": "run_20250127_143052",
    "total_responses": 20,
    "success": true,
    "analysis_available": true
  }
}

// Error occurred
{
  "type": "error",
  "data": {
    "error": "API rate limit exceeded",
    "code": "RATE_LIMIT_ERROR",
    "retry_after": 60
  }
}
```

---

## 🔧 **System & Health Endpoints**

### `GET /api/health`
Check system health and status.

**Response:**
```json
{
  "status": "success",
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "uptime": 3600,
    "components": {
      "database": "healthy",
      "module_registry": "healthy",
      "external_apis": {
        "openai": "healthy",
        "anthropic": "healthy"
      }
    },
    "statistics": {
      "total_audits_run": 1247,
      "total_responses_analyzed": 124700,
      "active_modules": 5,
      "external_modules_loaded": 2
    },
    "system_info": {
      "python_version": "3.9.7",
      "memory_usage": "2.3 GB",
      "cpu_usage": "15%",
      "disk_usage": "45%"
    }
  }
}
```

### `GET /api/version`
Get OpenAudit version information.

**Response:**
```json
{
  "status": "success",
  "data": {
    "version": "1.0.0",
    "build": "20250127-143052",
    "commit": "abc123def456",
    "python_version": "3.9.7",
    "dependencies": {
      "numpy": "1.21.0",
      "pandas": "1.3.0",
      "scipy": "1.7.0"
    }
  }
}
```

### `GET /api/config`
Get current configuration settings.

**Response:**
```json
{
  "status": "success",
  "data": {
    "analysis": {
      "default_alpha": 0.05,
      "default_effect_size_threshold": 0.3,
      "default_profile": "standard"
    },
    "models": {
      "default_model": "gpt-4",
      "timeout": 30,
      "max_retries": 3
    },
    "features": {
      "external_modules_enabled": true,
      "github_loading_enabled": true,
      "security_validation_enabled": true
    }
  }
}
```

---

## 🚫 **Error Codes**

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `VALIDATION_ERROR` | Invalid input data | 400 |
| `NOT_FOUND` | Resource not found | 404 |
| `RATE_LIMIT_ERROR` | API rate limit exceeded | 429 |
| `MODEL_ERROR` | AI model API error | 502 |
| `ANALYSIS_ERROR` | Analysis computation failed | 500 |
| `MODULE_ERROR` | Analysis module error | 500 |
| `SECURITY_ERROR` | Security validation failed | 403 |
| `DEPENDENCY_ERROR` | Missing dependencies | 500 |

## 💡 **Best Practices**

### Rate Limiting
```python
import time
import requests

def make_request_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        response = requests.get(url)
        if response.status_code == 429:
            # Rate limited, wait and retry
            retry_after = int(response.headers.get('Retry-After', 60))
            time.sleep(retry_after)
            continue
        return response
    
    raise Exception("Max retries exceeded")
```

### Pagination
```python
def get_all_runs():
    all_runs = []
    offset = 0
    limit = 50
    
    while True:
        response = requests.get(f'/api/runs?limit={limit}&offset={offset}')
        data = response.json()['data']
        
        all_runs.extend(data['runs'])
        
        if not data['pagination']['has_more']:
            break
            
        offset += limit
    
    return all_runs
```

### WebSocket Handling
```javascript
class AuditMonitor {
    constructor(auditId) {
        this.auditId = auditId;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
    }
    
    connect() {
        this.ws = new WebSocket(`ws://localhost:5000/ws/audit/${this.auditId}`);
        
        this.ws.onopen = () => {
            console.log('Connected to audit monitor');
            this.reconnectAttempts = 0;
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };
        
        this.ws.onclose = () => {
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                setTimeout(() => {
                    this.reconnectAttempts++;
                    this.connect();
                }, 1000 * Math.pow(2, this.reconnectAttempts));
            }
        };
    }
    
    handleMessage(data) {
        switch (data.type) {
            case 'progress':
                this.updateProgress(data.data);
                break;
            case 'completed':
                this.onAuditCompleted(data.data);
                break;
            case 'error':
                this.onError(data.data);
                break;
        }
    }
}
```

## 🔗 **SDKs & Libraries**

### Python SDK
```python
from openaudit import OpenAuditClient

# Initialize client
client = OpenAuditClient(base_url="http://localhost:5000")

# Run audit
audit = client.create_audit(
    model="gpt-4",
    audit_type="hiring",
    demographics=["race", "gender"]
)

# Monitor progress
for update in audit.stream_progress():
    print(f"Progress: {update.percentage}%")

# Get results
results = audit.get_results()
analysis = client.analyze(results.run_id, profile="research_grade")
```

### JavaScript SDK
```javascript
import { OpenAuditClient } from 'openaudit-js';

const client = new OpenAuditClient('http://localhost:5000');

// Run audit
const audit = await client.createAudit({
    model: 'gpt-4',
    auditType: 'hiring',
    demographics: ['race', 'gender']
});

// Get results
const results = await client.getAnalysis(audit.runId, {
    profile: 'research_grade'
});
```

---

## 📞 **Support**

Need help with the API?

- 📚 **Documentation**: [Full docs](README.md)
- 🐛 **Issues**: [GitHub Issues](https://github.com/openaudit/openaudit/issues)
- 💬 **Discord**: [Join our community](https://discord.gg/openaudit)
- 📧 **Email**: api@openaudit.org

**Happy integrating! 🚀** 