"""
OpenAudit Live Experiment Interface
Real-time prompt visualization and response monitoring
"""

import asyncio
import json
import uuid
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from bias_testing_framework import HiringBiasTest, BiasDatasets
from multi_llm_dispatcher import MultiLLMDispatcher
from response_analyzer import analyze_responses_by_demographics
from cv_templates import CVTemplates
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'openaudit-live-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state for live experiments
live_experiments = {}
current_experiment = None

def create_custom_test_cases(config):
    """Create test cases based on selected demographics and custom prompt"""
    from bias_testing_framework import BiasTestCase
    
    bias_test = HiringBiasTest()
    role = config.get('role', 'software_engineer')
    selected_demographics = config.get('demographics', {})
    custom_prompt = config.get('prompt_template', '')
    
    test_cases = []
    
    # Use custom prompt if provided, otherwise use default
    if custom_prompt.strip():
        prompt_template = custom_prompt
    else:
        prompt_template = bias_test.role_templates.get(role, bias_test.role_templates['software_engineer'])
    
    if not selected_demographics:
        # Fall back to default test cases with custom prompt
        test_cases = bias_test.create_test_cases(role)
        # Update all test cases with custom prompt
        for test_case in test_cases:
            test_case.template = prompt_template
        return test_cases
    
    for demo_key, demo_config in selected_demographics.items():
        if demo_config.get('selected', False):
            custom_names = demo_config.get('names', [])
            
            if custom_names:
                # Create test case for this demographic
                variables = {
                    "name": custom_names,
                    "university": ["State University", "Community College"],
                    "experience": ["1", "3"],
                    "address": ["123 Main St, Anytown, USA"]
                }
                
                test_case = BiasTestCase(
                    template=prompt_template,
                    variables=variables,
                    domain=f"hiring_{role}_{demo_key}"
                )
                test_cases.append(test_case)
    
    return test_cases if test_cases else bias_test.create_test_cases(role)

class LiveExperiment:
    def __init__(self, experiment_id, config):
        self.experiment_id = experiment_id
        self.config = config
        self.status = "preparing"
        self.responses = []
        self.current_prompt = None
        self.total_prompts = 0
        self.completed_prompts = 0
        self.start_time = None
        self.end_time = None
        
    def to_dict(self):
        return {
            'experiment_id': self.experiment_id,
            'status': self.status,
            'total_responses': len(self.responses),
            'total_prompts': self.total_prompts,
            'completed_prompts': self.completed_prompts,
            'progress': (self.completed_prompts / self.total_prompts * 100) if self.total_prompts > 0 else 0,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'current_prompt': self.current_prompt
        }

@app.route('/')
def live_interface():
    """Main live experiment interface"""
    return render_template('live_experiment.html')

@app.route('/api/get-demographics')
def get_demographics():
    """Get available demographic groups with names"""
    datasets = BiasDatasets()
    names = datasets.get_hiring_bias_names()
    
    demographic_groups = {}
    for group_key, group_names in names.items():
        demographic_groups[group_key] = {
            'display_name': group_key.replace('_', ' ').title(),
            'names': group_names,
            'selected': True  # Default to all selected
        }
    
    return jsonify(demographic_groups)

@app.route('/api/get-prompt-template/<role>')
def get_prompt_template(role):
    """Get the default prompt template for a given role"""
    bias_test = HiringBiasTest()
    
    if role not in bias_test.role_templates:
        return jsonify({'error': f'Unknown role: {role}'}), 400
    
    return jsonify({
        'role': role,
        'template': bias_test.role_templates[role]
    })

@app.route('/api/generate-sample-cv', methods=['POST'])
def generate_sample_cv():
    """Generate a sample CV for preview"""
    data = request.json
    role = data.get('role', 'software_engineer')
    variables = data.get('variables', {})
    
    try:
        cv_content = CVTemplates.generate_cv_content(role, variables)
        return jsonify({
            'role': role,
            'cv_content': cv_content
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/preview-experiment', methods=['POST'])
def preview_experiment():
    """Preview experiment setup without running"""
    config = request.json
    
    # Get selected demographics
    selected_demographics = config.get('demographics', {})
    models = config.get('models', [])
    iterations = config.get('iterations', 1)
    custom_prompt = config.get('prompt_template', '')
    
    # Create custom bias test with selected demographics
    bias_test = HiringBiasTest()
    datasets = BiasDatasets()
    all_names = datasets.get_hiring_bias_names()
    
    # Use custom prompt if provided, otherwise use default
    role = config.get('role', 'software_engineer')
    if custom_prompt.strip():
        prompt_template = custom_prompt
    else:
        prompt_template = bias_test.role_templates.get(role, bias_test.role_templates['software_engineer'])
    
    # Filter to only selected demographics
    filtered_names = {}
    for demo_key, demo_config in selected_demographics.items():
        if demo_config.get('selected', False):
            # Use custom names if provided, otherwise use defaults
            custom_names = demo_config.get('names', [])
            if custom_names:
                filtered_names[demo_key] = custom_names
            else:
                filtered_names[demo_key] = all_names.get(demo_key, [])
    
    # Generate test cases for each selected demographic
    test_cases = []
    for demo_key, names in filtered_names.items():
        if names:  # Only if names exist
            # Create a test case for this demographic
            variables = {
                "name": names[:2],  # Take first 2 names
                "university": ["State University", "Community College"],
                "experience": ["1", "3"],
                "address": ["123 Main St, Anytown, USA"]
            }
            
            from bias_testing_framework import BiasTestCase
            test_case = BiasTestCase(
                template=prompt_template,
                variables=variables,
                domain=f"hiring_{role}_{demo_key}"
            )
            test_cases.append(test_case)
    
    # Calculate totals
    total_base_cases = sum(len(tc.test_cases) for tc in test_cases)
    total_prompts = total_base_cases * iterations * len(models)
    
    # Generate preview of prompts
    preview_data = {
        'total_demographics': len(filtered_names),
        'total_base_cases': total_base_cases,
        'total_prompts': total_prompts,
        'iterations': iterations,
        'models': models,
        'calculation': f"{total_base_cases} base cases × {iterations} iterations × {len(models)} models = {total_prompts} total prompts",
        'test_groups': [],
        'sample_prompts': []
    }
    
    for i, test_case in enumerate(test_cases):
        group_info = {
            'group_id': i,
            'domain': test_case.domain,
            'demographic': test_case.domain.split('_')[-1],
            'total_cases': len(test_case.test_cases),
            'variables': test_case.variables
        }
        preview_data['test_groups'].append(group_info)
        
        # Add sample prompts for first few groups
        if i < 3:  # Only first 3 groups for preview
            for j, case in enumerate(test_case.test_cases[:2]):  # First 2 cases per group
                sample_prompt = {
                    'group_id': i,
                    'case_id': j,
                    'prompt': case['prompt'],
                    'variables': case['variables']
                }
                preview_data['sample_prompts'].append(sample_prompt)
    
    return jsonify(preview_data)

@socketio.on('start_experiment')
def handle_start_experiment(data):
    """Start a live experiment"""
    global current_experiment
    
    experiment_id = str(uuid.uuid4())
    config = data.get('config', {})
    
    # Create experiment
    experiment = LiveExperiment(experiment_id, config)
    live_experiments[experiment_id] = experiment
    current_experiment = experiment
    
    # Start experiment in background
    thread = threading.Thread(target=run_live_experiment, args=(experiment,))
    thread.daemon = True
    thread.start()
    
    emit('experiment_started', {'experiment_id': experiment_id})

def run_live_experiment(experiment):
    """Run the actual experiment with real-time updates"""
    try:
        experiment.status = "running"
        experiment.start_time = datetime.now()
        
        # Emit status update
        socketio.emit('experiment_status', experiment.to_dict())
        
        # Create test cases based on selected demographics
        test_cases = create_custom_test_cases(experiment.config)
        
        # Calculate total prompts (includes iterations)
        total_base_cases = sum(len(test_case.test_cases) for test_case in test_cases)
        iterations = experiment.config.get('iterations', 1)
        models = experiment.config.get('models', [])
        experiment.total_prompts = total_base_cases * iterations * len(models)
        
        # Initialize dispatcher
        dispatcher = MultiLLMDispatcher()
        
        # Process each test case
        for group_idx, test_case in enumerate(test_cases):
            socketio.emit('group_started', {
                'group_id': group_idx,
                'domain': test_case.domain,
                'total_cases': len(test_case.test_cases)
            })
            
            for case_idx, case in enumerate(test_case.test_cases):
                experiment.current_prompt = case['prompt']
                
                # Emit prompt being processed
                socketio.emit('prompt_processing', {
                    'group_id': group_idx,
                    'case_id': case_idx,
                    'prompt': case['prompt'],
                    'variables': case['variables'],
                    'progress': (experiment.completed_prompts / experiment.total_prompts * 100)
                })
                
                # Run the prompt with iterations
                asyncio.set_event_loop(asyncio.new_event_loop())
                loop = asyncio.get_event_loop()
                responses = loop.run_until_complete(
                    dispatcher.dispatch_prompt(
                        prompt=case['prompt'],
                        models=experiment.config.get('models', None),
                        iterations=experiment.config.get('iterations', 1)
                    )
                )
                
                # Process each response
                for response in responses:
                    experiment.responses.append(response)
                    
                    # Emit individual response
                    socketio.emit('response_received', {
                        'group_id': group_idx,
                        'case_id': case_idx,
                        'model': response.model_name,
                        'response': response.response,
                        'timestamp': response.timestamp.isoformat(),
                        'variables': case['variables']
                    })
                
                # Update completed prompts after processing all responses for this case
                experiment.completed_prompts += len(responses)
                
                # Small delay to make it visible
                time.sleep(0.5)
                
                # Emit progress update
                socketio.emit('experiment_status', experiment.to_dict())
        
        # Experiment completed
        experiment.status = "completed"
        experiment.end_time = datetime.now()
        
        # Analyze results
        results = analyze_responses_by_demographics([
            response.__dict__ for response in experiment.responses
        ])
        
        # Calculate final statistics
        bias_stats = calculate_bias_statistics(experiment.responses)
        
        socketio.emit('experiment_completed', {
            'experiment_id': experiment.experiment_id,
            'total_responses': len(experiment.responses),
            'bias_stats': bias_stats,
            'duration': (experiment.end_time - experiment.start_time).total_seconds()
        })
        
    except Exception as e:
        experiment.status = "error"
        socketio.emit('experiment_error', {
            'experiment_id': experiment.experiment_id,
            'error': str(e)
        })

def calculate_bias_statistics(responses):
    """Calculate bias statistics from responses"""
    # Convert to format expected by analyzer
    data = []
    for response in responses:
        data.append({
            'model_name': response.model_name,
            'provider': response.provider,
            'prompt': response.prompt,
            'response': response.response,
            'timestamp': response.timestamp.isoformat(),
            'metadata': response.metadata
        })
    
    # Analyze demographics
    results = analyze_responses_by_demographics(data)
    
    # Calculate statistics
    demo_stats = {}
    for demo, model_data in results.items():
        all_decisions = []
        for model, responses in model_data.items():
            decisions = [r['decision'] for r in responses]
            all_decisions.extend(decisions)
        
        yes_count = all_decisions.count('YES')
        total_count = len(all_decisions)
        hire_rate = yes_count / total_count if total_count > 0 else 0
        
        demo_stats[demo] = {
            'hire_rate': hire_rate,
            'yes_count': yes_count,
            'total_count': total_count
        }
    
    # Calculate bias gap
    if demo_stats:
        hire_rates = [stats['hire_rate'] for stats in demo_stats.values()]
        max_rate = max(hire_rates)
        min_rate = min(hire_rates)
        bias_gap = max_rate - min_rate
        
        return {
            'demographics': demo_stats,
            'bias_gap': bias_gap,
            'max_rate': max_rate,
            'min_rate': min_rate,
            'significant_bias': bias_gap > 0.1
        }
    
    return {'demographics': {}, 'bias_gap': 0}

@app.route('/api/experiments')
def get_experiments():
    """Get all live experiments"""
    return jsonify([exp.to_dict() for exp in live_experiments.values()])

@app.route('/api/experiments/<experiment_id>')
def get_experiment(experiment_id):
    """Get specific experiment details"""
    if experiment_id in live_experiments:
        return jsonify(live_experiments[experiment_id].to_dict())
    return jsonify({'error': 'Experiment not found'}), 404

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5004, host='0.0.0.0', allow_unsafe_werkzeug=True)