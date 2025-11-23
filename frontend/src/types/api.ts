/**
 * API type definitions for AGI Platform
 */

// Health Check
export interface HealthCheck {
  status: string;
  version?: string;
  environment?: string;
}

// Generation Types
export interface GenerationTask {
  id: string;
  name: string;
  description: string;
  supported_by_huggingface: boolean;
  supported_by_lightweight: boolean;
}

export interface GenerationRequest {
  task: string;
  prompt: string;
  parameters?: Record<string, any>;
  use_lightweight?: boolean;
  use_distributed?: boolean;
}

export interface GenerationResponse {
  content: any;
  model_used: string;
  provider: string;
  latency: number;
  tokens_used?: number;
  metadata: Record<string, any>;
  safety_flagged: boolean;
  safety_reason?: string;
}

// Analysis Types
export interface AnalysisTask {
  id: string;
  name: string;
  description: string;
  supported_by_huggingface: boolean;
  supported_by_lightweight: boolean;
}

export interface AnalysisRequest {
  task: string;
  input_data: any;
  context?: Record<string, any>;
  parameters?: Record<string, any>;
  require_reasoning_chain?: boolean;
  use_lightweight?: boolean;
}

export interface AnalysisResponse {
  result: any;
  confidence: number;
  confidence_level: string;
  reasoning_chain?: string[];
  processing_time: number;
  model_used: string;
  metadata: Record<string, any>;
}

// Cognitive Types
export interface CognitiveObjective {
  id: string;
  name: string;
  description: string;
  capabilities: string[];
}

export interface CognitiveRequest {
  input: any;
  objectives: string[];
  context?: Record<string, any>;
  parameters?: Record<string, any>;
  use_lightweight?: boolean;
}

export interface CognitiveResponse {
  results: Record<string, any>;
  reasoning_trace: Array<{
    step: string;
    result: string;
    confidence?: number;
  }>;
  final_output: any;
  processing_sequence: string[];
  metadata: Record<string, any>;
}

// Model Information
export interface ModelInfo {
  model_id: string;
  name: string;
  type: string;
  size: string;
  status: string;
  capabilities: string[];
}

// Workflow Types
export interface WorkflowStep {
  step_id: string;
  task_type: string;
  parameters: Record<string, any>;
}

export interface WorkflowRequest {
  name: string;
  description?: string;
  steps: WorkflowStep[];
}

export interface WorkflowInfo {
  workflow_id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  created_at: string;
  steps_completed: number;
  total_steps: number;
}

// Auth Types
export interface User {
  username: string;
  email: string;
  disabled?: boolean;
}

export interface AuthToken {
  access_token: string;
  token_type: string;
}

// Distributed Inference Types
export interface DistributedConfig {
  enable_distributed: boolean;
  num_nodes: number;
  gpus_per_node: number;
  parallelism_strategy: string;
  use_nvrar: boolean;
}

export interface DistributedStats {
  communication_stats: Record<string, any>;
  performance_improvement?: number;
  recommendation: string;
}

// Business Types
export interface BusinessDomain {
  id: string;
  name: string;
}

export interface BusinessUseCase {
  name: string;
  description: string;
  models: string[];
  business_value: string;
  roi_impact: string;
}

export interface BusinessModel {
  model_id?: string;
  model_name: string;
  model_type: string;
  task: string;
  description: string;
  domains: string[];
  max_length: number;
  performance: {
    speed: string;
    accuracy: string;
  };
  business_impact: string;
}

export interface BusinessGenerationRequest {
  business_domain: string;
  use_case: string;
  prompt: string;
  business_context: {
    industry: string;
    use_case: string;
    tone: string;
    audience?: string;
  };
  parameters?: Record<string, any>;
}

export interface BusinessAnalysisRequest {
  business_domain: string;
  use_case: string;
  input_data: any;
  business_context: {
    industry: string;
    use_case: string;
  };
  parameters?: Record<string, any>;
  require_reasoning_chain?: boolean;
}
