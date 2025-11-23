/**
 * API type definitions
 */

export interface CognitiveRequest {
  prompt: string;
  task_type: string;
  model_id?: string;
  parameters?: Record<string, any>;
}

export interface CognitiveResponse {
  result: string;
  model_used: string;
  task_type: string;
  metadata: Record<string, any>;
}

export interface ModelInfo {
  model_id: string;
  name: string;
  type: string;
  size: string;
  status: string;
  capabilities: string[];
}

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

export interface User {
  username: string;
  email: string;
  disabled?: boolean;
}

export interface AuthToken {
  access_token: string;
  token_type: string;
}
