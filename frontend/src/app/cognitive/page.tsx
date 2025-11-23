'use client'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { apiClient } from '@/lib/api'
import { formatConfidence, formatLatency } from '@/lib/utils'
import { AlertCircle, ArrowRight, Brain, CheckCircle2, Loader2 } from 'lucide-react'
import { useState } from 'react'

interface CognitiveResponse {
  reasoning_trace: Array<{
    step: number
    thought: string
    confidence: number
  }>
  conclusion: string
  evidence: string[]
  confidence_score: number
  processing_time_ms: number
}

export default function CognitivePage() {
  const [input, setInput] = useState('')
  const [objectives, setObjectives] = useState<string[]>(['analyze_sentiment', 'extract_entities'])
  const [response, setResponse] = useState<CognitiveResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const availableObjectives = [
    { id: 'analyze_sentiment', label: 'Sentiment Analysis', color: 'blue' },
    { id: 'extract_entities', label: 'Entity Extraction', color: 'purple' },
    { id: 'summarize', label: 'Summarization', color: 'green' },
    { id: 'classify_intent', label: 'Intent Classification', color: 'yellow' },
    { id: 'detect_emotion', label: 'Emotion Detection', color: 'red' },
    { id: 'answer_question', label: 'Question Answering', color: 'cyan' },
  ]

  const toggleObjective = (id: string) => {
    if (objectives.includes(id)) {
      setObjectives(objectives.filter(o => o !== id))
    } else {
      setObjectives([...objectives, id])
    }
  }

  const handleProcess = async () => {
    if (!input.trim()) {
      setError('Please enter text to analyze')
      return
    }

    if (objectives.length === 0) {
      setError('Please select at least one objective')
      return
    }

    setLoading(true)
    setError(null)
    setResponse(null)

    try {
      const data = await apiClient.processCognitive({
        text: input,
        objectives: objectives,
        model: 'microsoft/Phi-3.5-mini-instruct',
      })
      setResponse(data)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to process cognitive task')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="text-center space-y-2">
        <div className="flex items-center justify-center gap-3">
          <Brain className="h-10 w-10 text-blue-400" />
          <h1 className="text-4xl font-bold text-white">Cognitive Reasoning</h1>
        </div>
        <p className="text-gray-300">
          Multi-objective analysis with chain-of-thought reasoning
        </p>
      </div>

      {/* Input Section */}
      <Card className="bg-black/40 border-white/10">
        <CardHeader>
          <CardTitle className="text-white">Problem Statement</CardTitle>
          <CardDescription>
            Enter text to analyze using cognitive reasoning
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Textarea
            placeholder="Enter text to analyze... (e.g., 'The product exceeded my expectations. The customer service was fantastic and delivery was quick.')"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="min-h-[120px] bg-black/20 border-white/20 text-white"
          />

          <div className="space-y-3">
            <label className="text-sm font-medium text-gray-300">
              Analysis Objectives (select multiple)
            </label>
            <div className="flex flex-wrap gap-2">
              {availableObjectives.map((obj) => (
                <Badge
                  key={obj.id}
                  variant={objectives.includes(obj.id) ? 'default' : 'outline'}
                  className={`cursor-pointer transition-all ${
                    objectives.includes(obj.id)
                      ? `bg-${obj.color}-500 hover:bg-${obj.color}-600`
                      : 'hover:bg-white/10'
                  }`}
                  onClick={() => toggleObjective(obj.id)}
                >
                  {obj.label}
                </Badge>
              ))}
            </div>
          </div>

          {error && (
            <div className="flex items-center gap-2 text-red-400 text-sm">
              <AlertCircle className="h-4 w-4" />
              {error}
            </div>
          )}

          <Button
            onClick={handleProcess}
            disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-700"
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <Brain className="h-4 w-4 mr-2" />
                Process with Cognitive Engine
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Results Section */}
      {response && (
        <div className="space-y-6 animate-slide-up">
          {/* Reasoning Trace */}
          <Card className="bg-black/40 border-blue-500/30">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5 text-green-400" />
                Reasoning Trace
              </CardTitle>
              <CardDescription>
                Step-by-step cognitive reasoning process
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {response.reasoning_trace.map((step, idx) => (
                  <div
                    key={idx}
                    className="flex gap-4 p-4 rounded-lg bg-black/30 border border-white/10"
                  >
                    <div className="flex-shrink-0">
                      <div className="flex items-center justify-center w-8 h-8 rounded-full bg-blue-500/20 border border-blue-500/50">
                        <span className="text-sm font-bold text-blue-400">{step.step}</span>
                      </div>
                    </div>
                    <div className="flex-1 space-y-2">
                      <p className="text-gray-200">{step.thought}</p>
                      <div className="flex items-center gap-2">
                        <Badge variant="secondary" className="text-xs">
                          Confidence: {formatConfidence(step.confidence)}
                        </Badge>
                      </div>
                    </div>
                    {idx < response.reasoning_trace.length - 1 && (
                      <ArrowRight className="h-5 w-5 text-gray-500 self-center" />
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Conclusion */}
          <Card className="bg-black/40 border-green-500/30">
            <CardHeader>
              <CardTitle className="text-white">Conclusion</CardTitle>
              <CardDescription>
                Final cognitive analysis result
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-lg text-gray-200 mb-4">{response.conclusion}</p>
              <div className="flex items-center gap-4">
                <Badge variant="success" className="px-3 py-1">
                  Overall Confidence: {formatConfidence(response.confidence_score)}
                </Badge>
                <Badge variant="secondary" className="px-3 py-1">
                  Processing Time: {formatLatency(response.processing_time_ms)}
                </Badge>
              </div>
            </CardContent>
          </Card>

          {/* Evidence */}
          {response.evidence && response.evidence.length > 0 && (
            <Card className="bg-black/40 border-purple-500/30">
              <CardHeader>
                <CardTitle className="text-white">Supporting Evidence</CardTitle>
                <CardDescription>
                  Key evidence supporting the conclusion
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {response.evidence.map((item, idx) => (
                    <li key={idx} className="flex items-start gap-2 text-gray-300">
                      <CheckCircle2 className="h-5 w-5 text-green-400 flex-shrink-0 mt-0.5" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          )}
        </div>
      )}
    </div>
  )
}
