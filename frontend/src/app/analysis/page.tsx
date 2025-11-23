'use client'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { apiClient } from '@/lib/api'
import { formatLatency } from '@/lib/utils'
import { AlertCircle, Eye, Loader2 } from 'lucide-react'
import { useState } from 'react'

export default function AnalysisPage() {
  const [input, setInput] = useState('')
  const [task, setTask] = useState('sentiment')
  const [results, setResults] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const analysisTasks = [
    { id: 'sentiment', label: 'Sentiment Analysis', description: 'Detect positive, negative, or neutral sentiment' },
    { id: 'emotion', label: 'Emotion Detection', description: 'Identify emotions like joy, anger, sadness' },
    { id: 'entities', label: 'Named Entity Recognition', description: 'Extract people, organizations, locations' },
    { id: 'intent', label: 'Intent Classification', description: 'Classify user intent or purpose' },
    { id: 'summarization', label: 'Text Summarization', description: 'Generate concise summary' },
    { id: 'keywords', label: 'Keyword Extraction', description: 'Extract important keywords' },
    { id: 'language', label: 'Language Detection', description: 'Identify text language' },
    { id: 'toxicity', label: 'Toxicity Detection', description: 'Detect harmful content' },
    { id: 'style_transfer', label: 'Style Transfer', description: 'Rewrite in different style' },
    { id: 'qa', label: 'Question Answering', description: 'Answer questions about text' },
  ]

  const handleAnalyze = async () => {
    if (!input.trim()) {
      setError('Please enter text to analyze')
      return
    }

    setLoading(true)
    setError(null)
    setResults(null)

    try {
      const data = await apiClient.analyzeContent({
        text: input,
        task: task,
        model: 'microsoft/Phi-3.5-mini-instruct',
      })
      setResults(data)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Analysis failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="text-center space-y-2">
        <div className="flex items-center justify-center gap-3">
          <Eye className="h-10 w-10 text-purple-400" />
          <h1 className="text-4xl font-bold text-white">Universal Analysis</h1>
        </div>
        <p className="text-gray-300">10+ specialized analysis tasks</p>
      </div>

      <div className="grid md:grid-cols-3 gap-6">
        <Card className="md:col-span-1 bg-black/40 border-white/10">
          <CardHeader>
            <CardTitle className="text-white">Analysis Task</CardTitle>
            <CardDescription>Select analysis type</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {analysisTasks.map((t) => (
                <button
                  key={t.id}
                  onClick={() => setTask(t.id)}
                  className={`w-full text-left p-3 rounded-lg transition-all ${
                    task === t.id
                      ? 'bg-purple-500/20 border border-purple-500/50'
                      : 'bg-black/20 border border-white/10 hover:bg-white/5'
                  }`}
                >
                  <div className="text-sm font-medium text-white">{t.label}</div>
                  <div className="text-xs text-gray-400 mt-1">{t.description}</div>
                </button>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="md:col-span-2 bg-black/40 border-white/10">
          <CardHeader>
            <CardTitle className="text-white">Input Text</CardTitle>
            <CardDescription>Enter text to analyze</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Textarea
              placeholder="Enter text to analyze..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              className="min-h-[200px] bg-black/20 border-white/20 text-white"
            />

            {error && (
              <div className="flex items-center gap-2 text-red-400 text-sm">
                <AlertCircle className="h-4 w-4" />
                {error}
              </div>
            )}

            <Button
              onClick={handleAnalyze}
              disabled={loading}
              className="w-full bg-purple-600 hover:bg-purple-700"
            >
              {loading ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Eye className="h-4 w-4 mr-2" />
                  Analyze Text
                </>
              )}
            </Button>
          </CardContent>
        </Card>
      </div>

      {results && (
        <Card className="bg-black/40 border-purple-500/30 animate-slide-up">
          <CardHeader>
            <CardTitle className="text-white">Analysis Results</CardTitle>
            <CardDescription>Task: {analysisTasks.find(t => t.id === task)?.label}</CardDescription>
          </CardHeader>
          <CardContent>
            <pre className="text-gray-200 whitespace-pre-wrap bg-black/30 p-4 rounded-lg border border-white/10 overflow-x-auto">
              {JSON.stringify(results, null, 2)}
            </pre>
            {results.latency_ms && (
              <div className="mt-4">
                <Badge variant="secondary">
                  Processing Time: {formatLatency(results.latency_ms)}
                </Badge>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  )
}
