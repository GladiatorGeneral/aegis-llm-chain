'use client'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { apiClient } from '@/lib/api'
import { formatLatency } from '@/lib/utils'
import { AlertCircle, CheckCircle2, Copy, Loader2, Zap } from 'lucide-react'
import { useState } from 'react'

export default function GenerationPage() {
  const [prompt, setPrompt] = useState('')
  const [maxTokens, setMaxTokens] = useState(512)
  const [temperature, setTemperature] = useState(0.7)
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      setError('Please enter a prompt')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = await apiClient.generateText({
        prompt: prompt,
        max_tokens: maxTokens,
        temperature: temperature,
        model: 'microsoft/Phi-3.5-mini-instruct',
      })
      setResult(data)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Generation failed')
    } finally {
      setLoading(false)
    }
  }

  const copyToClipboard = () => {
    if (result?.generated_text) {
      navigator.clipboard.writeText(result.generated_text)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="text-center space-y-2">
        <div className="flex items-center justify-center gap-3">
          <Zap className="h-10 w-10 text-green-400" />
          <h1 className="text-4xl font-bold text-white">Text Generation</h1>
        </div>
        <p className="text-gray-300">High-performance LLM text generation</p>
      </div>

      <div className="grid md:grid-cols-3 gap-6">
        <Card className="md:col-span-2 bg-black/40 border-white/10">
          <CardHeader>
            <CardTitle className="text-white">Prompt</CardTitle>
            <CardDescription>Enter your generation prompt</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Textarea
              placeholder="Enter your prompt... (e.g., 'Write a short story about AI and humanity')"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="min-h-[200px] bg-black/20 border-white/20 text-white"
            />

            {error && (
              <div className="flex items-center gap-2 text-red-400 text-sm">
                <AlertCircle className="h-4 w-4" />
                {error}
              </div>
            )}

            <Button
              onClick={handleGenerate}
              disabled={loading}
              className="w-full bg-green-600 hover:bg-green-700"
            >
              {loading ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Zap className="h-4 w-4 mr-2" />
                  Generate Text
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        <Card className="md:col-span-1 bg-black/40 border-white/10">
          <CardHeader>
            <CardTitle className="text-white">Parameters</CardTitle>
            <CardDescription>Configure generation</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-300">
                Max Tokens: {maxTokens}
              </label>
              <input
                type="range"
                min="50"
                max="2048"
                step="50"
                value={maxTokens}
                onChange={(e) => setMaxTokens(Number(e.target.value))}
                className="w-full"
              />
              <p className="text-xs text-gray-500">Maximum length of generated text</p>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-300">
                Temperature: {temperature.toFixed(2)}
              </label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                className="w-full"
              />
              <p className="text-xs text-gray-500">Controls randomness (0=deterministic, 2=very random)</p>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-300">Model</label>
              <Input
                value="microsoft/Phi-3.5-mini-instruct"
                disabled
                className="bg-black/20 border-white/20 text-gray-400"
              />
            </div>
          </CardContent>
        </Card>
      </div>

      {result && (
        <Card className="bg-black/40 border-green-500/30 animate-slide-up">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-white">Generated Text</CardTitle>
                <CardDescription>AI-generated content</CardDescription>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={copyToClipboard}
                className="gap-2"
              >
                {copied ? (
                  <>
                    <CheckCircle2 className="h-4 w-4 text-green-400" />
                    Copied!
                  </>
                ) : (
                  <>
                    <Copy className="h-4 w-4" />
                    Copy
                  </>
                )}
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="p-4 rounded-lg bg-black/30 border border-white/10">
              <p className="text-gray-200 whitespace-pre-wrap leading-relaxed">
                {result.generated_text}
              </p>
            </div>

            <div className="flex flex-wrap gap-3">
              {result.tokens_generated && (
                <Badge variant="secondary">
                  Tokens: {result.tokens_generated}
                </Badge>
              )}
              {result.latency_ms && (
                <Badge variant="secondary">
                  Time: {formatLatency(result.latency_ms)}
                </Badge>
              )}
              {result.tokens_generated && result.latency_ms && (
                <Badge variant="success">
                  Speed: {(result.tokens_generated / (result.latency_ms / 1000)).toFixed(1)} tokens/s
                </Badge>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
