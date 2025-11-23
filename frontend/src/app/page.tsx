'use client'

import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { apiClient } from '@/lib/api'
import { BarChart3, Brain, CheckCircle2, Eye, Loader2, Network, Shield, XCircle, Zap } from 'lucide-react'
import { useEffect, useState } from 'react'

export default function Home() {
  const [health, setHealth] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const data = await apiClient.healthCheck()
        setHealth(data)
      } catch (error) {
        console.error('Health check failed:', error)
      } finally {
        setLoading(false)
      }
    }
    fetchHealth()
  }, [])

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Hero Section */}
      <div className="text-center space-y-4 py-12">
        <h1 className="text-5xl font-bold text-white">
          Advanced AGI Platform
        </h1>
        <p className="text-xl text-gray-300 max-w-3xl mx-auto">
          Enterprise-grade multi-LLM orchestration with cognitive reasoning, universal analysis, and distributed inference.
        </p>
        <div className="flex items-center justify-center gap-4 mt-6">
          {loading ? (
            <Badge variant="secondary" className="px-4 py-2">
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Checking Status...
            </Badge>
          ) : health?.status === 'healthy' ? (
            <Badge variant="success" className="px-4 py-2">
              <CheckCircle2 className="h-4 w-4 mr-2" />
              System Operational
            </Badge>
          ) : (
            <Badge variant="destructive" className="px-4 py-2">
              <XCircle className="h-4 w-4 mr-2" />
              System Offline
            </Badge>
          )}
        </div>
      </div>

      {/* Feature Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <Card className="bg-black/40 border-blue-500/30 hover:border-blue-500/60 transition-all">
          <CardHeader>
            <Brain className="h-12 w-12 text-blue-400 mb-4" />
            <CardTitle className="text-white">Cognitive Reasoning</CardTitle>
            <CardDescription>
              Multi-objective analysis with chain-of-thought reasoning and evidence synthesis
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 text-sm text-gray-400">
              <div>• Multi-step reasoning chains</div>
              <div>• Evidence-based conclusions</div>
              <div>• Confidence scoring</div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/40 border-purple-500/30 hover:border-purple-500/60 transition-all">
          <CardHeader>
            <Eye className="h-12 w-12 text-purple-400 mb-4" />
            <CardTitle className="text-white">Universal Analysis</CardTitle>
            <CardDescription>
              10+ specialized analysis tasks from sentiment to style transfer
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 text-sm text-gray-400">
              <div>• Sentiment & emotion analysis</div>
              <div>• Entity & intent recognition</div>
              <div>• Summarization & Q&A</div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/40 border-green-500/30 hover:border-green-500/60 transition-all">
          <CardHeader>
            <Zap className="h-12 w-12 text-green-400 mb-4" />
            <CardTitle className="text-white">Performance Optimized</CardTitle>
            <CardDescription>
              3-5x speedup with parallel execution and smart caching
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 text-sm text-gray-400">
              <div>• Multi-model parallelization</div>
              <div>• Intelligent response caching</div>
              <div>• Latency-based routing</div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/40 border-yellow-500/30 hover:border-yellow-500/60 transition-all">
          <CardHeader>
            <Network className="h-12 w-12 text-yellow-400 mb-4" />
            <CardTitle className="text-white">Distributed Inference</CardTitle>
            <CardDescription>
              NVRAR all-reduce for multi-GPU parallel processing
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 text-sm text-gray-400">
              <div>• Multi-GPU coordination</div>
              <div>• Gradient synchronization</div>
              <div>• Scalable architecture</div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/40 border-red-500/30 hover:border-red-500/60 transition-all">
          <CardHeader>
            <Shield className="h-12 w-12 text-red-400 mb-4" />
            <CardTitle className="text-white">Enterprise Security</CardTitle>
            <CardDescription>
              Multi-layered security with content filtering and rate limiting
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 text-sm text-gray-400">
              <div>• JWT authentication</div>
              <div>• Content validation</div>
              <div>• Rate limiting & quotas</div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-black/40 border-cyan-500/30 hover:border-cyan-500/60 transition-all">
          <CardHeader>
            <BarChart3 className="h-12 w-12 text-cyan-400 mb-4" />
            <CardTitle className="text-white">Real-time Analytics</CardTitle>
            <CardDescription>
              Comprehensive monitoring with performance metrics and insights
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 text-sm text-gray-400">
              <div>• Request tracking</div>
              <div>• Performance profiling</div>
              <div>• Resource monitoring</div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Stats Section */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
        <Card className="bg-gradient-to-br from-blue-500/20 to-blue-600/10 border-blue-500/30">
          <CardHeader className="pb-2">
            <CardDescription className="text-gray-300">Analysis Tasks</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-white">10+</div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-purple-500/20 to-purple-600/10 border-purple-500/30">
          <CardHeader className="pb-2">
            <CardDescription className="text-gray-300">Speedup</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-white">3-5x</div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-green-500/20 to-green-600/10 border-green-500/30">
          <CardHeader className="pb-2">
            <CardDescription className="text-gray-300">LLM Support</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-white">Multi</div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-yellow-500/20 to-yellow-600/10 border-yellow-500/30">
          <CardHeader className="pb-2">
            <CardDescription className="text-gray-300">Uptime</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-white">99.9%</div>
          </CardContent>
        </Card>
      </div>

      {/* Research Highlights */}
      <Card className="bg-black/40 border-white/10">
        <CardHeader>
          <CardTitle className="text-white">Research Highlights</CardTitle>
          <CardDescription>
            Key innovations powering our AGI platform
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-3">
              <h3 className="text-lg font-semibold text-blue-400">Cognitive Architecture</h3>
              <p className="text-gray-400">
                Novel multi-objective reasoning system combining decomposition, parallel analysis, 
                and evidence synthesis for human-like problem solving.
              </p>
            </div>
            <div className="space-y-3">
              <h3 className="text-lg font-semibold text-purple-400">Universal Analysis Engine</h3>
              <p className="text-gray-400">
                Unified framework supporting 10+ analysis tasks with consistent interfaces and 
                automatic model selection based on task requirements.
              </p>
            </div>
            <div className="space-y-3">
              <h3 className="text-lg font-semibold text-green-400">Optimized Inference</h3>
              <p className="text-gray-400">
                Advanced parallelization with intelligent caching and latency-based routing 
                achieving 3-5x performance improvements.
              </p>
            </div>
            <div className="space-y-3">
              <h3 className="text-lg font-semibold text-yellow-400">Distributed Processing</h3>
              <p className="text-gray-400">
                NVRAR-based all-reduce implementation enabling seamless multi-GPU coordination 
                for large-scale inference workloads.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
