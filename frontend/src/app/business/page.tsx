'use client'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { apiClient } from '@/lib/api'
import { Building, Loader2, TrendingUp } from 'lucide-react'
import { useEffect, useState } from 'react'

export default function BusinessPage() {
  const [domains, setDomains] = useState<any[]>([])
  const [selectedDomain, setSelectedDomain] = useState<string>('')
  const [useCases, setUseCases] = useState<any[]>([])
  const [models, setModels] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadBusinessData()
  }, [])

  useEffect(() => {
    if (selectedDomain) {
      loadDomainData(selectedDomain)
    }
  }, [selectedDomain])

  const loadBusinessData = async () => {
    try {
      const domainsData = await apiClient.getBusinessDomains()
      setDomains(domainsData.data || [])
      if (domainsData.data && domainsData.data.length > 0) {
        setSelectedDomain(domainsData.data[0].id)
      }
    } catch (error) {
      console.error('Failed to load business data:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadDomainData = async (domain: string) => {
    try {
      const [useCasesData, modelsData] = await Promise.all([
        apiClient.getDomainUseCases(domain),
        apiClient.getDomainModels(domain)
      ])
      setUseCases(useCasesData.data || [])
      setModels(modelsData.data || [])
    } catch (error) {
      console.error(`Failed to load domain data for ${domain}:`, error)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Loader2 className="h-8 w-8 animate-spin text-blue-400" />
      </div>
    )
  }

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="text-center space-y-4">
        <div className="flex items-center justify-center gap-3">
          <Building className="h-10 w-10 text-blue-400" />
          <h1 className="text-4xl font-bold text-white">Business Solutions</h1>
        </div>
        <p className="text-xl text-gray-300">Industry-specific AI solutions and use cases</p>
      </div>

      {/* Domain Selector */}
      <Card className="bg-black/40 border-white/10">
        <CardHeader>
          <CardTitle className="text-white">Select Business Domain</CardTitle>
          <CardDescription>Choose your industry to see relevant AI solutions</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {domains.map((domain) => (
              <Button
                key={domain.id}
                variant={selectedDomain === domain.id ? 'default' : 'outline'}
                onClick={() => setSelectedDomain(domain.id)}
                className="h-auto py-3"
              >
                {domain.name}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Use Cases */}
      <div>
        <h2 className="text-2xl font-bold text-white mb-4">Business Use Cases</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {useCases.map((useCase, index) => (
            <Card key={index} className="bg-black/40 border-blue-500/30 hover:border-blue-500/60 transition-all">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <TrendingUp className="h-5 w-5 text-green-400" />
                  {useCase.name}
                </CardTitle>
                <CardDescription>{useCase.description}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div>
                  <h4 className="font-semibold text-sm text-gray-300 mb-2">Business Value</h4>
                  <p className="text-sm text-gray-400">{useCase.business_value}</p>
                </div>
                <div>
                  <h4 className="font-semibold text-sm text-gray-300 mb-2">ROI Impact</h4>
                  <p className="text-sm text-green-400">{useCase.roi_impact}</p>
                </div>
                <div>
                  <h4 className="font-semibold text-sm text-gray-300 mb-2">AI Models Used</h4>
                  <div className="flex flex-wrap gap-1">
                    {useCase.models?.map((model: string, i: number) => (
                      <Badge key={i} variant="secondary" className="text-xs">
                        {model}
                      </Badge>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* Models */}
      <div>
        <h2 className="text-2xl font-bold text-white mb-4">Specialized AI Models ({models.length})</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {models.map((model, index) => (
            <Card key={index} className="bg-black/40 border-purple-500/30">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg text-white">
                  {model.model_name?.split('/').pop() || 'Model'}
                </CardTitle>
                <CardDescription>{model.description}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Speed:</span>
                  <Badge
                    variant={
                      model.performance?.speed === 'fast' ? 'success' :
                      model.performance?.speed === 'medium' ? 'secondary' : 'outline'
                    }
                    className="text-xs"
                  >
                    {model.performance?.speed}
                  </Badge>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Accuracy:</span>
                  <Badge
                    variant={
                      model.performance?.accuracy === 'very_high' || model.performance?.accuracy === 'high' 
                        ? 'success' : 'secondary'
                    }
                    className="text-xs"
                  >
                    {model.performance?.accuracy}
                  </Badge>
                </div>
                <div className="pt-2">
                  <span className="text-xs font-semibold text-gray-300">Business Impact:</span>
                  <p className="text-xs text-gray-400 mt-1">{model.business_impact}</p>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  )
}
