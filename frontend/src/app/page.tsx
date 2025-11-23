export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm">
        <h1 className="text-4xl font-bold mb-8">AGI Platform</h1>
        <p className="text-lg mb-4">
          Universal AI platform with security-first architecture
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
          <div className="border rounded-lg p-6">
            <h2 className="text-2xl font-semibold mb-2">Cognitive Engine</h2>
            <p>Unified interface for generation, analysis, and reasoning</p>
          </div>
          <div className="border rounded-lg p-6">
            <h2 className="text-2xl font-semibold mb-2">Model Management</h2>
            <p>Secure model deployment and orchestration</p>
          </div>
          <div className="border rounded-lg p-6">
            <h2 className="text-2xl font-semibold mb-2">Workflows</h2>
            <p>Build and execute complex AI workflows</p>
          </div>
        </div>
      </div>
    </main>
  )
}
