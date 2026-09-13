import { Component, type ReactNode } from 'react'
import { AlertOctagon, RotateCw } from 'lucide-react'

interface Props {
  children: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('[Unfold ErrorBoundary]', error, info.componentStack)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex min-h-screen flex-col items-center justify-center bg-void px-6 text-center">
          <div className="mb-6 rounded-3xl border border-red-500/30 bg-red-500/10 p-5">
            <AlertOctagon className="h-12 w-12 text-red-400" />
          </div>
          <h1 className="text-2xl font-bold text-white">Something went wrong</h1>
          <p className="mt-2 max-w-md text-sm text-slate-400">
            Unfold hit an unexpected error and the UI couldn't recover.
            You can reload the page to start fresh.
          </p>
          {this.state.error && (
            <pre className="mt-4 max-w-lg overflow-auto rounded-xl border border-edge bg-panel/80 p-4 text-left text-xs text-slate-500">
              {this.state.error.message}
            </pre>
          )}
          <button
            onClick={() => window.location.reload()}
            className="btn-primary mt-8"
          >
            <RotateCw className="h-4 w-4" />
            Reload the page
          </button>
        </div>
      )
    }

    return this.props.children
  }
}
