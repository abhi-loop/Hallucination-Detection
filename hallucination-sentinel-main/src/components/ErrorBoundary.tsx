/**
 * src/components/ErrorBoundary.tsx
 *
 * Catches any React render crash and shows a recovery UI instead of a
 * blank page. Wraps the root App so nothing can escape it.
 */
import { Component, type ReactNode, type ErrorInfo } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface Props {
    children: ReactNode;
}

interface State {
    hasError: boolean;
    message: string;
}

class ErrorBoundary extends Component<Props, State> {
    constructor(props: Props) {
        super(props);
        this.state = { hasError: false, message: '' };
    }

    static getDerivedStateFromError(error: unknown): State {
        const message =
            error instanceof Error ? error.message : String(error);
        return { hasError: true, message };
    }

    componentDidCatch(error: unknown, info: ErrorInfo) {
        console.error('[ErrorBoundary] Caught error:', error, info.componentStack);
    }

    handleReset = () => {
        this.setState({ hasError: false, message: '' });
    };

    render() {
        if (this.state.hasError) {
            return (
                <div className="h-screen flex items-center justify-center bg-background p-8">
                    <div className="rounded-xl border border-destructive/40 bg-destructive/5 p-8 max-w-lg w-full space-y-4 text-center">
                        <AlertTriangle className="w-10 h-10 text-destructive mx-auto" />
                        <h2 className="text-base font-mono font-bold text-foreground">
                            Something went wrong
                        </h2>
                        <p className="text-xs font-mono text-muted-foreground break-words">
                            {this.state.message || 'An unexpected error occurred.'}
                        </p>
                        <button
                            onClick={this.handleReset}
                            className="inline-flex items-center gap-2 rounded-md bg-primary text-primary-foreground px-4 py-2 text-sm font-mono hover:opacity-90 transition-opacity"
                        >
                            <RefreshCw className="w-4 h-4" />
                            Try again
                        </button>
                    </div>
                </div>
            );
        }

        return this.props.children;
    }
}

export default ErrorBoundary;
