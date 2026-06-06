import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  /** Friendly section name shown in the fallback UI. */
  label: string;
  /** Children to render under the boundary. */
  children: ReactNode;
  /** Optional reset key — when this value changes, the boundary clears
   *  its error state and re-attempts to render. Pass e.g. the page's
   *  active id so navigating to a different entry recovers. */
  resetKey?: string | number | null;
}

interface State {
  error: Error | null;
}

/** Catches render-time exceptions from a subtree and shows a fallback
 *  instead of blowing up the entire React app. We use this around
 *  pages that load remote data (InvestigationPage, future replay
 *  flows) where a malformed payload could throw deep in render.
 */
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidUpdate(prev: Props) {
    if (prev.resetKey !== this.props.resetKey && this.state.error) {
      this.setState({ error: null });
    }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Log to console for devs. Production telemetry would hook here.
    console.error(`[ErrorBoundary:${this.props.label}]`, error, info);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="error-boundary">
        <h3>This {this.props.label} couldn't be rendered.</h3>
        <p className="muted">
          The data for this entry may be corrupted. Try selecting a
          different entry from the sidebar or starting a new one.
        </p>
        <pre className="error-detail">{this.state.error.message}</pre>
      </div>
    );
  }
}
