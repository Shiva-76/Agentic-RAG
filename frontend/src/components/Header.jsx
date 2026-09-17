import './Header.css'

export default function Header({ onNewChat, traceOpen, onToggleTrace }) {
  return (
    <header className="app-header">
      <div className="header-left">
        <h1>Verity</h1>
        <span className="badge">Agentic RAG</span>
      </div>
      <div className="header-right">
        <button className="btn-secondary" onClick={onToggleTrace}>
          {traceOpen ? 'Hide Trace' : 'Show Trace'}
        </button>
        <button className="btn-primary" onClick={onNewChat}>
          New Chat
        </button>
      </div>
    </header>
  )
}
