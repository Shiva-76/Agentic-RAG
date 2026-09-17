import { useEffect, useRef } from 'react'
import './NodeTracePanel.css'

export default function NodeTracePanel({ events, isStreaming }) {
  const panelRef = useRef(null)

  useEffect(() => {
    // Auto-scroll to bottom of trace as new events come in
    if (panelRef.current) {
      panelRef.current.scrollTop = panelRef.current.scrollHeight
    }
  }, [events])

  if (events.length === 0 && !isStreaming) return null

  return (
    <div className="trace-panel">
      <div className="trace-header">
        <h3>Agent Trace</h3>
        {isStreaming && <span className="pulsing-dot"></span>}
      </div>
      <div className="trace-content" ref={panelRef}>
        {events.length === 0 && isStreaming && (
          <div className="trace-wait">Waiting for pipeline...</div>
        )}
        {events.map((evt, idx) => (
          <NodeEventItem key={idx} event={evt} />
        ))}
      </div>
    </div>
  )
}

function NodeEventItem({ event }) {
  const { node } = event
  
  // Format the node payload nicely
  const formatPayload = (data) => {
    const copy = { ...data }
    delete copy.node
    return JSON.stringify(copy, null, 2)
  }

  // Determine badge color based on node type
  let badgeClass = 'badge-default'
  if (node === 'retrieve') badgeClass = 'badge-blue'
  else if (node === 'synthesize') badgeClass = 'badge-purple'
  else if (node === 'relevance_guard') {
    badgeClass = event.pass_ ? 'badge-green' : 'badge-red'
  }

  return (
    <div className="trace-item fade-in-up">
      <div className="trace-node-name">
        <span className={`node-badge ${badgeClass}`}>{node || 'system'}</span>
      </div>
      <pre className="trace-payload">{formatPayload(event)}</pre>
    </div>
  )
}
