import { useState, useRef, useCallback } from 'react'
import ChatWindow from './components/ChatWindow'
import InputBar from './components/InputBar'
import NodeTracePanel from './components/NodeTracePanel'
import Header from './components/Header'
import './App.css'

const SESSION_ID = `session_${Date.now()}`

export default function App() {
  const [messages, setMessages] = useState([])
  const [nodeEvents, setNodeEvents] = useState([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [traceOpen, setTraceOpen] = useState(false)
  const abortRef = useRef(null)

  const sendMessage = useCallback(async (query) => {
    if (!query.trim() || isStreaming) return

    // Add user message
    const userMsg = { role: 'user', content: query, id: Date.now() }
    setMessages(prev => [...prev, userMsg])
    setNodeEvents([])
    setTraceOpen(true)
    setIsStreaming(true)

    // Placeholder AI message
    const aiId = Date.now() + 1
    setMessages(prev => [...prev, { role: 'assistant', content: '', id: aiId, loading: true }])

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, session_id: SESSION_ID }),
        signal: abortRef.current?.signal,
      })

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const parsed = JSON.parse(line.slice(6))

            if (parsed.event === 'node_update') {
              setNodeEvents(prev => [...prev, parsed.data])
            } else if (parsed.event === 'final_answer') {
              setMessages(prev => prev.map(m =>
                m.id === aiId
                  ? { ...m, content: parsed.data.answer, loading: false,
                      meta: { rewritten: parsed.data.rewritten_query, retries: parsed.data.retry_count, sources: parsed.data.sources_used } }
                  : m
              ))
            } else if (parsed.event === 'error') {
              setMessages(prev => prev.map(m =>
                m.id === aiId ? { ...m, content: `⚠️ ${parsed.data.message}`, loading: false, error: true } : m
              ))
            }
          } catch { /* skip malformed */ }
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        setMessages(prev => prev.map(m =>
          m.id === aiId ? { ...m, content: '⚠️ Connection error. Is the backend running?', loading: false, error: true } : m
        ))
      }
    } finally {
      setIsStreaming(false)
    }
  }, [isStreaming])

  const newChat = () => {
    fetch(`/history/${SESSION_ID}`, { method: 'DELETE' }).catch(() => {})
    setMessages([])
    setNodeEvents([])
    setTraceOpen(false)
  }

  return (
    <div className="app">
      <Header onNewChat={newChat} traceOpen={traceOpen} onToggleTrace={() => setTraceOpen(p => !p)} />
      <div className="app-body">
        <ChatWindow messages={messages} />
        {traceOpen && <NodeTracePanel events={nodeEvents} isStreaming={isStreaming} />}
      </div>
      <InputBar onSend={sendMessage} disabled={isStreaming} />
    </div>
  )
}
