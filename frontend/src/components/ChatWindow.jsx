import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble'
import './ChatWindow.css'

export default function ChatWindow({ messages }) {
  const endRef = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div className="chat-window">
      {messages.length === 0 ? (
        <div className="chat-empty">
          <h2>How can I help you?</h2>
          <p>Ask a question about your documents, the web, or perform calculations.</p>
        </div>
      ) : (
        <div className="chat-messages">
          {messages.map((msg, i) => (
            <MessageBubble key={msg.id || i} message={msg} />
          ))}
          <div ref={endRef} />
        </div>
      )}
    </div>
  )
}
