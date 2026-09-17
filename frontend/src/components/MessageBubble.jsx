import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './MessageBubble.css'

export default function MessageBubble({ message }) {
  const isUser = message.role === 'user'

  return (
    <div className={`message-wrapper ${isUser ? 'user' : 'ai'}`}>
      {!isUser && <div className="avatar ai-avatar">AI</div>}
      <div className="message-content">
        {message.loading ? (
          <div className="typing-indicator">
            <span></span><span></span><span></span>
          </div>
        ) : (
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {message.content}
          </ReactMarkdown>
        )}
        
        {message.meta && message.meta.sources && message.meta.sources.length > 0 && (
          <div className="sources-list">
            <span className="sources-label">Sources:</span>
            {message.meta.sources.map(s => (
              <span key={s} className="source-tag">{s}</span>
            ))}
          </div>
        )}
      </div>
      {isUser && <div className="avatar user-avatar">U</div>}
    </div>
  )
}
