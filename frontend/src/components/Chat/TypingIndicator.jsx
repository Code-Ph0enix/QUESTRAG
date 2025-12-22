import { motion } from 'framer-motion'
import { BsRobot } from 'react-icons/bs'
import { cn } from '../../lib/utils'

const TypingIndicator = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="flex items-start gap-3"
    >
      {/* Avatar */}
      <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-gradient-to-br from-violet-600 to-cyan-600 flex items-center justify-center shadow-lg">
        <BsRobot className="w-4 h-4 text-white" />
      </div>

      {/* Typing Bubble */}
      <div className="flex flex-col gap-1">
        <span className="text-xs text-muted-foreground px-1">QuestRAG AI</span>
        <div className="flex items-center gap-1 px-4 py-3 rounded-2xl rounded-tl-md bg-accent border border-border/50">
          <span className="w-2 h-2 bg-violet-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
          <span className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
          <span className="w-2 h-2 bg-cyan-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
        </div>
        <span className="text-xs text-muted-foreground px-1">
          Thinking...
        </span>
      </div>
    </motion.div>
  )
}

export default TypingIndicator
