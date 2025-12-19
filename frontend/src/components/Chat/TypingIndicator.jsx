import { motion } from 'framer-motion'
import { BsRobot } from 'react-icons/bs'

const TypingIndicator = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="flex items-start gap-3"
    >
      {/* Avatar */}
      <div className="flex-shrink-0 w-9 h-9 rounded-xl bg-gradient-to-br from-gray-700 to-gray-800 border border-gray-600 flex items-center justify-center shadow-lg">
        <BsRobot className="w-5 h-5 text-blue-400" />
      </div>

      {/* Typing Bubble */}
      <div className="flex flex-col gap-1">
        <span className="text-xs text-gray-500 px-1">QuestRAG AI</span>
        <div className="flex items-center gap-1 px-4 py-3 rounded-2xl rounded-tl-md bg-gray-800 border border-gray-700">
          {[0, 1, 2].map((index) => (
            <motion.div
              key={index}
              className="w-2 h-2 bg-blue-500 rounded-full"
              animate={{
                y: [0, -6, 0],
                opacity: [0.5, 1, 0.5]
              }}
              transition={{
                duration: 0.6,
                repeat: Infinity,
                delay: index * 0.15,
                ease: "easeInOut"
              }}
            />
          ))}
        </div>
        <span className="text-xs text-gray-600 px-1">
          Thinking...
        </span>
      </div>
    </motion.div>
  )
}

export default TypingIndicator
