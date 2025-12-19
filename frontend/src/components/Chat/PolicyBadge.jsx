import { motion } from 'framer-motion'
import { HiDatabase, HiLightningBolt } from 'react-icons/hi'

const PolicyBadge = ({ action, confidence }) => {
  const isFetch = action === 'FETCH'
  
  return (
    <motion.div
      initial={{ scale: 0, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ delay: 0.2, type: 'spring', damping: 15 }}
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${
        isFetch
          ? 'bg-emerald-900/30 text-emerald-400 border-emerald-500/30'
          : 'bg-amber-900/30 text-amber-400 border-amber-500/30'
      }`}
    >
      {isFetch ? (
        <HiDatabase className="w-3.5 h-3.5" />
      ) : (
        <HiLightningBolt className="w-3.5 h-3.5" />
      )}
      <span>{isFetch ? 'RAG' : 'Direct'}</span>
      {confidence && (
        <span className="ml-0.5 opacity-70">
          {(confidence * 100).toFixed(0)}%
        </span>
      )}
    </motion.div>
  )
}

export default PolicyBadge
