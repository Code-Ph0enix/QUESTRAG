import { motion } from 'framer-motion'

const Loader = ({ size = 'md', text, fullScreen = false }) => {
  const sizes = {
    sm: 'h-6 w-6 border-2',
    md: 'h-10 w-10 border-3',
    lg: 'h-16 w-16 border-4',
  }
  
  const spinnerVariants = {
    animate: {
      rotate: 360,
      transition: {
        duration: 0.8,
        repeat: Infinity,
        ease: "linear"
      }
    }
  }
  
  const pulseVariants = {
    animate: {
      scale: [1, 1.1, 1],
      opacity: [0.5, 0.8, 0.5],
      transition: {
        duration: 1.5,
        repeat: Infinity,
        ease: "easeInOut"
      }
    }
  }
  
  const LoaderContent = () => (
    <div className="flex flex-col items-center justify-center gap-5">
      <div className="relative">
        {/* Outer glow ring */}
        <motion.div
          className={`absolute inset-0 rounded-full bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 blur-lg`}
          variants={pulseVariants}
          animate="animate"
        />
        
        {/* Spinner */}
        <motion.div
          className={`relative ${sizes[size]} rounded-full bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600`}
          style={{
            background: 'conic-gradient(from 0deg, transparent 0deg 270deg, rgb(99 102 241) 270deg 360deg)',
            borderRadius: '50%'
          }}
          variants={spinnerVariants}
          animate="animate"
        />
      </div>
      
      {text && (
        <motion.p
          className="text-slate-600 dark:text-slate-400 text-sm font-medium"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          {text}
        </motion.p>
      )}
    </div>
  )
  
  if (fullScreen) {
    return (
      <motion.div 
        className="fixed inset-0 flex items-center justify-center bg-white/90 dark:bg-slate-900/90 backdrop-blur-md z-50"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
      >
        <LoaderContent />
      </motion.div>
    )
  }
  
  return <LoaderContent />
}

export default Loader
